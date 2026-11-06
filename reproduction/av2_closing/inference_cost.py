"""Isolated frozen AV2 inference benchmark. No training or metric selection.

Each method runs in its own process on the same GPU and fixed FIT scenes.
All numerical kernels, weights, constraints and neural inputs are unchanged.
Only unrelated evaluation branches and GT label values are omitted.
"""
import argparse, gc, hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/workspace/output');OUT=Path('/workspace/checks')
sys.path.insert(0,str(ROOT/'code'))
from av2_core import point_tables,refine
from frozen_solvers import solve,geometry_arrays,project,select
from av2_pipeline import load_rcim,load_head
from av2_refiner import collate

METHODS=('NDPNet','NDPNet_refined','OldRCIM_refined','GeometryB_two_refiners','Full_without_RCIM','FullRCIM')
FIELDS=('scene','original_agent_id','context_ids','world','pi','near','far','refine',
        'current','heading','history_position','history_velocity','history_heading',
        'history_valid','observed_type','is_focal')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def check_frozen():
    z=json.loads((ROOT/'MODELS_FROZEN.json').read_text())
    for rel,digest in z['sha256'].items():assert sha(ROOT/rel)==digest,rel
    return sha(ROOT/'MODELS_FROZEN.json')


def configure():
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1


def prediction_row(row):
    p={k:row[k] for k in FIELDS if k in row};n=len(row['pi'])
    # Frozen collation allocates label placeholders; no real labels are passed.
    p['gt']=torch.zeros(n,60,2);p['valid']=torch.zeros(n,60,dtype=torch.bool)
    return p


class Isolated:
    def __init__(self,method):
        assert method in METHODS;self.method=method;self.scorer=None;self.heads={}
        if method in ('OldRCIM_refined','FullRCIM'):self.scorer=load_rcim()
        arm={'NDPNet_refined':'A','OldRCIM_refined':'E','GeometryB_two_refiners':'B',
             'Full_without_RCIM':'E','FullRCIM':'E'}.get(method)
        if arm:self.heads['I1']=load_head(42,'I1',arm,'cuda')
        second={'GeometryB_two_refiners':'B','Full_without_RCIM':'NoRCIM','FullRCIM':'RCIM'}.get(method)
        if second:self.heads['I2']=load_head(42,'I2',second,'cuda')

    @torch.inference_mode()
    def coordinate(self,row):
        method=self.method;n=len(row['pi']);a=row['pi'].argmax(-1).tolist();ii=torch.arange(n)
        w=row['world'].double()
        if method in ('NDPNet','NDPNet_refined'):return w[ii,a].numpy(),a,a
        edges=[(i,j) for i in range(n) for j in range(i+1,n) if float((row['current'][i]-row['current'][j]).norm())<75.]
        allowed=(w-w[ii,a,None]).norm(dim=-1).amax(-1)<=20.
        C=(self.scorer(row['near'],row['far'],row['refine'],row['world'],torch.tensor(edges,dtype=torch.long).reshape(-1,2)).cpu()
           if self.scorer is not None else torch.zeros(len(edges),6,6))
        r=dict(pi=row['pi'].double().numpy(),base=a,edges=edges,allowed=allowed.numpy(),C_old=C)
        v={'m1':{'world':w}}
        if method in ('GeometryB_two_refiners','OldRCIM_refined'):
            hard,G=geometry_arrays(point_tables(v),edges,n)
            c=.5*G if method=='GeometryB_two_refiners' else C.double().numpy()
            m,_=solve(r['pi'],c,edges,r['allowed'],a,hard=hard if method=='OldRCIM_refined' else None,max_sweeps=8)
            return w[ii,m].numpy(),m,a
        project.__globals__['tables']=point_tables
        vp,rp,alpha=project(v,r)
        m,_=select(vp,rp,'NoRCIM' if method=='Full_without_RCIM' else 'main')
        return vp['m1']['world'][ii,m].numpy(),m,a

    @torch.inference_mode()
    def predict(self,row):
        path,m,a=self.coordinate(row)
        if self.method=='NDPNet':return path,m
        first,_,_=refine(row,path,m,self.heads['I1'],collate,'cuda')
        if 'I2' not in self.heads:return first,m
        original=row['world'].double()[torch.arange(len(m)),a].numpy()
        radius=((np.asarray(m)==np.asarray(a)).astype(np.float32) if self.method=='GeometryB_two_refiners'
                else np.clip(5.-1e-4-np.linalg.norm(first-original,axis=-1).max(1),0.,1.).astype(np.float32))
        last,_,_=refine(row,first,m,self.heads['I2'],collate,'cuda',radius)
        return last,m


def prepare():
    configure();check_frozen()
    from frozen_features import Extractor
    files=(ROOT/'manifests/train_files.txt').read_text().splitlines()
    eligible=[i for i,n in enumerate(files) if int(hashlib.sha256(Path(n).stem.encode()).hexdigest()[:8],16)%10>=2]
    chosen=sorted(eligible,key=lambda i:hashlib.sha256(('av2-closing-latency:'+Path(files[i]).stem).encode()).hexdigest())[:256]
    rows={};groups={}
    for i in chosen:groups.setdefault(i//64,[]).append(i)
    for shard,indices in groups.items():
        z=torch.load(ROOT/('cache/train/%05d.pt'%shard),map_location='cpu')
        for i in indices:
            assert z['filenames'][i%64]==files[i]
            rows[i]=prediction_row(z['rows'][i%64])
    e=Extractor();torch.set_num_threads(1);pack=[]
    for at,i in enumerate(chosen):
        data,ids,meta,unused_labels=e.load('train',files[i])
        data=data.cpu();ids=ids.cpu();a=data['agent']
        for key in ('position','velocity','heading','abs_vel'):assert not a[key][:,50:].abs().any()
        assert not a['valid_mask'][:,50:].any() and not a['target'].abs().any()
        pack.append(dict(scene=Path(files[i]).stem,filename=files[i],row=rows[i],graph=data,ids=ids,meta=meta))
        if (at+1)%32==0:print('BENCHMARK_INPUTS',at+1,'/256',flush=True)
    assert len({x['scene'] for x in pack})==256
    tmp=OUT/'BENCHMARK_INPUTS.pt.tmp'
    torch.save(pack,tmp);os.replace(tmp,OUT/'BENCHMARK_INPUTS.pt')
    save(OUT/'BENCHMARK_PROTOCOL.json',dict(passed=True,scenes=256,scene_ids=[p['scene'] for p in pack],
         selection='First256sha256(av2-closing-latency:+scene) among original FIT hash remainders2..9; no labels/error selection',
         subset='FIT only; no validation timing selection',refiner_seed=42,RCIM_seed=42,warmup_scenes=20,repetitions=3,
         input_graph_future_fields_zero=True,real_GT_not_in_benchmark_inputs=True,
         input_file_sha256=sha(OUT/'BENCHMARK_INPUTS.pt'),frozen_manifest_sha256=check_frozen()))


def verify():
    configure();check_frozen()
    from av2_core import choose
    from av2_pipeline import final_paths
    rows=torch.load(OUT/'BENCHMARK_INPUTS.pt',map_location='cpu')[:32]
    scorer=load_rcim();heads={(s,a):load_head(42,s,a,'cuda') for s,arms in
                             [('I1',('A','E','B')),('I2',('RCIM','NoRCIM','B'))] for a in arms}
    references=[]
    for r in rows:
        row=r['row'];ch=choose(row,scorer);references.append((ch,final_paths(row,ch,heads,42,'cuda')))
    maxdiff={};maps={'NDPNet':'A','NDPNet_refined':'A','OldRCIM_refined':'E','GeometryB_two_refiners':'B',
                     'Full_without_RCIM':'NoRCIM','FullRCIM':'RCIM'}
    for method in METHODS:
        isolated=Isolated(method);error=0.
        for row,(ch,paths) in zip(rows,references):
            p,m=isolated.predict(row['row']);assert m==ch[maps[method]]['modes'],(method,row['scene'])
            d=float(np.max(np.abs(p-paths[method])));error=max(error,d);assert d<=1e-8,(method,d)
        maxdiff[method]=error;del isolated;gc.collect();torch.cuda.empty_cache()
    save(OUT/'ISOLATED_REPLAY_CHECK.json',dict(passed=True,scenes=32,all_candidate_choices_identical=True,
         maximum_path_difference_m=maxdiff,reference='Unmodified frozen choose()+final_paths() on exactly same prediction features',
         scorer_and_refiner_seed=42,frozen_manifest_sha256=check_frozen()))
    print('ISOLATED_METHOD_REPLAY_PASS',json.dumps(maxdiff),flush=True)


@torch.inference_mode()
def backbone_row(extractor,data,ids,meta,method):
    # Run the same backbone. Avoid copying unused hidden outputs for the baseline.
    extractor.hidden.clear();p=extractor.model(data);n=len(ids)
    logits=p['pi'][ids].cpu().float();c=meta['heading'][:,None,None].cos();s=meta['heading'][:,None,None].sin()
    if method=='NDPNet':
        a=logits.argmax(-1);local=p['loc_refine_pos'][ids,a.to(ids.device)].cpu().float()
        cc,ss=c[:,0],s[:,0]
        path=torch.stack([local[...,0]*cc-local[...,1]*ss,local[...,0]*ss+local[...,1]*cc],-1)+meta['current'][:,None]
        extractor.hidden.clear();return path.double().numpy()
    local=p['loc_refine_pos'][ids].cpu().float()
    world=torch.stack([local[...,0]*c-local[...,1]*s,local[...,0]*s+local[...,1]*c],-1)+meta['current'][:,None,None]
    hidden=extractor.hidden['refine'].reshape(int(data['agent']['num_nodes']),6,128)[ids].cpu().float()
    row=dict(meta,world=world,pi=logits,refine=hidden,near=torch.zeros_like(hidden),far=torch.zeros_like(hidden),
             gt=torch.zeros(n,60,2),valid=torch.zeros(n,60,dtype=torch.bool))
    extractor.hidden.clear();return row


def statistics(ms,targets):
    x=np.asarray(ms,float);n=np.asarray(targets,int)
    def part(v):
        return dict(samples=len(v),mean_ms=float(v.mean()),median_ms=float(np.median(v)),
                    p95_ms=float(np.percentile(v,95)),max_ms=float(v.max())) if len(v) else dict(samples=0)
    out=part(x);out['by_requested_targets']={'1':part(x[n==1]),'2-4':part(x[(n>=2)&(n<=4)]),'>=5':part(x[n>=5])}
    return out


def memory():
    return dict(peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)


def benchmark(method):
    configure();frozen=check_frozen();assert json.loads((OUT/'ISOLATED_REPLAY_CHECK.json').read_text())['passed']
    inputs=torch.load(OUT/'BENCHMARK_INPUTS.pt',map_location='cpu');model=Isolated(method)
    print('BENCHMARK_METHOD',method,'GPU',torch.cuda.get_device_name(0),flush=True)
    # Cached-prediction suffix, including CPU coordination, neural transfers and protection.
    for x in inputs[:20]:model.predict(x['row'])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();suffix=[];counts=[]
    for rep in range(3):
        for x in inputs:
            torch.cuda.synchronize();t=time.perf_counter();path,_=model.predict(x['row']);torch.cuda.synchronize()
            suffix.append((time.perf_counter()-t)*1000);counts.append(len(x['row']['pi']))
        print('CACHED_SUFFIX',method,rep+1,'/3',flush=True)
    suffix_memory=memory();del path;gc.collect();torch.cuda.empty_cache()
    from frozen_features import Extractor
    extractor=Extractor();torch.set_num_threads(1)
    # The existing CUDA scatter kernels can have numerical roundoff jitter.
    # No weights/precision/algorithm are tuned based on timing or performance.
    def end_to_end(x,timed=True):
        graph=x['graph'].clone() # input-copy scaffolding excluded from inference timing
        torch.cuda.synchronize();t=time.perf_counter()
        data=graph.cuda();ids=x['ids'].cuda()
        # Match historical backbone extraction: graph scatter reductions have
        # no deterministic CUDA implementation. Restore strict refinement mode
        # after the backbone rather than changing any method or weight.
        torch.use_deterministic_algorithms(False)
        try:row_or_path=backbone_row(extractor,data,ids,x['meta'],method)
        finally:torch.use_deterministic_algorithms(True)
        result=row_or_path if method=='NDPNet' else model.predict(row_or_path)[0]
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
        del data,ids,graph,row_or_path
        return result,elapsed
    # CUDA repeatability control on the common prediction output, no label access.
    control=[]
    for x in inputs[:4]:
        a,_=end_to_end(x);b,_=end_to_end(x);control.append(float(np.max(np.abs(a-b))))
    for x in inputs[:20]:end_to_end(x)
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();resident=torch.cuda.memory_allocated()/2**20
    torch.cuda.reset_peak_memory_stats();whole=[]
    for rep in range(3):
        for x in inputs:
            path,ms=end_to_end(x);whole.append(ms)
        print('END_TO_END',method,rep+1,'/3',flush=True)
    peak=memory();peak['resident_models_MiB']=resident;peak['incremental_peak_over_resident_MiB']=peak['peak_allocated_MiB']-resident
    out=dict(passed=True,method=method,seed=42,scenes=256,repetitions=3,samples=768,
         cached_prediction_suffix=statistics(suffix,counts),end_to_end_inference=statistics(whole,counts),
         cached_suffix_memory=suffix_memory,end_to_end_memory=peak,
         gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,cpu_threads=1,
         measured_scope='Synchronized wall time, batch1; GPU frozen backbone+required D2H outputs+isolated CPU coordination+refiner transfers+geometry protection. H2D graph transfer included. Disk IO, adapters, graph cloning,model load,metrics excluded.',
         baseline_scope='Native NDPNet baseline copies only logits and selected trajectory, no unused hidden-feature copies.',
         dummy_label_scaffolding='Frozen bank/collate allocate zero targets; no true GT/future validity provided; this scaffolding remains included.',
         hidden_feature_scope='Near/Far stages retained in backbone; zero unused Near/Far hidden input tensors replace unnecessary host copies; Refine host copy retained when used.',
         numerical_modes='Backbone scatter uses original extraction non-forced-determinism; strict deterministic refinement/coordination restored afterward. TF32 disabled for all methods; same fixed protocol.',
         exact_replay='Verified separately against full frozen evaluation on32same-feature scenes before timing.',
         fresh_backbone_repeat_max_path_difference_m=control,
         extra_trainable_parameters=(44449 if model.scorer is not None else 0)+sum(sum(p.numel() for p in m.parameters()) for m in model.heads.values()),
         backbone_parameters=sum(p.numel() for p in extractor.model.parameters()),
         frozen_manifest_sha256=frozen,inputs_sha256=sha(OUT/'BENCHMARK_INPUTS.pt'),
         measurement_records=dict(scene_ids=[x['scene'] for x in inputs],targets=counts,
              suffix_ms=suffix,end_to_end_ms=whole),completed_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    assert frozen==check_frozen();save(OUT/('method_cost/'+method+'.json'),out)


def main():
    begun=time.perf_counter();frozen=check_frozen()
    for mode in ('prepare','verify'):
        marker=OUT/('BENCHMARK_PROTOCOL.json' if mode=='prepare' else 'ISOLATED_REPLAY_CHECK.json')
        if not marker.exists():
            p=subprocess.run([sys.executable,'-u',__file__,'--'+mode]);assert p.returncode==0,mode
    for method in METHODS:
        marker=OUT/('method_cost/'+method+'.json')
        if marker.exists():assert json.loads(marker.read_text())['frozen_manifest_sha256']==frozen;continue
        p=subprocess.run([sys.executable,'-u',__file__,'--method',method]);assert p.returncode==0,method
    table={m:json.loads((OUT/('method_cost/'+m+'.json')).read_text()) for m in METHODS}
    assert frozen==check_frozen()
    save(OUT/'INFERENCE_COST.json',dict(passed=True,methods=table,protocol=json.loads((OUT/'BENCHMARK_PROTOCOL.json').read_text()),
         numerical_replay=json.loads((OUT/'ISOLATED_REPLAY_CHECK.json').read_text()),frozen_files_unchanged=True,
         code_sha256=sha(__file__),seconds_this_session=time.perf_counter()-begun,
         interpretation='Single fixed FIT sample,seed42 reference weights. No benchmark-performance tuning; batch1 latency,not throughput or closed-loop planning rate.'))
    print('AV2_ISOLATED_INFERENCE_COST_COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--verify',action='store_true');ap.add_argument('--method',choices=METHODS);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.verify:verify()
    elif a.method:benchmark(a.method)
    else:main()
