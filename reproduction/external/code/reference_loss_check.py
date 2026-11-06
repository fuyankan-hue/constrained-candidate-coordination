"""Check the ported loss against the unmodified author training_step AST.

The author loss receives labels, as supervised training requires. Its forward
is supplied with previously computed prediction-only network outputs.
"""
import ast,importlib.util,types
import torch
import torch.nn.functional as F
from baseline_data import ROOT

class Bag(dict):
    def __getattr__(self,name):
        try:return self[name]
        except KeyError:raise AttributeError(name)
    def __setattr__(self,name,value):self[name]=value
    def to(self,device):return self

def loss_class(method,filename,name):
    p=ROOT/'sources'/method/'losses'/filename
    spec=importlib.util.spec_from_file_location('author_loss_'+method+'_'+name,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return getattr(module,name)

def check(model,args,labels,info):
    method=model.method
    path=ROOT/'sources'/method/('models/refine.py' if method=='SmartRefine' else 'refine/av2_refine.py')
    tree=ast.parse(path.read_text())
    target=next(n for cls in tree.body if isinstance(cls,ast.ClassDef)
                for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='training_step')
    ns={'torch':torch,'F':F}
    if method=='SRefiner':
        from torch_geometric.utils import unbatch
        ns['unbatch']=unbatch
    ast.fix_missing_locations(target)
    exec(compile(ast.Module(body=[target],type_ignores=[]),str(path),'exec'),ns)
    from baseline_model import objective
    model.eval()
    with torch.no_grad():
        stages,heads=model(args,info);ours=objective(model,args,labels,info)
    base=info['local'].transpose(0,1)
    deltas=[];previous=base
    for positions,scale in stages:
        deltas.append(torch.cat((positions-previous,scale),-1));previous=positions
    n=len(labels['gt']);device=labels['gt'].device
    d=Bag(agent_index=torch.arange(n,device=device),num_graphs=n,
          padding_mask=torch.cat((torch.zeros((n,50),device=device,dtype=torch.bool),~labels['valid']),-1),
          y=labels['gt'])
    class Facade:
        def __call__(self,*unused):return self.result
        def forward(self,*unused):return self.result
        def log(self,*unused,**kw):pass
        def to_global_coord(self,*unused):pass
    facade=Facade();facade.refine_num=model.iterations;facade.future_steps=60;facade.cls_temperture=1
    facade.reg_loss=loss_class(method,'laplace_nll_loss.py','LaplaceNLLLoss')()
    if method=='SmartRefine':
        facade.cls_loss=loss_class(method,'soft_target_cross_entropy_loss.py','SoftTargetCrossEntropyLoss')()
        facade.score_loss=loss_class(method,'score_reg_l1_loss.py','ScoreRegL1Loss')()
        facade.result=(torch.cat((base,base),-1),None,deltas,heads)
        reference=ns['training_step'](facade,(d,None),0)
    else:
        facade.device=device
        d.update(y_hat_final=info['local'],embeds_final=torch.zeros((n,6,128),device=device),
                 x_scored=torch.ones(n,device=device,dtype=torch.bool),
                 x_padding_mask=d.padding_mask.clone(),y_gt_agentcoords=labels['gt'],
                 scenario_id=list(range(len(info['nlist']))),
                 batch=torch.repeat_interleave(torch.arange(len(info['nlist']),device=device),torch.tensor(info['nlist'],device=device)))
        facade.result=(torch.cat((info['local'],info['local']),-1),None,deltas,heads)
        reference=ns['training_step'](facade,d)
    difference=abs(float(ours)-float(reference))
    assert difference<=max(1e-5,2e-5*abs(float(reference))),('author loss mismatch',method,float(ours),float(reference),difference)
    return dict(author_loss=float(reference),ported_loss=float(ours),absolute_difference=difference,
                checked_against='unmodified official training_step AST',source=str(path.relative_to(ROOT)))
