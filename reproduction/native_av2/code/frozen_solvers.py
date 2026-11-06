import itertools,math,time
import numpy as np
import torch

def links_for(n,edges):
    links=[[] for _ in range(n)]
    for e,(i,j) in enumerate(edges):links[i].append((e,j,True));links[j].append((e,i,False))
    return links

def conditional(pi,c,links,m,i,base):
    s=pi[i]-pi[i,base[i]]
    for e,j,left in links[i]:
        v=c[e,:,m[j]] if left else c[e,m[j],:]
        s=s+(v-v[base[i]])
    return s

def energy(pi,c,edges,m):
    return float(pi[np.arange(len(m)),m].sum()+sum(c[e,m[i],m[j]] for e,(i,j) in enumerate(edges)))

def geometry_arrays(t,edges,n):
    lookup={tuple(p):i for i,p in enumerate(t['pairs'])}
    ps=[];hits=[];steps=[];penalty=np.zeros((len(edges),6,6))
    for e,(i,j) in enumerate(edges):
        if (i,j) in lookup:
            k=lookup[(i,j)];ps.append((i,j));hits.append(t['any'][k]);steps.append(t['steps'][k]);penalty[e]=-t['any'][k].astype(float)
    ps=np.asarray(ps,dtype=int).reshape(-1,2);hits=np.asarray(hits,dtype=bool).reshape(-1,6,6);steps=np.asarray(steps,dtype=int).reshape(-1,6,6)
    return {'pairs':ps,'any':hits,'steps':steps,'links':links_for(n,ps)},penalty

def hard_anchor(g,base):
    p=g['pairs'];ix=np.arange(len(p));m=np.asarray(base)
    return {'any':g['any'][ix,m[p[:,0]],m[p[:,1]]], 'step_limit':int(g['steps'][ix,m[p[:,0]],m[p[:,1]]].sum())}

def feasible_k(g,anchor,m,i,current_steps):
    ok=np.ones(6,dtype=bool);oldsteps=newsteps=0
    for e,j,left in g['links'][i]:
        h=g['any'][e,:,m[j]] if left else g['any'][e,m[j],:]
        z=g['steps'][e,:,m[j]] if left else g['steps'][e,m[j],:]
        if not anchor['any'][e]:ok&=~h
        oldsteps+=int(z[m[i]]);newsteps=newsteps+z
    ok&=current_steps-oldsteps+newsteps<=anchor['step_limit']
    assert ok[m[i]]
    return ok

def total_steps(g,m):
    m=np.asarray(m);p=g['pairs'];return int(g['steps'][np.arange(len(p)),m[p[:,0]],m[p[:,1]]].sum())

def solve(pi,c,edges,allowed,base,order=None,initial=None,hard=None,max_sweeps=8):
    start=time.perf_counter();n=len(base);links=links_for(n,edges);m=list(base if initial is None else initial);order=list(range(n)) if order is None else order
    blocked=np.zeros(n,dtype=int);updates=np.zeros(n,dtype=int);reject=np.zeros(n,dtype=int);trace=[];changed_total=0;converged=False;sweep=0
    anchor=hard_anchor(hard,base) if hard is not None else None
    for sweep in range(1,max_sweeps+1):
        changes=0;before=energy(pi,c,edges,m)
        for i in order:
            s=conditional(pi,c,links,m,i,base);unrestricted=int(s.argmax());blocked[i]+=int(not allowed[i,unrestricted]);ok=allowed[i].copy()
            if hard is not None:
                feas=feasible_k(hard,anchor,m,i,total_steps(hard,m));reject[i]+=int(np.sum(ok&~feas));ok&=feas
            s=np.where(ok,s,-np.inf);k=int(s.argmax())
            if k!=m[i]:assert s[k]>=s[m[i]]-1e-10;m[i]=k;changes+=1;updates[i]+=1
        after=energy(pi,c,edges,m);assert after>=before-1e-8,(before,after)
        trace.append(after-before);changed_total+=changes
        if not changes:converged=True;break
    assert np.all(allowed[np.arange(n),m])
    residual=[];margin=[]
    for i in range(n):
        s=conditional(pi,c,links,m,i,base);ok=allowed[i].copy()
        if hard is not None:ok&=feasible_k(hard,anchor,m,i,total_steps(hard,m))
        s=np.where(ok,s,-np.inf);residual.append(float(s.max()-s[m[i]]));other=s.copy();other[m[i]]=-np.inf;margin.append(float(s[m[i]]-other.max()) if np.isfinite(other.max()) else None)
    if hard is not None:
        p=hard['pairs'];mm=np.asarray(m);hit=hard['any'][np.arange(len(p)),mm[p[:,0]],mm[p[:,1]]]
        assert not (hit&~anchor['any']).any() and total_steps(hard,m)<=anchor['step_limit']
    return m,{'sweeps':sweep,'converged':converged,'seconds':time.perf_counter()-start,'accepted_updates':changed_total,'node_updates':updates.tolist(),'unconstrained_winner_blocked_by_CJM':blocked.tolist(),'hard_rejected_candidate_evaluations':reject.tolist(),'single_node_residual_gain':residual,'final_winner_margin':margin,'sweep_energy_gains':trace,'final_energy':energy(pi,c,edges,m)}

def project(v,r):
    a=np.array(r['base']);n=len(a);ii=np.arange(n);w=v['m1']['world'].double().numpy();base=w[ii,a];diff=w-base[:,None];dist=np.linalg.norm(diff,axis=-1).max(-1);alpha=np.minimum(1.,(4.-1e-8)/np.maximum(dist,1e-12));new=base[:,None]+alpha[:,:,None,None]*diff;vv=dict(v);vv['m1']=dict(v['m1'],world=torch.as_tensor(new,dtype=torch.double));t=tables(vv);C=r['C_old'].double().numpy();CP=np.zeros_like(C)
    for e,(i,j) in enumerate(r['edges']):
        x,y=alpha[i,:,None],alpha[j,None,:];c00=C[e,a[i],a[j]];c10=C[e,:,a[j]][:,None];c01=C[e,a[i],:][None,:];CP[e]=c00+x*(c10-c00)+y*(c01-c00)+x*y*(C[e]-c10-c01+c00)
    unary=r['pi'][ii,a,None]+alpha*(r['pi']-r['pi'][ii,a,None]);rr=dict(r,pi=unary,C_old=torch.as_tensor(CP,dtype=torch.double),geometry={k:t[k] for k in ('pairs','any','steps')});assert np.linalg.norm(new-base[:,None],axis=-1).max()<=4.
    return vv,rr,alpha

def select(v,r,kind='main'):
    a=np.array(r['base']);n=len(a);w=v['m1']['world'].double().numpy();distance=np.linalg.norm(w-w[np.arange(n),a,None],axis=-1).max(-1);allowed=r['allowed']&(distance<=4.);options=[np.flatnonzero(z).tolist() for z in allowed];count=math.prod(map(len,options));C=r['C_old'].double().numpy() if kind=='main' else np.zeros(tuple(r['C_old'].shape),dtype=float);t=r['geometry'];p=t['pairs'];ix=np.arange(len(p));inside=np.array([tuple(q) in set(r['edges']) for q in p],dtype=bool);ahit=t['any'][ix,a[p[:,0]],a[p[:,1]]];asteps=t['steps'][ix,a[p[:,0]],a[p[:,1]]]
    def evaluate(cc):
        cc=np.asarray(cc,dtype=int).reshape(-1,n);hit=t['any'][ix[None,:],cc[:,p[:,0]],cc[:,p[:,1]]];steps=t['steps'][ix[None,:],cc[:,p[:,0]],cc[:,p[:,1]]];mask=~(hit&~ahit).any(1)&(steps.sum(1)<=asteps.sum())&(steps[:,inside].sum(1)<=asteps[inside].sum());un=r['pi'][np.arange(n)[None,:],cc].sum(1);score=un.copy()
        for e,(i,j) in enumerate(r['edges']):score+=C[e,cc[:,i],cc[:,j]]
        return mask,hit.sum(1),steps.sum(1),score
    best=None;bestm=None;feasible=0;tested=0;exact=count<=65536
    if exact:
        it=itertools.product(*options)
        while True:
            cc=list(itertools.islice(it,4096))
            if not cc:break
            cc=np.array(cc,dtype=int);mask,hit,steps,s=evaluate(cc);feasible+=int(mask.sum());tested+=len(cc)
            if mask.any():
                good=np.flatnonzero(mask);order=np.lexsort((good,-s[good],steps[good],hit[good]));k=int(good[order[0]]);key=(int(hit[k]),int(steps[k]),-float(s[k]))
                if best is None or key<best:best=key;bestm=cc[k].tolist()
    else:
        bestm=a.tolist();_,hit,steps,s=evaluate([bestm]);best=(int(hit[0]),int(steps[0]),-float(s[0]))
        for sweep in range(8):
            change=False
            for i in range(n):
                cc=np.array([bestm.copy() for k in options[i]],int);cc[:,i]=options[i];mask,hit,steps,s=evaluate(cc);tested+=len(cc);good=np.flatnonzero(mask);assert len(good);order=np.lexsort((good,-s[good],steps[good],hit[good]));k=int(good[order[0]]);key=(int(hit[k]),int(steps[k]),-float(s[k]))
                if key<best:bestm=cc[k].tolist();best=key;change=True
            if not change:break
    assert bestm is not None and allowed[np.arange(n),bestm].all();mask,hit,steps,s=evaluate([bestm]);assert mask[0]
    return bestm,{'exact':exact,'product':count,'evaluated':tested,'feasible_combinations':feasible if exact else None,'objective':list(best),'max_path_shift':float(distance[np.arange(n),bestm].max()),'fallback_to_B':False}
