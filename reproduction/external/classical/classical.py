"""Prediction-only classical postprocessors. CHOMP/SCP are explicit 2D adaptations.

CHOMP-style: covariant-gradient optimization with a temporal quadratic metric
and simultaneous time-indexed inter-agent distance potentials.
SCP/TrajOpt-style: sequential convexification of distance hinge costs, sparse
ADMM convex subproblems and a trust region. Neither is an unmodified robotics
benchmark reproduction. ORCA calls pinned upstream RVO2 through a small C ABI.
"""
import ctypes,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import factorized

DT=.1

@lru_cache(maxsize=8)
def temporal(T,weight):
    D=np.zeros((T-1,T))
    for k in range(T-1):
        D[k,k]=-2/DT**2;D[k,k+1]=1/DT**2
        if k:D[k,k-1]=1/DT**2
    H=np.eye(T)+weight*(D.T@D)
    return D,H,linalg.cho_factor(H,check_finite=False)

def terms(anchor,current,weight):
    N,T,_=anchor.shape;D,H,chol=temporal(T,weight)
    const=np.zeros((N,T-1,2));const[:,0]=current/DT**2
    b=anchor-weight*np.einsum('ts,ntc->nsc',D,const)
    return D,H,chol,const,b

def pair_data(x,current):
    i,j=np.triu_indices(len(x),1)
    r=x[i]-x[j];dist=np.linalg.norm(r,axis=-1)
    unit=r/np.maximum(dist[...,None],1e-10)
    # A deterministic observed-side direction resolves coincident centers.
    for edge,k in np.argwhere(dist<1e-10):
        d=current[i[edge]]-current[j[edge]]
        if np.linalg.norm(d)<1e-10:d=np.array([1.,0.])
        unit[edge,k]=d/np.linalg.norm(d)
    return i,j,dist,unit

def cost_grad(x,anchor,current,cfg,penalty='squared',gradient=True):
    D,H,chol,const,b=terms(anchor,current,cfg['smoothness'])
    accel=np.einsum('ts,nsc->ntc',D,x)+const
    value=.5*np.square(x-anchor).sum()+.5*cfg['smoothness']*np.square(accel).sum()
    g=x-anchor+cfg['smoothness']*np.einsum('ts,ntc->nsc',D,accel)
    i,j,dist,unit=pair_data(x,current);short=np.maximum(1.-dist,0.)
    if penalty=='squared':value+=.5*cfg['collision_weight']*np.square(short).sum();mag=short
    else:value+=cfg['collision_weight']*short.sum();mag=(short>0).astype(float)
    if gradient and len(i):
        contribution=-cfg['collision_weight']*mag[...,None]*unit
        np.add.at(g,i,contribution);np.add.at(g,j,-contribution)
    return float(value),g

def covariant(anchor,current,cfg):
    x=anchor.copy();N,T,_=x.shape;D,H,chol,const,b=terms(anchor,current,cfg['smoothness'])
    value,g=cost_grad(x,anchor,current,cfg);calls=1;accepted=0;begin=time.perf_counter()
    for iteration in range(cfg['max_iterations']):
        rhs=g.transpose(1,0,2).reshape(T,-1)
        direction=-linalg.cho_solve(chol,rhs,check_finite=False).reshape(T,N,2).transpose(1,0,2)
        if np.max(np.abs(direction))<1e-6:break
        directional=float((g*direction).sum());step=1.;ok=False
        for _ in range(16):
            trial=x+step*direction;v,_=cost_grad(trial,anchor,current,cfg,gradient=False);calls+=1
            if v<=value+1e-4*step*directional:
                x=trial;accepted+=1;ok=True;break
            step*=.5
        if not ok:break
        new,g=cost_grad(x,anchor,current,cfg);calls+=1
        if abs(value-new)<1e-8*max(1.,value):value=new;break
        value=new
    assert np.isfinite(x).all()
    return x,dict(iterations=iteration+1,accepted_steps=accepted,objective_calls=calls,
        final_objective=value,seconds=time.perf_counter()-begin)

def convex_step(Q,q,A,offset,penalty,radius,cfg):
    """min .5 d'Qd+q'd+penalty sum hinge(offset-A d), |d|<=radius."""
    size=len(q);rho=cfg['admm_rho'];M=A.shape[0]
    if not M:
        candidate=factorized(Q.tocsc())(-q)
        if np.max(np.abs(candidate))<=radius:
            return candidate,dict(iterations=1,converged=True,primal_residual=0.,dual_residual=0.)
    AtA=(A.T@A).tocsc();identity=sparse.eye(size,format='csc')
    solve=factorized((Q+rho*AtA+rho*identity).tocsc())
    delta=np.zeros(size);z=np.zeros(M);u=np.zeros(M);box=np.zeros(size);v=np.zeros(size)
    converged=False
    for k in range(cfg['admm_max_iterations']):
        delta=solve(-q+rho*A.T@(z-u)+rho*(box-v))
        ad=A@delta;zz=z.copy();bb=box.copy()
        proposed=ad+u;z=np.where(proposed>=offset,proposed,np.minimum(offset,proposed+penalty/rho))
        box=np.clip(delta+v,-radius,radius)
        u+=ad-z;v+=delta-box
        primal=max(np.linalg.norm(ad-z,np.inf) if M else 0.,np.linalg.norm(delta-box,np.inf))
        dual=rho*max(np.linalg.norm(A.T@(z-zz),np.inf),np.linalg.norm(box-bb,np.inf))
        if max(primal,dual)<cfg['admm_tolerance']:converged=True;break
        # Standard residual balancing changes solver conditioning, not the QP.
        if (k+1)%25==0:
            newrho=rho
            if primal>10*dual:newrho=min(1000.,rho*2)
            elif dual>10*primal:newrho=max(1e-4,rho*.5)
            if newrho!=rho:
                u*=rho/newrho;v*=rho/newrho;rho=newrho
                solve=factorized((Q+rho*AtA+rho*identity).tocsc())
    return box,dict(iterations=k+1,converged=converged,primal_residual=float(primal),dual_residual=float(dual))

def sequential_convex(anchor,current,cfg):
    x=anchor.copy();N,T,_=x.shape;_,H,_,_,b=terms(anchor,current,cfg['smoothness'])
    Q=sparse.kron(sparse.eye(N),sparse.kron(sparse.csc_matrix(H),sparse.eye(2)),format='csc')
    radius=cfg['trust_region'];value,_=cost_grad(x,anchor,current,cfg,'linear');calls=1;accepted=0;inner=[];begin=time.perf_counter()
    for iteration in range(cfg['max_iterations']):
        i,j,dist,unit=pair_data(x,current)
        edges,times=np.where(dist<=1.+2*np.sqrt(2)*radius+1e-10)
        rr=[];cc=[];vv=[]
        for m,(e,t) in enumerate(zip(edges,times)):
            for coord in (0,1):
                rr.extend((m,m));cc.extend(((i[e]*T+t)*2+coord,(j[e]*T+t)*2+coord));vv.extend((unit[e,t,coord],-unit[e,t,coord]))
        A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(edges),N*T*2)).tocsc()
        q=np.einsum('ts,nsc->ntc',H,x).ravel()-b.ravel()
        delta,info=convex_step(Q,q,A,1.-dist[edges,times],cfg['collision_weight'],radius,cfg);inner.append(info)
        trial=x+delta.reshape(x.shape);new,_=cost_grad(trial,anchor,current,cfg,'linear',False);calls+=1
        if new<value-1e-9:
            improvement=value-new;x=trial;value=new;accepted+=1
            radius=min(1.,radius*1.5)
            if np.max(np.abs(delta))<1e-5 or improvement<1e-7*max(1.,value):break
        else:
            radius*=.5
            if radius<1e-3:break
    assert np.isfinite(x).all()
    return x,dict(iterations=iteration+1,accepted_steps=accepted,objective_calls=calls,
        final_objective=value,convex_subproblems=len(inner),subproblem_converged=sum(z['converged'] for z in inner),
        max_primal_residual=max((z['primal_residual'] for z in inner),default=0),
        max_dual_residual=max((z['dual_residual'] for z in inner),default=0),seconds=time.perf_counter()-begin)

@lru_cache(maxsize=1)
def orca_library():
    library=ctypes.CDLL(str(Path(__file__).resolve().parent/'liborca_adapter.so'))
    pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    library.orca_rollout.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer,pointer,ctypes.c_double,pointer]
    library.orca_rollout.restype=ctypes.c_int
    return library

def orca(anchor,current,velocity,cfg):
    begin=time.perf_counter();output=np.zeros_like(anchor,dtype=np.float64)
    if len(anchor)==1:return anchor.copy(),dict(iterations=60,seconds=time.perf_counter()-begin)
    code=orca_library().orca_rollout(len(anchor),anchor.shape[1],np.ascontiguousarray(current),np.ascontiguousarray(velocity),
        np.ascontiguousarray(anchor),cfg['time_horizon'],output)
    assert code==0 and np.isfinite(output).all()
    return output,dict(iterations=60,seconds=time.perf_counter()-begin,numerical_degeneracies=int(orca_library().orca_degeneracies()))

def run(method,row,cfg):
    # Deliberately construct the predictor inputs from observed/frozen fields only.
    current=row['current'].double().numpy();origin=current[0].copy()
    modes=row['pi'].argmax(-1);anchor=row['world'].double().numpy()[np.arange(len(modes)),modes.numpy()]
    local=anchor-origin;observed=current-origin
    if method=='ORCA':output,details=orca(local,observed,row['history_velocity'][:,-1,:].double().numpy(),cfg)
    elif method=='CHOMP_style':output,details=covariant(local,observed,cfg)
    elif method=='SCP_style':output,details=sequential_convex(local,observed,cfg)
    else:raise ValueError(method)
    return output+origin,details
