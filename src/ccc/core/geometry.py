"""Explicit offline metrics. No score selection or inference inputs use GT.

Primary geometry: 2D predicted-target vs predicted-target oriented boxes at10Hz,
using observed dimensions and trajectory-tangent headings. This is a custom
overlap proxy, NOT the official Waymo overlap metric or a collision guarantee.
"""
import itertools
import numpy as np
def asnp(v):
    return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
def headings(paths,current,initial):
    w=np.asarray(paths,dtype=np.float64);previous=np.concatenate([current[:,None,:],w[:,:-1]],axis=1);delta=w-previous
    angle=np.arctan2(delta[...,1],delta[...,0]);moving=np.linalg.norm(delta,axis=-1)>=0.05
    carry=np.asarray(initial,dtype=np.float64).copy()
    for t in range(w.shape[1]):carry=np.where(moving[:,t],angle[:,t],carry);angle[:,t]=carry
    return angle
def boxes_overlap(a,angle_a,size_a,b,angle_b,size_b):
    """Separating axis test, strict positive penetration >1e-6m on all axes."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    ua=np.stack([np.cos(angle_a),np.sin(angle_a)],-1);va=np.stack([-np.sin(angle_a),np.cos(angle_a)],-1)
    ub=np.stack([np.cos(angle_b),np.sin(angle_b)],-1);vb=np.stack([-np.sin(angle_b),np.cos(angle_b)],-1)
    d=b-a;inside=np.ones(d.shape[:-1],dtype=bool)
    for axis in (ua,va,ub,vb):
        ra=size_a[...,0]/2*np.abs((ua*axis).sum(-1))+size_a[...,1]/2*np.abs((va*axis).sum(-1))
        rb=size_b[...,0]/2*np.abs((ub*axis).sum(-1))+size_b[...,1]/2*np.abs((vb*axis).sum(-1))
        inside&=np.abs((d*axis).sum(-1)) < ra+rb-1e-6
    return inside
def observed(a,history):
    ids=asnp(a['original_agent_id']).astype(int);p=asnp(history['position'])[ids,10,:2];h=asnp(history['heading'])[ids,10]
    length=asnp(history['length']);width=asnp(history['width']);length=length[ids,10] if length.ndim==2 else length[ids];width=width[ids,10] if width.ndim==2 else width[ids]
    size=np.stack([length,width],-1);kind=asnp(history['type'])[ids];ok=np.isfinite(size).all(-1)&(size>0).all(-1)
    pairs=np.array([(i,j) for i,j in itertools.combinations(range(len(ids)),2) if ok[i] and ok[j]],dtype=np.int64).reshape(-1,2)
    return p,h,size,kind,pairs,int((~ok).sum())
def selected_geometry(world,current,initial,size,pairs):
    w=np.asarray(world,dtype=np.float64);angles=headings(w,current,initial)
    if not len(pairs):return np.zeros((0,w.shape[1]),dtype=bool),angles
    i,j=pairs.T
    overlap=boxes_overlap(w[i],angles[i],size[i,None,:],w[j],angles[j],size[j,None,:])
    return overlap,angles
def pair_tables(a,history):
    world=asnp(a['world']);current,h,size,kind,pairs,bad=observed(a,history);n,k,t,_=world.shape
    all_angles=np.stack([headings(world[:,m],current,h) for m in range(k)],1)
    table=np.zeros((len(pairs),k,k),dtype=np.float32)
    for e,(i,j) in enumerate(pairs):
        overlap=boxes_overlap(world[i,:,None],all_angles[i,:,None],size[i,None,None,None,:],world[j,None,:],all_angles[j,None,:],size[j,None,None,None,:])
        table[e]=overlap.any(-1)
    return pairs,table
def scene_metrics(a,history,choices):
    w=asnp(a['world']).astype(np.float64);gt=asnp(a['gt']).astype(np.float64);valid=asnp(a['valid']).astype(bool);valid8=asnp(a['valid8']).astype(bool)
    n=len(w);idx=np.arange(n);old=asnp(a['pi']).argmax(-1);base=w[idx,old];current,h,size,kind,pairs,bad=observed(a,history)
    base_overlap,_=selected_geometry(base,current,h,size,pairs);result={}
    for name,modes in choices.items():
        modes=np.asarray(modes,dtype=np.int64);assert modes.shape==(n,) and ((modes>=0)&(modes<6)).all()
        selected=w[idx,modes];error=np.linalg.norm(selected-gt,axis=-1)
        ade=(error*valid).sum(-1)/np.maximum(1,valid.sum(-1));fde=error[:,-1];base_error=np.linalg.norm(base[:,-1]-gt[:,-1],axis=-1);delta=fde-base_error
        overlap,_=selected_geometry(selected,current,h,size,pairs);any_=overlap.any(-1);old_any=base_overlap.any(-1)
        relative_ade=relative_fde=0.;rel_pairs=0;vehicles=vehicle_overlaps=0
        for e,(i,j) in enumerate(pairs):
            both=valid[i]&valid[j]
            if valid8[i] and valid8[j] and both.sum()>=60:
                r=(selected[i]-selected[j])-(gt[i]-gt[j]);err=np.linalg.norm(r,axis=-1)
                relative_ade+=float(err[both].mean());relative_fde+=float(err[-1]);rel_pairs+=1
            if kind[i]==kind[j]==1:vehicles+=1;vehicle_overlaps+=int(any_[e])
        result[name]=dict(targets=int(valid8.sum()),fde_sum=float(fde[valid8].sum()),ade_sum=float(ade[valid8].sum()),
          changed=int((modes[valid8]!=old[valid8]).sum()),improved=int((delta[valid8]<-1e-6).sum()),degraded=int((delta[valid8]>1e-6).sum()),
          severe=int((delta[valid8]>20+1e-6).sum()),worst_delta=float(max(0.,delta[valid8].max())) if valid8.any() else 0.,
          overlap_pairs=int(any_.sum()),eligible_pairs=len(pairs),overlap_pair_steps=int(overlap.sum()),eligible_pair_steps=len(pairs)*80,
          overlap_scenes=int(any_.any()),eligible_scenes=int(len(pairs)>0),new_overlap_pairs=int((any_&~old_any).sum()),resolved_overlap_pairs=int((~any_&old_any).sum()),
          relative_ade_sum=relative_ade,relative_fde_sum=relative_fde,relative_pairs=rel_pairs,
          vehicle_overlap_pairs=vehicle_overlaps,vehicle_pairs=vehicles,invalid_observed_dimensions=bad)
    return result
def aggregate(records):
    names=list(records[0]['methods']);out={}
    for name in names:
        sums={k:sum(r['methods'][name][k] for r in records) for k in records[0]['methods'][name] if k!='worst_delta'}
        def ratio(num,den):return sums[num]/sums[den] if sums[den] else None
        out[name]=dict(sums, FDE=ratio('fde_sum','targets'),ADE=ratio('ade_sum','targets'),
          predicted_pair_overlap_rate=ratio('overlap_pairs','eligible_pairs'),pair_time_overlap_rate=ratio('overlap_pair_steps','eligible_pair_steps'),
          predicted_scene_overlap_rate=ratio('overlap_scenes','eligible_scenes'),relative_ADE=ratio('relative_ade_sum','relative_pairs'),relative_FDE=ratio('relative_fde_sum','relative_pairs'),
          worst_delta=max(r['methods'][name]['worst_delta'] for r in records))
    return out
def bootstrap(records,new,ref,num,den,reps=10000,seed=42):
    x=np.array([[r['methods'][m][key] for m in (new,ref) for key in (num,den)] for r in records],dtype=np.float64)
    def delta(z):return z[...,0]/np.maximum(z[...,1],1)-z[...,2]/np.maximum(z[...,3],1)
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,reps,32):
        ix=rng.integers(0,len(x),size=(min(32,reps-start),len(x)));draws.extend(delta(x[ix].sum(1)).tolist())
    return {'delta':float(delta(x.sum(0))),'ci95':np.quantile(draws,[.025,.975]).tolist(),'scene_resamples':reps,'seed':seed}
def self_test():
    z=np.array([0.]);size=np.array([[4.,2.]])
    assert boxes_overlap(np.array([[0.,0.]]),z,size,np.array([[1.,0.]]),z,size).item()
    assert not boxes_overlap(np.array([[0.,0.]]),z,size,np.array([[4.,0.]]),z,size).item()
    assert not boxes_overlap(np.array([[0.,0.]]),z,size,np.array([[0.,3.]]),z,size).item()
    assert boxes_overlap(np.array([[0.,0.]]),z,size,np.array([[0.,2.5]]),z+np.pi/2,size).item()
    p=np.zeros((2,80,2));p[0,:,0]=np.linspace(-8,8,80);p[1,:,1]=np.linspace(-8,8,80)
    current=np.array([[-8.,0.],[0.,-8.]]);initial=np.array([0.,np.pi/2]);sizes=np.array([[4.,2.],[4.,2.]]);pairs=np.array([[0,1]])
    overlap,_=selected_geometry(p,current,initial,sizes,pairs);assert overlap.any()
    swapped,_=selected_geometry(p[::-1],current[::-1],initial[::-1],sizes[::-1],pairs);assert np.array_equal(overlap,swapped)
    # Same spatial crossing, temporally separated traversal.
    shifted=p.copy();shifted[1,:60,1]=-8.;shifted[1,60:,1]=np.linspace(-8,8,20)
    late,_=selected_geometry(shifted,current,initial,sizes,pairs);assert not late.any()
    angles=headings(np.zeros((1,80,2)),np.zeros((1,2)),np.array([1.2]));assert np.allclose(angles,1.2)
    return {'pass':True,'SAT_axis_aligned_and_rotated':True,'same_time_crossing':True,'different_time_crossing':True,'agent_swap':True,'stationary_heading':True}
