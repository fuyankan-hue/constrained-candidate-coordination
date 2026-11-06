"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
import numpy as np

def global_xy(pred,pos,h):
    c=h.cos();s=h.sin()
    return torch.stack((pred[...,0]*c-pred[...,1]*s,pred[...,0]*s+pred[...,1]*c),dim=-1)+pos


def candidate_features(pred,pi):
    # Input is model-predicted LOCAL trajectories. No ground truth or future masks.
    prob=pi.log_softmax(-1)
    p3=pred[:,:,29,:2]
    p5=pred[:,:,49,:2]
    p8=pred[:,:,79,:2]
    travelled=torch.linalg.vector_norm(pred[:,:,1:,:2]-pred[:,:,:-1,:2],dim=-1).sum(-1)/80.0
    net3=torch.linalg.vector_norm(p3,dim=-1)
    net5=torch.linalg.vector_norm(p5,dim=-1)
    net8=torch.linalg.vector_norm(p8,dim=-1)
    v35=torch.linalg.vector_norm(p5-p3,dim=-1)/2.0
    v58=torch.linalg.vector_norm(p8-p5,dim=-1)/3.0
    turns=((p3[...,0]*p8[...,1]-p3[...,1]*p8[...,0])/(net3*net8+1e-3)).clamp(-1,1)
    bend=torch.linalg.vector_norm(p8-2*p5+p3,dim=-1)
    return torch.stack((prob,net3/30,net5/50,net8/80,v35/10,v58/10,
                        turns,bend/20,travelled/30),dim=-1)


def interaction_features(world,pi,pos,ids):
    # [N,6,80,2]; each target's six modes compared against neighbor's six modes,
    # weighted with the neighbor's ORIGINAL model scores. No GT/masks used.
    n=len(ids);features=world.new_zeros((n,6,8))
    if n<2:return features
    prob=pi.softmax(-1)
    for i in range(n):
        nbr=0
        for j in range(n):
            if i==j:continue
            current_sep=float(torch.linalg.vector_norm(pos[i]-pos[j]))
            if current_sep>90:continue
            nbr+=1
            distance=torch.linalg.vector_norm(world[i,:,None,:,:]-world[j,None,:,:,:],dim=-1)
            close=distance[:,:,:30].amin(-1)
            distant=distance[:,:,30:].amin(-1)
            near_pen=((6-close).clamp(min=0)/6).square()
            far_pen=((6-distant).clamp(min=0)/6).square()
            mean_near=(near_pen*prob[j][None,:]).sum(-1)
            mean_far=(far_pen*prob[j][None,:]).sum(-1)
            closest=(torch.minimum(close,distant)*prob[j][None,:]).sum(-1)
            # Features estimate interaction compatibility of each candidate with
            # other agents' predicted multimodal futures, not ground-truth futures.
            features[i,:,0]=torch.maximum(features[i,:,0],mean_near)
            features[i,:,1]=torch.maximum(features[i,:,1],mean_far)
            features[i,:,2]+=mean_near
            features[i,:,3]+=mean_far
            features[i,:,4]=torch.maximum(features[i,:,4],(1-closest/20).clamp(0,1))
            features[i,:,5]+=((closest<10).float())
            features[i,:,6]+=((closest<5).float())
        if nbr>0:
            features[i,:,2:4]/=nbr
            features[i,:,5:7]/=nbr
            features[i,:,7]=min(nbr,10)/10
    return features

