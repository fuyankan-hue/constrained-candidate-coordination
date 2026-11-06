"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import numpy as np
from .geometry import observed,headings,boxes_overlap

def tables(obs):
    a=obs['m1'];w=a['world'].numpy().astype(float);current,h,size,kind,pairs,bad=observed(a,obs['history']);angle=np.stack([headings(w[:,m],current,h) for m in range(6)],1);hits=[]
    for i,j in pairs: hits.append(boxes_overlap(w[i,:,None],angle[i,:,None],size[i,None,None,None,:],w[j,None,:],angle[j,None,:],size[j,None,None,None,:]))
    hits=np.asarray(hits,dtype=bool).reshape(len(pairs),6,6,80);return {'pairs':pairs,'any':hits.any(-1),'steps':hits.sum(-1),'world':w,'current':current,'valid_box_count':len(w)-bad}

