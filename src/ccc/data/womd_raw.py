"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import hashlib,struct
import numpy as np
import torch
from pathlib import Path
import datasets.waymo_dataset as wm
from datasets.waymo_dataset import WaymoDataset
from waymo_open_dataset.protos import scenario_pb2
RAW=None

def interp_arc(n,points):
    p=np.asarray(points);assert p.shape[0]==2;return np.linspace(p[0],p[1],n,endpoint=True)


class SafeDataset(WaymoDataset):
    def _download(self):return
    def _process(self):return


def make(dataset,row):
    split=row['split'];count=1000 if split=='training' else 150;p=RAW/split/'raw'/f'{split}.tfrecord-{row["shard"]:05d}-of-{count:05d}'
    with p.open('rb') as f:
        f.seek(row['offset']);length=struct.unpack('<Q',f.read(8))[0];f.read(4);assert length==row['length'];encoded=f.read(length)
    assert hashlib.sha256(encoded).hexdigest()==row['sha256'];s=scenario_pb2.Scenario();s.ParseFromString(encoded);assert s.scenario_id==row['scene']
    agent=dataset.get_agent_features(s)
    now={t.id:bool(t.states[10].valid) for t in s.tracks}
    agent['observed_current_valid']=torch.tensor([now[int(t)] for t in agent['id']],dtype=torch.bool)
    maps=dataset.get_map_features(s);raw=dict(scenario_id=s.scenario_id,agent=agent);raw.update(maps)
    return dict(record=row,raw=raw,official_track_ids=[s.tracks[t.track_index].id for t in s.tracks_to_predict],
      num_map_polygons=int(maps['map_polygon']['num_nodes']),objects_of_interest=list(s.objects_of_interest))

