"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import hashlib,json,os,re,struct,tempfile,time
from pathlib import Path
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from ccc.io import sha,save_json
RUN=None
def save(value,path):return save_json(value,path)
def disk(gb=0):
    import shutil
    assert shutil.disk_usage(RUN).free>gb*2**30,'Insufficient free disk'
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def crc_smoke():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'check.tfrecord'
        with tf.io.TFRecordWriter(str(p)) as w:w.write(b'known_payload')
        assert list(tf.compat.v1.io.tf_record_iterator(str(p)))==[b'known_payload']
        b=bytearray(p.read_bytes());b[13]^=1;p.write_bytes(b)
        try:list(tf.compat.v1.io.tf_record_iterator(str(p)))
        except tf.errors.DataLossError:return True
        raise AssertionError('CRC checker accepted corrupted payload')


def main(worker,workers):
    assert crc_smoke(); inventory=json.loads((RUN/'inventory.json').read_text()); files=inventory['files'][worker::workers]
    start=time.monotonic()
    for item in files:
        disk(60);p=Path(item['path']);tag='%s_%04d'%(item['split'],item['shard']);out=RUN/'index'/ (tag+'.json')
        frozen={k:item[k] for k in ('path','size','mtime_ns')}
        assert p.stat().st_size==item['size'] and p.stat().st_mtime_ns==item['mtime_ns'],('SOURCE_CHANGED',str(p))
        if out.exists():
            old=json.loads(out.read_text());assert old['source']==frozen and old['pass'] and old['source_code_sha256']==sha(__file__)
            continue
        rows=[];offset=0;record_digest=hashlib.sha256()
        for record_index,encoded in enumerate(tf.compat.v1.io.tf_record_iterator(str(p))):
            scene=scenario_pb2.Scenario();scene.ParseFromString(encoded);sid=scene.scenario_id
            assert sid and re.fullmatch('[A-Za-z0-9_-]{1,120}',sid),(p,record_index,'invalid scene id')
            assert 0<=scene.current_time_index<len(scene.timestamps_seconds) and scene.current_time_index==10,(sid,'history contract')
            assert len(scene.timestamps_seconds)==91,(sid,'sequence length',len(scene.timestamps_seconds))
            assert all(len(t.states)==91 for t in scene.tracks),(sid,'track length')
            assert len({t.id for t in scene.tracks})==len(scene.tracks),(sid,'duplicate track IDs')
            assert 0<=scene.sdc_track_index<len(scene.tracks),(sid,'SDC index')
            target_indices=[t.track_index for t in scene.tracks_to_predict]
            assert len(set(target_indices))==len(target_indices) and all(0<=i<len(scene.tracks) for i in target_indices)
            digest=hashlib.sha256(encoded).hexdigest();record_digest.update(bytes.fromhex(digest))
            rows.append(dict(scene=sid,record_index=record_index,offset=offset,length=len(encoded),sha256=digest,
              agents=len(scene.tracks),targets=len(target_indices),history_agents=sum(any(x.valid for x in t.states[:11]) for t in scene.tracks)))
            offset+=len(encoded)+16
        assert rows and offset==item['size'],('TRUNCATED_OR_EMPTY',str(p),offset,item['size'])
        assert len({r['scene'] for r in rows})==len(rows),(str(p),'duplicate scenes within file')
        assert p.stat().st_size==item['size'] and p.stat().st_mtime_ns==item['mtime_ns']
        save(dict(pass_=True,**{'pass':True},source=frozen,source_code_sha256=sha(__file__),CRC_smoke_pass=True,
             payload_digest_sha256=record_digest.hexdigest(),split=item['split'],shard=item['shard'],records=rows),out)
        print('CRC_PROTOBUF_FILE_PASS',worker,tag,'records',len(rows),'elapsed',round(time.monotonic()-start,1),flush=True)
    save({'pass':True,'worker':worker,'files':len(files),'inventory_sha256':sha(RUN/'inventory.json'),'source_code_sha256':sha(__file__)},RUN/'index'/('worker%d_complete.json'%worker))

