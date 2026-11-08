"""No datasets, credentials, model execution or network access."""
import hashlib,io,sys,tarfile,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from restore_paper_splits import restore
from download_paper_artifacts import extract
import download_paper_artifacts as artifacts


class PublicArtifactTests(unittest.TestCase):
    def test_offline_chunk_integrity_and_reassembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chunk=root/'part.hex';data=b'fixed inference asset'
            encoded=(data.hex()+'\n').encode('ascii');chunk.write_bytes(encoded)
            registry={'asset_transport':'repository_hex_chunks','assets':{'asset.gz':{
                'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                'chunks':[{'path':str(chunk),'sha256':hashlib.sha256(encoded).hexdigest(),'decoded_bytes':len(data)}]}}}
            result=artifacts.fetch('asset.gz',registry,root)
            self.assertEqual(result.read_bytes(),data)

    def test_offline_chunk_corruption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chunk=root/'part.hex';chunk.write_text('00')
            registry={'asset_transport':'repository_hex_chunks','assets':{'asset.gz':{
                'bytes':1,'sha256':'0'*64,'chunks':[{'path':str(chunk),'sha256':'1'*64,'decoded_bytes':1}]}}}
            with self.assertRaises(ValueError):artifacts.fetch('asset.gz',registry,root)
            self.assertFalse((root/'asset.gz').exists())

    def test_membership_order_and_metadata_preserved(self):
        doc={'kind':'dict','value':{'audit':['bb','aa'],'rule':'frozen'}}
        self.assertEqual(restore(doc,{'aa':'scene1','bb':'scene2'},'test'),
                         {'audit':['scene2','scene1'],'rule':'frozen'})

    def test_missing_membership_fails_instead_of_substitution(self):
        with self.assertRaises(ValueError):
            restore({'kind':'list','value':['not-in-local-corpus']},{},'test')

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'unsafe.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                data=b'no';m=tarfile.TarInfo('../outside.txt');m.size=len(data);t.addfile(m,io.BytesIO(data))
            with self.assertRaises(ValueError):extract(archive,root,{'../outside.txt':hashlib.sha256(b'no').hexdigest()})
            self.assertFalse((root.parent/'outside.txt').exists())


if __name__=='__main__':unittest.main()
