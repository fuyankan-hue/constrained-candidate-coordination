"""No datasets, credentials, model execution or network access."""
import hashlib,io,sys,tarfile,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from restore_paper_splits import restore
from download_paper_artifacts import extract


class PublicArtifactTests(unittest.TestCase):
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
