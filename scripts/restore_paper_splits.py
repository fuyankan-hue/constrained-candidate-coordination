"""Recover exact paper cohorts from public one-way memberships and user-owned metadata."""
import argparse,gzip,hashlib,json
from pathlib import Path
from _bootstrap import ROOT


def restore(document,lookup,namespace):
    def decode(values):
        result=[]
        for token in values:
            if token not in lookup:
                raise ValueError('A paper membership is absent from the supplied local corpus; use the original dataset version and complete indexes.')
            result.append(lookup[token])
        return result
    value=document['value']
    if document['kind']=='list':return decode(value)
    if document['kind']!='dict':raise ValueError('Unknown membership format')
    return {k:decode(v) if isinstance(v,list) else v for k,v in value.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['womd','av2'])
    p.add_argument('--memberships',type=Path,default=ROOT/'local/paper_artifacts/paper-split-hashes.json.gz')
    p.add_argument('--training-index',type=Path)
    p.add_argument('--validation-index',type=Path)
    p.add_argument('--processed-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    registry=json.loads((ROOT/'configs/artifact_release.json').read_text(encoding='utf-8'))
    raw=a.memberships.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=registry['assets']['paper-split-hashes.json.gz']['sha256']:
        p.error('Membership asset checksum mismatch')
    bundle=json.loads(gzip.decompress(raw))
    if bundle['format_version']!=1:p.error('Unknown membership version')
    names=[]
    if a.dataset=='womd':
        if not a.training_index or not a.validation_index:p.error('WOMD requires both local raw-record indexes')
        for path,expected,count in [(a.training_index,'training',486995),(a.validation_index,'validation',44097)]:
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(rows)!=count or any(r['split']!=expected for r in rows):p.error('Incomplete or wrong-split WOMD raw index')
            names.extend(str(r['scene']) for r in rows)
    else:
        if not a.processed_root:p.error('AV2 requires its local processed corpus')
        for split,count in [('train',199908),('val',24988)]:
            files=sorted(x.name for x in (a.processed_root/split/'processed').glob('*.pkl'))
            if len(files)!=count:p.error('AV2 requires the complete original processed split')
            names.extend(files)
            names.extend(Path(x).stem for x in files)
    lookup={}
    namespace=bundle['namespace']
    for name in names:
        token=hashlib.sha256((namespace+'\0'+name).encode('utf-8')).hexdigest()
        if token in lookup and lookup[token]!=name:p.error('Membership hash collision')
        lookup[token]=name
    from ccc.splits import write_documents
    rules=json.loads((ROOT/'configs/split_rules.json').read_text(encoding='utf-8'))
    documents={name:restore(doc,lookup,namespace) for name,doc in bundle['documents'].items() if name.startswith(a.dataset+'_')}
    if len(documents)!=(3 if a.dataset=='womd' else 4):p.error('Incomplete paper membership bundle')
    # Original canonical hashes check every member AND ordering before any write.
    write_documents(documents,a.output,rules)
    print(json.dumps(dict(pass_=True,dataset=a.dataset,files=sorted(documents),
        exact_original_membership_and_order=True,private_exclusion_or_qualification_inputs_required=False,
        raw_records_or_future_labels_read=False,model_execution=False)))


if __name__=='__main__':main()
