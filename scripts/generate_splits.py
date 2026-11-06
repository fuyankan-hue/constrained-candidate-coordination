"""Regenerate local historical metadata from authorized user-owned inputs; no model execution."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='dataset',required=True)
    w=sub.add_parser('womd');w.add_argument('--raw-index',type=Path,required=True)
    w.add_argument('--known-exposure',type=Path,required=True)
    w.add_argument('--old-probe',type=Path,required=True)
    w.add_argument('--confirmation-qualified',type=Path)
    w.add_argument('--validation-index',type=Path)
    a=sub.add_parser('av2');a.add_argument('--processed-root',type=Path,required=True)
    for command in (w,a):command.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from ccc.splits import read_ids,womd_documents,av2_documents,write_documents
    rules=json.loads((ROOT/'configs/split_rules.json').read_text())
    if args.dataset=='womd':
        docs=womd_documents(records(args.raw_index),read_ids(args.known_exposure),read_ids(args.old_probe),rules,
            read_ids(args.confirmation_qualified) if args.confirmation_qualified else None,
            records(args.validation_index) if args.validation_index else None)
    else:
        docs=av2_documents([p.name for p in (args.processed_root/'train/processed').glob('*.pkl')],
                           [p.name for p in (args.processed_root/'val/processed').glob('*.pkl')],rules)
    write_documents(docs,args.output,rules)
    print(json.dumps(dict(pass_=True,dataset=args.dataset,files=sorted(docs),
        historical_membership_and_order_fingerprint_match=True,model_execution=False,
        note='Local metadata only. No inference, future-label access, parameter selection or unseen-data claim.')))


if __name__=='__main__':main()
