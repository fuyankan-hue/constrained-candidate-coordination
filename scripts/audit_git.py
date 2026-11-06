"""Scan working files, index blobs, all stored Git objects and history without network access."""
import argparse,hashlib,ipaddress,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
 'email':r'(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
 'ipv4':r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',
 'personal_path':r'(?:[A-Za-z]:[\\/]Users[\\/][^\s\"\']+|/(?:home|Users|root)/[^\s\"\']+|/data/(?:yanjinlong|datasets)/[^\s\"\']+)',
 'dataset_identifier_literal':r'''["'](?:[0-9a-f]{16}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})["']''',
 'private_key':r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
 'credential':r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{15,})',
 'password_literal':r'(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\"\'][^\"\']{4,}[\"\']',
}
DENIED_SUFFIXES={'.pt','.pth','.ckpt','.pkl','.pickle','.npy','.npz','.tfrecord','.sqlite','.db','.pem','.key','.so','.dll','.pyc'}

def git(*args,allow=(0,)):
 p=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True)
 if p.returncode not in allow:raise RuntimeError('Git inspection failed: '+args[0])
 return p.stdout

def scan(data,label,scope,paths):
 findings=[];exceptions=[]
 try:text=data.decode('utf-8')
 except UnicodeDecodeError:
  return [dict(scope=scope,file=label,kind='non_text_payload',bytes=len(data))],[]
 for kind,pattern in PATTERNS.items():
  for match in re.finditer(pattern,text):
   value=match.group();line=text[text.rfind('\n',0,match.start())+1:text.find('\n',match.end()) if '\n' in text[match.end():] else len(text)]
   exception=None
   if kind=='ipv4':
    try:addr=ipaddress.ip_address(value)
    except ValueError:continue
    if not (addr.is_private or addr.is_loopback or addr.is_link_local):continue
    if value=='10.4.0.35' and re.search(r'^\s*-\s*libcurand=10\.4\.0\.35=',line):
     exception='Exact libcurand package-version token, not a network endpoint'
   if kind=='personal_path' and 'path/to/root/split/' in line:
    exception='Generic upstream dataset-layout documentation, not an actual home directory'
   if kind=='email' and value=='geom@cs.unc.edu' and any(x.endswith('reproduction/external/classical/orca_port.cpp') for x in paths):
    exception='Public RVO2 bug-report contact in the retained original copyright header'
   if label.endswith('scripts/audit_git.py') and ((kind=='email' and value=='geom@cs.unc.edu' and "value==" in line) or
        (kind=='ipv4' and value=='10.4.0.35' and "value==" in line)):
    exception='Explicit scanner policy declaration for an independently reviewed nonsecret token'
   if kind=='email' and value=='open-dataset@waymo.com' and (any(x=='LICENSES/Waymo_Dataset_License_March_2025.html' for x in paths) or (label.endswith('scripts/audit_git.py') and 'value==' in line)):
    exception='Public provider support address in the mandatory unmodified Waymo agreement copy, or its explicit scanner policy'
   if kind=='email' and value.rsplit('@',1)[-1] in ('example.com','example.org','example.invalid'):
    exception='Reserved documentation example domain'
   if label.endswith('scripts/audit_git.py') and ('r\'' in line or 'r"' in line) and kind=='personal_path':
    exception='Scanner regular-expression declaration, not an execution path'
   item=dict(scope=scope,file=label,kind=kind,line=text.count('\n',0,match.start())+1,
             matched_value_SHA256=hashlib.sha256(value.encode()).hexdigest())
   if exception:item['reason']=exception;exceptions.append(item)
   else:findings.append(item)
 return findings,exceptions

def inspect(require_staged=False,expect_no_remote=False):
 assert (ROOT/'.git').is_dir() and not (ROOT/'.git').is_symlink(),'Unexpected external Git directory'
 assert git('rev-parse','--show-toplevel').decode().strip()==str(ROOT)
 findings=[];exceptions=[];large=[];working=0;links=[]
 for path in ROOT.rglob('*'):
  if '.git' in path.parts:continue
  rel=path.relative_to(ROOT).as_posix()
  if path.is_symlink():links.append(rel);continue
  if not path.is_file():continue
  if rel=='.ccc-review-candidate':continue
  data=path.read_bytes();working+=1
  if path.suffix.lower() in DENIED_SUFFIXES:findings.append(dict(scope='working',file=rel,kind='denied_artifact_type'))
  if len(data)>=1024**2:large.append(dict(file=rel,bytes=len(data),type='scene-ID metadata' if rel.startswith('configs/splits/') else 'manual-review text file'))
  a,b=scan(data,rel,'working',[rel]);findings.extend(a);exceptions.extend(b)
 for link in links:findings.append(dict(scope='working',file=link,kind='symlink'))
 if (ROOT/'.git/objects/info/alternates').exists():findings.append(dict(scope='git',kind='external_object_alternates'))
 if (ROOT/'.git/commondir').exists():findings.append(dict(scope='git',kind='external_common_directory'))
 index=[];by_oid={}
 for entry in git('ls-files','--stage','-z').split(b'\0'):
  if not entry:continue
  head,name=entry.split(b'\t',1);mode,oid,stage=head.decode().split();rel=name.decode('utf-8');index.append(rel);by_oid.setdefault(oid,[]).append(rel)
  if stage!='0' or mode not in ('100644','100755'):findings.append(dict(scope='index',file=rel,kind='symlink_submodule_or_unmerged_entry'))
  if Path(rel).suffix.lower() in DENIED_SUFFIXES:findings.append(dict(scope='index',file=rel,kind='denied_artifact_type'))
  if Path(rel).is_absolute() or '..' in Path(rel).parts:findings.append(dict(scope='index',file=rel,kind='unsafe_index_path'))
 if require_staged and not index:findings.append(dict(scope='index',kind='no_staged_files'))
 # Historical path names and commit metadata are checked in addition to object content.
 for row in git('rev-list','--objects','--all').decode().splitlines():
  a=row.split(' ',1)
  if len(a)==2:by_oid.setdefault(a[0],[]).append(a[1])
 object_counts={};object_bytes=0
 for row in git('cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)').decode().splitlines():
  oid,kind,size=row.split();object_counts[kind]=object_counts.get(kind,0)+1;object_bytes+=int(size)
  paths=by_oid.get(oid,[])
  if kind in ('blob','commit','tag'):
   label=paths[0] if paths else 'object:'+oid
   data=git('cat-file',kind,oid);a,b=scan(data,label,'all_git_objects',paths);findings.extend(a);exceptions.extend(b)
  if kind=='blob' and int(size)>50*1024**2:findings.append(dict(scope='all_git_objects',object=oid,kind='oversized_blob',bytes=int(size)))
 fsck=subprocess.run(['git','-C',str(ROOT),'fsck','--full','--no-reflogs','--unreachable'],capture_output=True)
 if fsck.returncode:findings.append(dict(scope='git',kind='fsck_error'))
 remote_names=git('remote').decode().splitlines()
 if expect_no_remote and remote_names:findings.append(dict(scope='git',kind='remote_present'))
 # Local configuration is not uploaded, but embedded credentials/paths are still audited.
 for file in (ROOT/'.git/config',):
  a,b=scan(file.read_bytes(),'.git/config','local_git_config',[]);findings.extend(a);exceptions.extend(b)
 return dict(pass_=not findings,working_text_files=working,index_files=len(index),commits=int(git('rev-list','--all','--count')),
    refs=len(git('show-ref',allow=(0,1)).decode().splitlines()),reflog_entries=len(git('reflog','show','--all','--format=%H').decode().splitlines()),
    git_object_counts=object_counts,git_object_bytes=object_bytes,git_fsck_pass=fsck.returncode==0,
    remote_count=len(remote_names),symlinks=links,large_text_files=large,
    findings=findings,reviewed_nonsecret_matches=exceptions,
    scope='All working files including ignored artifacts, all index paths, every stored Git object (including unreachable), refs and reflogs. Pattern-based audit, not a proof of ownership or exhaustive secret detection.')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--require-staged',action='store_true');p.add_argument('--expect-no-remote',action='store_true');p.add_argument('--report',type=Path)
 a=p.parse_args();result=inspect(a.require_staged,a.expect_no_remote)
 if a.report:
  a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result));raise SystemExit(0 if result['pass_'] else 1)

if __name__=='__main__':main()
