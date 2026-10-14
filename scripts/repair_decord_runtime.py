import base64,csv,hashlib,io,json,subprocess,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import requests
import argparse,importlib.util,shutil
parser=argparse.ArgumentParser(description='Repair Decord wheel metadata, retaining verified official native libraries')
parser.add_argument('--source',type=Path,required=True,help='Existing official Decord v0.6.0 checkout')
parser.add_argument('--work-dir',type=Path,required=True)
parser.add_argument('--video',type=Path,required=True,help='Real video for the post-install decode check')
args=parser.parse_args()
revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
if revision!='6a3617cef035535193f390f86399dde139fa2a53':raise SystemExit('Expected official Decord v0.6.0 source revision')
package=Path(importlib.util.find_spec('decord').origin).parent
build=args.source/'build';build.mkdir(exist_ok=True)
shutil.copy2(next(package.rglob('libdecord.so')),build/'libdecord.so')
root=args.work_dir;root.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','pip','download','--no-deps','decord==0.6.0','-d',str(root)],check=True)
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',str(args.source/'python'),'-w',str(root)],check=True)
official=next(root.glob('*manylinux2010*.whl'))
metadata=requests.get('https://pypi.org/pypi/decord/0.6.0/json',timeout=30).json()
expected=next(file['digests']['sha256'] for file in metadata['urls'] if file['filename']==official.name)
assert hashlib.sha256(official.read_bytes()).hexdigest()==expected
built=next(path for path in root.glob('*.whl') if 'manylinux2010' not in path.name)
with ZipFile(built) as archive:entries={name:archive.read(name) for name in archive.namelist() if not name.endswith('/')}
with ZipFile(official) as archive:
    for name in archive.namelist():
        if name.startswith('decord.libs/') and not name.endswith('/'):entries[name]=archive.read(name)
assert any(name.startswith('decord.libs/') for name in entries)
record=next(name for name in entries if name.endswith('.dist-info/RECORD'))
rows=[]
for name,data in sorted(entries.items()):
    if name==record:continue
    rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(),str(len(data))])
rows.append([record,'',''])
text=io.StringIO();csv.writer(text,lineterminator='\n').writerows(rows);entries[record]=text.getvalue().encode()
temporary=built.with_suffix('.tmp')
with ZipFile(temporary,'w',ZIP_DEFLATED) as archive:
    for name,data in entries.items():archive.writestr(name,data)
temporary.replace(built)
subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','--no-deps',str(built)],check=True)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
from decord import VideoReader
reader=VideoReader(str(args.video))
assert len(reader)>0 and reader.get_batch(list(range(min(5,len(reader))))).asnumpy().ndim==4
print('Rebuilt official wrapper plus unchanged verified native/FFmpeg SDK: real decoding and pip check passed')
