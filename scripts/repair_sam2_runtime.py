"""Install an existing pinned SAM2 checkout using existing NVIDIA headers only."""
import argparse, importlib.util, os, shutil, subprocess, sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Official SAM2 checkout 0e78a118995e66bb27d78518c4bd9a3e95b4e266')
    args=parser.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!='0e78a118995e66bb27d78518c4bd9a3e95b4e266':raise SystemExit('Expected the official Wan-pinned SAM2 revision')
    import nvidia
    roots=[Path(p) for p in nvidia.__path__]
    env=dict(os.environ)
    for variable,pattern in [('CPATH','*/include'),('LIBRARY_PATH','*/lib')]:
        paths=[str(path) for root in roots for path in root.glob(pattern)]
        if env.get(variable):paths.append(env[variable])
        env[variable]=os.pathsep.join(paths)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',str(args.source)],env=env,check=True)
    target=Path(importlib.util.find_spec('sam2_configs').origin).parent
    files=list((args.source/'sam2_configs').glob('*.yaml'))
    if not files:raise RuntimeError('Pinned checkout is missing official YAML configs')
    for path in files:shutil.copy2(path,target/path.name)
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    print(f'Installed {len(files)} official YAML configs. Real segmentation still requires its validation suite.')

if __name__=='__main__':main()
