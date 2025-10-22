"""
Kaggle setup script - Run this first in a Kaggle notebook cell.
This script:
1. Copies repo from /kaggle/input to /kaggle/working
2. Patches code/utils.py to fix torch.load weights_only issue
3. Downloads AFT weights if not present

Usage:
    Run in a Kaggle notebook cell:
    !python kaggle_setup.py
"""

import shutil
from pathlib import Path
import gdown

print("=" * 60)
print("KAGGLE SETUP - CLIP Test-time Counterattacks")
print("=" * 60)

# Step 1: Copy repo from input to working
print("\n[1/3] Copying repository to /kaggle/working...")
src = Path('/kaggle/input/clip-test/CLIP-Test-time-Counterattacks-main')
dst = Path('/kaggle/working/clip_repo')

if dst.exists():
    print(f"  Removing existing {dst}")
    shutil.rmtree(dst)

print(f"  Copying {src} -> {dst}")
shutil.copytree(src, dst)
print(f"  ✓ Repository copied to {dst}")

# Step 2: Patch code/utils.py
print("\n[2/3] Patching code/utils.py for PyTorch 2.6+ compatibility...")
utils_file = dst / 'code' / 'utils.py'
utils_text = utils_file.read_text()

old_func = '''def load_resume_file(file:str, gpu:int):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        if gpu is None:
            checkpoint = torch.load(file)
        else:
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(file, map_location=loc)
        print("=> loaded checkpoint '{}' (epoch {})".format(file, checkpoint['epoch']))
        return checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return None'''

new_func = '''def load_resume_file(file:str, gpu:int):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        map_location = None if gpu is None else 'cuda:{}'.format(gpu)
        
        # Try safe load first (PyTorch 2.6+ defaults to weights_only=True)
        try:
            checkpoint = torch.load(file, map_location=map_location)
        except Exception as e:
            # If unpickling fails due to weights_only restriction, retry with weights_only=False
            err_msg = str(e)
            if 'Weights only load failed' in err_msg or 'UnpicklingError' in str(type(e)):
                print("Warning: Safe torch.load failed. Retrying with weights_only=False (unsafe - use only with trusted checkpoints).")
                try:
                    checkpoint = torch.load(file, map_location=map_location, weights_only=False)
                except Exception as e2:
                    print("Retry with weights_only=False also failed:", e2)
                    raise
            else:
                raise
        
        print("=> loaded checkpoint '{}' (epoch {})".format(file, checkpoint['epoch']))
        return checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return None'''

if old_func in utils_text:
    utils_text = utils_text.replace(old_func, new_func)
    utils_file.write_text(utils_text)
    print("  ✓ Patched code/utils.py successfully")
else:
    print("  ⚠ Warning: Could not find target function to patch (may already be patched)")

# Step 3: Download weights
print("\n[3/3] Downloading AFT weights...")
weights_dir = Path('/kaggle/working/AFT_model_weights')
weights_dir.mkdir(parents=True, exist_ok=True)

weights = {
    'FARE.pth.tar': '1IMtb5SG1ajYphR8cK-3w3Nr7zvAHa8bi',
    'TeCoA.pth.tar': '1m4Iw9pCjtBHj7OVqHFlRu2OkO0rd7C7j',
    'PMG_AFT.pth.tar': '1JMXdMheNaYWiwqcWI0tvRrp6MBweQagn'
}

for filename, file_id in weights.items():
    output_path = weights_dir / filename
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024**2)
        print(f"  ✓ {filename} already exists ({size_mb:.1f} MB)")
    else:
        print(f"  Downloading {filename} (~700 MB)...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(output_path), quiet=False)
        size_mb = output_path.stat().st_size / (1024**2)
        print(f"  ✓ Downloaded {filename} ({size_mb:.1f} MB)")

print("\n" + "=" * 60)
print("SETUP COMPLETE!")
print("=" * 60)
print("\nNext step: Run experiments with:")
print("!python /kaggle/working/clip_repo/scripts/run_evals_with_weights.py \\")
print("  --root /kaggle/working/clip_repo/data \\")
print("  --outdir /kaggle/working/TTC_results \\")
print("  --weights_dir /kaggle/working/AFT_model_weights \\")
print("  --keep-download")
