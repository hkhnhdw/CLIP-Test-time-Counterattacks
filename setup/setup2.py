import shutil
print("\n[1/3] Copying repository...")
src = Path('/kaggle/input/clip-test/CLIP-Test-time-Counterattacks-main')
dst = Path('/kaggle/working/clip_repo')
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"  ✓ Copied to {dst}")