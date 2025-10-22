# Tải AFT weights về /kaggle/working/AFT_model_weights
import gdown
from pathlib import Path

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
        print(f'{filename} already exists, skipping')
    else:
        print(f'Downloading {filename}...')
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(output_path), quiet=False)
        print(f'Downloaded {filename}')

print(f'\nAll weights downloaded to {weights_dir}')
print('Files:', list(weights_dir.glob('*.pth.tar')))