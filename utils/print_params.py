import torch
from pathlib import Path
from glob import glob

CKPT_PATH = Path('/home/dpietsch/4DCT/parameters')
DEVICE = 'cuda'

def print_params():
    model_ckpts = sorted(glob(r'*.pth', root_dir=CKPT_PATH))

    for ckpt in model_ckpts:
        model_path = CKPT_PATH / ckpt
        ckpt_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
        print(f'\n{ckpt}  (epoch {ckpt_data.get("epoch")}, val loss {ckpt_data.get("val_loss")})\n')
        for name, param in ckpt_data['model'].items():
            n = name.rsplit('.', 1)[-1] 
            if n.startswith('log_') or n.startswith('raw_'):
                continue
                k = torch.exp(param)
                print(k)
                print(name)
                print('(normalised to sum 1)')
                print(k / k.sum())
            else:
                print(name, param)

print_params()
