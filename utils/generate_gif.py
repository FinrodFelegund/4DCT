from PIL import Image
import glob
import pydicom
import argparse
import numpy as np
import noise
from data.CustomTransforms import (
    ForegroundBBox,
    CropBatchTrain,
)
from monai.transforms import Compose
import torch


def read_data(input_path):
    """Reads the full sequence of images for all breathing phases"""
    
    phases = sorted(list(glob.glob(pathname='**', root_dir=input_path)))
    phases_with_slices = []
    for phase in phases:
        slices = glob.glob(pathname='**', root_dir=f'{input_path}{phase}')
        slices_with_metadata = []
        for slice in slices:
            dicom_data = pydicom.dcmread(fp=f'{input_path}{phase}/{slice}')
  
            z_position = dicom_data.ImagePositionPatient[2]
            slices_with_metadata.append((z_position, f'{input_path}{phase}/{slice}'))

        slices_with_metadata.sort(key=lambda x: x[0], reverse=True)
        slices_with_metadata = [Image.fromarray(pydicom.dcmread(path).pixel_array) for _, path in slices_with_metadata]
        phases_with_slices.append(slices_with_metadata)

    return phases_with_slices

def generate_gif_full_thorax(input_path, output_path, duration=200):
    """Generate a .gif file of the full thorax from a sequence of CT images."""

    phases_with_slices = read_data(input_path)
    for i, phase in enumerate(phases_with_slices):
        for img in phase:
            img = img.save(
                f'{output_path}/phase{i}.gif',
                format='GIF',
                append_images=phase[1:],
                save_all=True,
                duration=duration,
                loop=0
            )
            break

def slice_images(input_path, output_path):
    scan = np.load(input_path)
    print(scan.shape)
    C, D, H, W = scan.shape
    for d in range(D):
        sliced = (scan[0, d, :, :] * 255.0).astype(np.uint8)
        print(sliced.shape)
        slice = Image.fromarray(sliced)
        slice.save(f'{output_path}/slice_{d}_noisy.png')

def analyze_scan(input_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DEVICE, IMAGE_SIZE = 'cuda:0', 512
    MAX_PROJ = 128.98109436035156
    DOSAGE, PIXEL_SIZE_MM, SLAB = 0.25, 1.16, 16

    x = np.load(input_path)
    C, D, H, W = x.shape
    z0 = max(0, D // 2 - SLAB // 2)

    torch.manual_seed(42)
    model = noise.SinogramNoise(max_proj=MAX_PROJ, b_type='Fan-Beam', image_size=IMAGE_SIZE, dosage=DOSAGE, pixel_size_mm=PIXEL_SIZE_MM)
    with torch.no_grad():
        print(x.shape)
        x = torch.from_numpy(x[0, z0:z0+SLAB, :, :].astype(np.float32)).to(device=DEVICE)
        clean_fbp = model.backward(model.forward(x)).clamp(0., 1.)  # noiseless round-trip
        noisy = model(x)

    xc, fbpc, nzc = x.cpu(), clean_fbp.cpu(), noisy.cpu()

    def psnr(a, b, r=1.0):
        mse = torch.mean((a-b)**2).item()
        return float('inf') if mse == 0 else 10*np.log10(r*r/mse)
    def report(name, a, b):
        d = a-b
        print(f'{name:22s} PSNR={psnr(a,b):6.2f}dB  MAE={d.abs().mean():.5f}  '
            f'meanΔ={d.mean():+.5f}  std={d.std():.5f}')

    print(f'slab z=[{z0}, {z0+SLAB})  shape={tuple(x.shape)}')
    report('clean vs clean_fbp', xc, fbpc)   # (A) pure operator error — want this SMALL
    report('clean_fbp vs noisy', fbpc, nzc)  # (B) the noise you actually inject — the FAIR reference
    report('clean vs noisy',     xc, nzc)    # (C) what an un-fixed metric reports (= A + B contaminated)

    # histograms (log-y): a level shift shows as a shifted/compressed curve, not just wider
    fig, ax = plt.subplots(figsize=(7,4)); bins = np.linspace(0,1,200)
    for arr, lab in [(xc,'clean'),(fbpc,'clean_fbp'),(nzc,'noisy')]:
        ax.hist(arr.flatten().numpy(), bins=bins, histtype='step', density=True, label=lab)
    ax.set_yscale('log'); ax.legend(); ax.set_xlabel('intensity'); ax.set_title('intensity distributions')
    fig.tight_layout(); fig.savefig('diag_hist.png', dpi=120)

    # mid-slice panel + signed difference map
    m = SLAB//2; fig, a = plt.subplots(1,4, figsize=(16,4))
    a[0].imshow(xc[m],   cmap='gray', vmin=0, vmax=1); a[0].set_title('clean')
    a[1].imshow(fbpc[m], cmap='gray', vmin=0, vmax=1); a[1].set_title('clean_fbp')
    a[2].imshow(nzc[m],  cmap='gray', vmin=0, vmax=1); a[2].set_title('noisy')
    im = a[3].imshow((nzc[m]-fbpc[m]), cmap='bwr', vmin=-0.2, vmax=0.2); a[3].set_title('noisy − clean_fbp')
    for ax_ in a: ax_.axis('off')
    fig.colorbar(im, ax=a[3], fraction=0.046); fig.tight_layout(); fig.savefig('diag_slices.png', dpi=120)
    print('wrote diag_hist.png and diag_slices.png')

def test_crop(input_paths, keys):
    transforms = Compose([
        ForegroundBBox(keys=keys, min_size=127),
        CropBatchTrain(keys=keys, patch_size=127),
        ]
    )
    input_tensor_clean = torch.from_numpy(np.load(input_paths[0]).astype(np.float32)).unsqueeze(1)
    input_tensor_noisy = torch.from_numpy(np.load(input_paths[1]).astype(np.float32)).unsqueeze(1)
    data = {
        keys[0]: input_tensor_clean,
        keys[1]: input_tensor_noisy,
    }

    output = transforms(data)
    for key, value in output.items():
        print(value.shape)
        for i in range(value.shape[0]):
            img = (value[i, 0, 0, 0, :, :].numpy() * 255.0).astype(np.uint8)

            img = Image.fromarray(img)
            img.save(fp=f'/home/dpietsch/4DCT/animations/ForegroundCrops/{key}_{i}.png')


    





    

def generate_gif_breathing_motion(input_path, output_path, duration=200):
    """Generate a .gif file for a breathing phase from a sequence of CT Images."""
    
    phases_with_slices = read_data(input_path)
    phases_with_slices = [list(i) for i in zip(*phases_with_slices)]
    for i, slices in enumerate(phases_with_slices):
        for img in slices:
            img = img.save(
                f'{output_path}/slice{i}.gif',
                format='GIF',
                append_images=slices[1:],
                save_all=True,
                duration=duration,
                loop=0
            )
            break
    
    


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a .gif file from a sequence of CT images.')
    parser.add_argument('--input_path', type=str, help='Path to the input folder of a 4D CT Scan')
    parser.add_argument('--output_path', type=str, help='Path to the output .gif file')
    parser.add_argument('--duration', type=int, default=200, help='Duration of each image in the gif in milliseconds')
    
    return parser.parse_args()

if __name__ == "__main__":
    #args = parse_arguments()
    #generate_gif_breathing_motion(args.input_path, args.output_path, args.duration)
    #generate_gif_full_thorax(args.input_path, args.output_path, args.duration)
    #slice_images(args.input_path, args.output_path)
    #analyze_scan('/home/dpietsch/Data/.cache/Inhouse1Processed/022_4DCT_Lunge_amplitudebased_complete/images/phase_00.npy')
    test_crop(['/home/dpietsch/Data/.cache/Inhouse1Processed/022_4DCT_Lunge_amplitudebased_complete/images/phase_00.npy', '/home/dpietsch/Data/.cache/Inhouse1Processed/022_4DCT_Lunge_amplitudebased_complete/noise/noisy_phase_00.npy'], ['clean', 'noise'])
