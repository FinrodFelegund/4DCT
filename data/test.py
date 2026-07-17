from .dataset import CT4dDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import noise
import torch
import numpy as np
import cv2 as cv

DATASETPATHS = ['/home/dpietsch/Pictures/.cache/Inhouse1Processed']


def iterate_dataset():
    data = CT4dDataset.generate_dataframe(DATASETPATHS)
    print(data.head())

    ct4d_dataset = CT4dDataset(data, CT4dDataset.get_train_transforms())
    tqdm_dataloader = tqdm(DataLoader(ct4d_dataset, batch_size=1, num_workers=0, collate_fn=CT4dDataset.collate_fn))

    for gt, noise in tqdm_dataloader:
        tqdm_dataloader.set_description(f'Shape gt: {gt.shape} shape noise: {noise.shape}, Type: {type(gt)}')

def view_stack(clean: str, noisy: str):
    img_clean = np.load(clean).astype(np.float32)
    img_noisy = np.load(noisy).astype(np.float32)
    print('Clean')
    print(img_clean.shape)
    print(img_clean.min(), img_clean.max())
    print('Noise')
    print(img_noisy.shape)
    print(img_noisy.min(), img_noisy.max())
    
    D, H, W = img_clean.shape[1:]
    for i in range(D):
        slice_img_clean = (img_clean[0, i, :, :] * 255).astype(np.uint8)
        slice_img_noisy = (img_noisy[0, i, :, :] * 255).astype(np.uint8)
        out_image = np.zeros(shape=[H, 3*W], dtype=np.uint8)
        out_image[0:H, 0:W] = slice_img_clean
        out_image[0:H, W:2*W] = slice_img_noisy
        difference = slice_img_clean.astype(np.float32)-slice_img_noisy.astype(np.float32)
        difference = (difference - difference.min()) / (difference.max() - difference.min())
        difference = difference * 255.0
        out_image[0:H, 2*W:3*W] = difference.astype(np.uint8)
        
        cv.imshow('patch', out_image)
        cv.waitKey(0)

    cv.destroyAllWindows()


def sinogram_domain_noise():

    import matplotlib.pyplot as plt

    data = CT4dDataset.generate_dataframe(DATASETPATHS)
    ct4d_dataset = CT4dDataset(data, CT4dDataset.get_transforms())
    dataloader = DataLoader(ct4d_dataset, batch_size=1, num_workers=0, collate_fn=CT4dDataset.collate_fn)

    data = next(iter(dataloader))[:, :1]

    image_size = data.shape[-2]



    noiser = noise.SinogramNoise(b_type='Parallel-Beam', image_size=image_size)
    noisy_images = noiser(data).to(torch.device('cpu'))

    print(f'Shapes: {data.shape}, {noisy_images.shape}')
    print(f'Info: Original Image min : {data.min()} max: {data.max()}')
    print(f'Info: Noisy Image min : {noisy_images.min()} max: {noisy_images.max()}')
    print(f'Error: {((data - noisy_images)**2).mean() }')

    f, axis = plt.subplots(5, 2, figsize=(10, 20))
    
    for i in range(5):
        axis[i, 0].imshow(data[0, 0, :, :, i].T, cmap='gray')
        axis[i, 0].axis('off')
        axis[i, 1].imshow(noisy_images[0, 0, :, :, i].T, cmap='gray')
        axis[i, 1].axis('off')

    f.suptitle('Sinogram Domain Noise')
    plt.tight_layout()
    plt.show()


def image_domain_noise():
    import matplotlib.pyplot as plt

    data = CT4dDataset.generate_dataframe(DATASETPATHS)
    ct4d_dataset = CT4dDataset(data, CT4dDataset.get_transforms())
    dataloader = DataLoader(ct4d_dataset, batch_size=1, num_workers=0, collate_fn=CT4dDataset.collate_fn)

    data = next(iter(dataloader))[:, :1]

    image_size = data.shape[-2]

    noiser = noise.SinogramNoise(b_type='Parallel-Beam', image_size=image_size)
    noisy_images = noiser(data).to(torch.device('cpu'))
    noiser = noise.ImageNoise(noise_level=0.1)
    noisy_images = noiser(data).to(torch.device('cpu'))

    print(f'Shapes: {data.shape}, {noisy_images.shape}')
    print(f'Info: Original Image min : {data.min()} max: {data.max()}')
    print(f'Info: Noisy Image min : {noisy_images.min()} max: {noisy_images.max()}')
    print(f'Error: {((data - noisy_images)**2).mean() }')

    f, axis = plt.subplots(5, 2, figsize=(10, 20))
    
    for i in range(5):
        axis[i, 0].imshow(data[0, 0, :, :, i].T, cmap='gray')
        axis[i, 0].axis('off')
        axis[i, 1].imshow(noisy_images[0, 0, :, :, i].T, cmap='gray')
        axis[i, 1].axis('off')

    f.suptitle('Image Domain Noise')
    plt.tight_layout()
    plt.show()


        



def test():
    print('=== Running Tests ===')
    #iterate_dataset()
    view_stack(clean='/home/dpietsch/Pictures/.cache/Inhouse1Processed/022_4DCT_Lunge_amplitudebased_complete/images/phase_00.npy', noisy='/home/dpietsch/Pictures/.cache/Inhouse1Processed/022_4DCT_Lunge_amplitudebased_complete/noise/noisy_phase_00.npy')
    #view_stack(clean='/home/dpietsch/Pictures/.cache/Inhouse2Processed/case_01/images/phase_00.npy', noisy='/home/dpietsch/Pictures/.cache/Inhouse2Processed/case_01/noise/noisy_phase_00.npy')
    #sinogram_domain_noise()
    #image_domain_noise()