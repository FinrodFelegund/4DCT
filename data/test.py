from .dataset import CT4dDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import noise
import torch

DATASETPATHS = ['/home/dpietsch/Pictures/Inhouse2']


def iterate_dataset():
    data = CT4dDataset.generate_dataframe(DATASETPATHS)
    
    ct4d_dataset = CT4dDataset(data, CT4dDataset.get_train_transforms())
    tqdm_dataloader = tqdm(DataLoader(ct4d_dataset, batch_size=1, num_workers=4, collate_fn=CT4dDataset.collate_fn))

    for gt, noise in tqdm_dataloader:
        tqdm_dataloader.set_description(f'Shape gt: {gt.shape} shape noise: {noise.shape}, Type: {type(gt)}')
    


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
    iterate_dataset()
    #sinogram_domain_noise()
    #image_domain_noise()