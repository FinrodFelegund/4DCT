import torch
from models.SPBF import FilterBank
from data.dataset import CT4dDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
import wandb
from utils.processing import compute_patch_offsets, make_sigma_plot

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)



def train(config: Dict):

    
    wandb.init(
        entity='ipmi',
        project='4DCT-Denoising',
        name=config['name'],
        config=config,
    )
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FilterBank(num_spatial=config.get('num_spatial'), num_temporal=config.get('num_temporal'))
    model = model.to(device)

    train_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('train_data')),
        transforms=CT4dDataset.get_train_transforms(),
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.get('batch_size', 1), shuffle=True, num_workers=4, collate_fn=CT4dDataset.collate_fn)
    
    validation_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('validation_data')),
        transforms=CT4dDataset.get_validation_transforms(),
        scan_transforms=CT4dDataset.get_scan_transforms(),
    )

    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=1, collate_fn=CT4dDataset.collate_fn)
    
    range_sigmas = [p for n, p in model.named_parameters() if n.endswith('sigma_r')]
    st_sigmas   = [p for n, p in model.named_parameters() if not n.endswith('sigma_r')]
    optimizer = torch.optim.Adam([
        {'params': st_sigmas, 'lr': config.get('lr_st', 0.01)},
        {'params': range_sigmas, 'lr': config.get('lr_r', 0.01)},
    ])


    scaler = torch.amp.GradScaler(device=device.type)

    criterion = torch.nn.HuberLoss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    epochs = config.get('epochs', 10)
    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Training]')
        train_loss = 0
        for i, (clean, noisy) in enumerate(train_pbar):
            
            clean, noisy = clean.to(device), noisy.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                prediction = model(noisy)
                loss = criterion(prediction, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name.endswith('sigma_r'):
                        param.clamp_(min=0.01)
                    else:
                        param.clamp_(min=0.01, max=2.0)

            kernel_sizes = model.get_kernel_size()
            wandb_dict = {}
            current_sigmas = model.get_sigmas_by_type()
            for param_name, layer_list in current_sigmas.items():
                for layer_idx, sigma_value in enumerate(layer_list):
                    key = f'Sigmas/{param_name}_Layer_{layer_idx + 1}'
                    wandb_dict[key] = sigma_value

            for key, value in kernel_sizes.items():
                if isinstance(value, tuple):
                    for axis, k in zip(('z', 'y', 'x'), value):
                        wandb_dict[f'Kernels/{key}_{axis}'] = k
                else:
                    wandb_dict[f'Kernels/{key}'] = value

            wandb_dict['Train/Loss'] = loss.item()
 
            wandb.log(wandb_dict)

  

            train_pbar.set_postfix(Loss=f'{loss.item():.6f}')
            train_loss += loss.item()


        wandb.log({
            'Train/Mean-Loss-Epoch': train_loss / len(train_dataloader),
        })

        
        
 
        if epoch % 2 == 0:
            model.eval()
            psnr_metric.reset()
            ssim_metric.reset()
            val_loss = 0
            n_batches = 0
            val_pbar = tqdm(validation_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]')


            with torch.no_grad():
                for i, (clean, noisy) in enumerate(val_pbar):

                    patch_batch_size = 4


                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        for idx in range(0, clean.shape[0], patch_batch_size):
                            batch_clean, batch_noisy = clean[idx:idx+patch_batch_size, ...].to(device), noisy[idx:idx+patch_batch_size, ...].to(device)
                            batch_prediction = model(batch_noisy)
                            val_loss += criterion(batch_prediction, batch_clean)
                            n_batches += 1
                            psnr_metric.update(batch_prediction, batch_clean)

                            B, C, T, D, H, W = batch_prediction.shape
                            ssim_metric.update(batch_prediction.view(B*T*D, C, H, W), batch_clean.view(B*T*D, C, H, W))


                            val_pbar.set_postfix(Info=f'Image: {i+1} | Batch: { idx // patch_batch_size + 1} / {clean.shape[0] // patch_batch_size} | Shape: {clean.shape}')

                        clean, noisy = validation_dataset.get_whole_slice()
                        clean, noisy = clean.to(device), noisy.to(device)
                        pred = model(noisy)
                        val_image = torch.cat([clean[:, :, 0, 0, :, :], pred[:, :, 0, 0, :, :], noisy[:, :, 0, 0, :, :]], dim=0)
                        val_image = make_grid(val_image, nrow=3, normalize=False)

            val_psnr = psnr_metric.compute().item()
            val_ssim = ssim_metric.compute().item()
            val_loss = (val_loss / n_batches)

            wandb.log({
                'Validation/Loss': val_loss,
                'Validation/Psnr': val_psnr,
                'Validation/ssim': val_ssim,
                'Validation/Epoch': epoch + 1,
                'Validation/Images': wandb.Image(
                    val_image,
                    caption=f'Epoch {epoch+1}'
                )
            })

            print(f'\nEpoch {epoch + 1} Summary: Validation Loss: {val_loss:.6f} | Validation PSNR: {val_psnr:.6f} | Validation SSIM: {val_ssim:.6f}')

    wandb.finish()



     

            
