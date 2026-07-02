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
    model = FilterBank(num_spatial=config.get('num_spatial'), num_temporal=config.get('num_temporal'), device=device)

    train_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('train_data')),
        transforms=CT4dDataset.get_train_transforms(),
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.get('batch_size', 1), shuffle=True, num_workers=4, collate_fn=CT4dDataset.collate_fn)
    
    validation_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('validation_data')),
        transforms=CT4dDataset.get_validation_transforms(),
    )

    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=1, collate_fn=CT4dDataset.collate_fn)
    
    sigmas_st, sigmas_r = model.parameters()
    optimizer = torch.optim.Adam([
        {'params': sigmas_st, 'lr': config.get('lr_st', 0.0005)},
        {'params': sigmas_r, 'lr': config.get('lr_r', 0.01)},
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
                p_1, p_2 = model.parameters()
                torch.nn.utils.clip_grad_norm_(p_1 + p_2, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            kernel_sizes = model.get_kernel_size()
            wandb_dict = {}
            current_sigmas = model.get_sigmas_by_type()
            for param_name, layer_list in current_sigmas.items():
                for layer_idx, sigma_value in enumerate(layer_list):
                    key = f'Sigmas/{param_name}_Layer_{layer_idx + 1}'
                    wandb_dict[key] = sigma_value

            for key, value in kernel_sizes.items():
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
            val_images = None
            val_pbar = tqdm(validation_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]')


            with torch.no_grad():
                norm_factor = 0
                for i, (clean, noisy) in enumerate(val_pbar):
                    
                    full_pred = torch.zeros_like(clean, device='cpu', dtype=torch.float32)
                    count_map = torch.zeros_like(clean, device='cpu', dtype=torch.float32)
                    B, C, T, D, H, W = clean.shape
                    d_offsets, h_offsets, w_offsets, num_patches = compute_patch_offsets(D, H, W, patch_size=32, window_step=16)

                    

                    patch_batch_size = 16
                    batch_noisy = []
                    batch_clean = []
                    batch_coords = []

                    all_coords = [(d, h, w) for d in d_offsets for h in h_offsets for w in w_offsets]

                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        for idx, (d, h, w) in enumerate(all_coords):
                            batch_clean.append(clean[:, :, :, d:d+32, h:h+32, w:w+32])
                            batch_noisy.append(noisy[:, :, :, d:d+32, h:h+32, w:w+32])
                            batch_coords.append((d, h, w))

                            is_last_patch = (idx == len(all_coords) - 1)

                            if len(batch_noisy) == patch_batch_size or is_last_patch:
                                clean_patches = torch.cat(batch_clean, dim=0).to(device)
                                noisy_patches = torch.cat(batch_noisy, dim=0).to(device)
                                prediction = model(noisy_patches)
                                val_loss += criterion(prediction, clean_patches).item() * len(clean_patches)
                                norm_factor += len(clean_patches)



                                psnr_metric.update(prediction, clean_patches)

                                B, C, T, D, H, W = prediction.shape

                                ssim_metric.update(prediction.view(B*T*D, C, H, W), clean_patches.view(B*T*D, C, H, W))

                                prediction_f32 = prediction.float()
    

                                if val_images == None:
                                    for batch_idx, (pd, ph, pw) in enumerate(batch_coords):
                                        full_pred[:, :, :, pd:pd+32, ph:ph+32, pw:pw+32] += prediction_f32[batch_idx:batch_idx+1].detach().cpu()
                                        count_map[:, :, :, pd:pd+32, ph:ph+32, pw:pw+32] += 1 
                                
                                batch_clean.clear()
                                batch_noisy.clear()
                                batch_coords.clear()

                            val_pbar.set_postfix(Info=f'Image: {i+1} | Patch: {idx} / {len(all_coords)} | Shape: {clean.shape}')

                                    





                    if val_images == None:
                        full_pred = full_pred / (count_map + 1e-8)
                        T, D = full_pred.shape[2], full_pred.shape[3]

                        t_mid = T // 2
                        d_mid = D // 2

                        clean_slice = clean[:, :, t_mid, d_mid, :, :].cpu().float()
                        noise_slice = noisy[:, :, t_mid, d_mid, :, :].cpu().float()
                        pred_slice = full_pred[:, :, t_mid, d_mid, :, :]
                        
                        val_images = torch.cat([clean_slice, noise_slice, pred_slice], dim=0)
                        val_images = make_grid(val_images, nrow=3, normalize=False)
                        del full_pred, count_map


            val_psnr = psnr_metric.compute().item()
            val_ssim = ssim_metric.compute().item()
            val_loss = (val_loss / norm_factor)

            wandb.log({
                'Validation/Loss': val_loss,
                'Validation/Psnr': val_psnr,
                'Validation/ssim': val_ssim,
                'Validation/Epoch': epoch + 1,
                'Validation/Images': wandb.Image(
                    val_images,
                    caption=f'Epoch {epoch+1} | T={t_mid}, Z={d_mid}'
                )
            })

            print(f'\nEpoch {epoch + 1} Summary: Validation Loss: {val_loss:.6f} | Validation PSNR: {val_psnr:.6f} | Validation SSIM: {val_ssim:.6f}')



    wandb.finish()



     

            
