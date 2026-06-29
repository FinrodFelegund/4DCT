import torch
from models.SPBF import FilterBank
from torch.amp import autocast
from data.dataset import CT4dDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
import wandb


def train(config: Dict):

    wandb.init(
        entity='ipmi',
        project='$DCT-Denoising',
        name=config['name'],
        config=config,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FilterBank(num_spatial=config.get('num_spatial'), num_temporal=config.get('num_temporal'), device=device)
    train_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('train_data')),
        transforms=CT4dDataset.get_train_transforms(),
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=4, collate_fn=CT4dDataset.collate_fn)
    
    validation_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('validation_data')),
        transforms=CT4dDataset.get_validation_transforms(),
    )

    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=4, collate_fn=CT4dDataset.collate_fn)
    
    sigmas_st, sigmas_r = model.parameters()
    optimizer = torch.optim.Adam([
        {'params': sigmas_st, 'lr': config.get('lr_st', 0.01)},
        {'params': sigmas_r, 'lr': config.get('lr_r', 0.005)},
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
        for clean, noisy in train_pbar:
            clean, noisy = clean.to(device), noisy.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                prediction = model(noisy)
                loss = criterion(prediction, clean)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
  

            train_pbar.set_postfix(loss=f'{loss.item():.6f}')
            train_loss += loss.item()


        kernel_sizes = model.get_kernel_size()
        for key, value in kernel_sizes.items():
            wandb.log({
                f'Train/{key}': value
            }, commit=False)
        sigmas = model.get_sigmas()

        for key, value in sigmas.items():
            wandb.log({
                f'Train/{key}-mean': sigmas[key]['mean'],
                f'Train/{key}-std': sigmas[key]['std']
            }, commit=False)


        wandb.log({
            'Train/Loss': train_loss / len(train_dataloader),
        }, commit=True)
 
        if epoch % 2 == 0:
            model.eval()
            psnr_metric.reset()
            ssim_metric.reset()
            val_loss = 0
            val_images = None
            val_pbar = tqdm(validation_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]')

            cpu_seed = torch.get_rng_state()
            gpu_seed = torch.cuda.get_rng_state()

            torch.manual_seed(seed=42)
            torch.cuda.manual_seed(seed=42)
            with torch.no_grad():
                for i, (clean, noisy) in enumerate(val_pbar):
                    clean, noisy = clean.to(device), noisy.to(device)
                    

                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        prediction = model(noisy)
                        val_loss += criterion(prediction, clean).item()

                    prediction_f32 = prediction.float()
                    clean_f32 = clean.float()
                    noisy_f32 = noisy.float()

                    psnr_metric.update(prediction_f32, clean_f32)

                    T = clean_f32.shape[2]
                    Z = clean_f32.shape[3]

                    for time in range(T):
                        for depth in range(Z):
                            prediction_slice = prediction_f32[:, :, time, depth, :, :]
                            clean_slice = clean_f32[:, :, time, depth, :, :]
                            ssim_metric.update(prediction_slice, clean_slice)

                    if val_images == None:
                        mid_t = T // 2
                        start_z = Z // 2
                        end_z = start_z + 5 if start_z + 5 < Z else start_z + 1
                        
                        prediction_slices = prediction_f32[0, :, mid_t, start_z:end_z, :, :].permute(1, 0, 2, 3)
                        clean_slices = clean_f32[0, :, mid_t, start_z:end_z, :, :].permute(1, 0, 2, 3)
                        noisy_slices = noisy_f32[0, :, mid_t, start_z:end_z, :, :].permute(1, 0, 2, 3)
                        val_images = make_grid(
                            tensor=torch.cat([prediction_slices, clean_slices, noisy_slices], dim=0),
                            nrow=end_z-start_z,
                            padding=3,
                            normalize=False,
                        )


            torch.set_rng_state(cpu_seed)
            torch.cuda.set_rng_state(gpu_seed)
            val_psnr = psnr_metric.compute().item()
            val_ssim = ssim_metric.compute().item()
            val_loss = (val_loss / len(validation_dataloader))

            wandb.log({
                'Validation/Loss': val_loss,
                'Validation/Psnr': val_psnr,
                'Validation/ssim': val_ssim,
                'Validation/Epoch': epoch + 1,
                'Validation/Images': wandb.Image(
                    val_images,
                    caption=f'Epoch {epoch+1} | T={mid_t}, Z={start_z}-{end_z}'
                )
            })

            print(f'\nEpoch {epoch + 1} Summary: Validation Loss: {val_loss:.6f} | Validation PSNR: {val_psnr:.6f} | Validation SSIM: {val_ssim:.6f}')



    wandb.finish()



     

            
