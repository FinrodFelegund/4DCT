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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FilterBank(num_spatial=config.get('num_spatial'), num_temporal=config.get('num_temporal')).cuda()
    train_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('train_data')),
        transforms=CT4dDataset.get_train_transforms(dosage=config.get('dosage', 0.5)),
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.get('batch_size', 1), shuffle=True, num_workers=0, collate_fn=CT4dDataset.collate_fn)
    
    validation_dataset = CT4dDataset(
        dataframe=CT4dDataset.generate_dataframe(config.get('validation_data')),
        transforms=CT4dDataset.get_validation_transforms(dosage=config.get('dosage', 0.5)),
    )

    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=0, collate_fn=CT4dDataset.collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    criterion = torch.nn.HuberLoss().to(device)
    scaler = torch.amp.GradScaler(device=device.type)

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


        wandb.log({
            'Train/Loss': train_loss / len(train_dataloader),
        })
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()
        val_loss = 0
        val_images = None
        val_pbar = tqdm(validation_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]')

        if epoch % 2 == 0:
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


                    if (i + 1) % 5 == 0:
                        break

            val_psnr = psnr_metric.compute().item()
            val_ssim = ssim_metric.compute().item()
            val_loss = (val_loss / 5) #len(validation_dataloader)).item()

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



     

            
