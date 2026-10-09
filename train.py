import os
import warnings
from typing import Dict

import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

from data.dataset import CTTrainDataset, CTValidationDataset
from models.filterbank import FilterBank
from models.cnn import CNN
from utils.processing import interior
from utils.explore_sigmas import sweep_sigmas

warnings.filterwarnings('ignore', category=DeprecationWarning)

LUNG_LO, LUNG_HI = -1250.0, 250.0
SOFT_LO, SOFT_HI = -160.0, 240.0
def to_display(x, lo, hi):
    hu = x * 4024.0 - 1024.0
    return ((hu - lo) / (hi - lo)).clamp(0.0, 1.0) 

RANGE_SUFFIXES = ('color_sigma', 'sigma_r')
LUT_SUFFIXES = ('raw_lut', 'lut')
KERNEL_SUFFIXES = ('kernel', )
CENTER_SUFFIXES = ('center_alpha',)

def group_of(name: str) -> str:
    if name.endswith(RANGE_SUFFIXES):
        return 'range'
    if name.endswith(LUT_SUFFIXES):
        return 'lut'
    if name.endswith(KERNEL_SUFFIXES):
        return 'kernel'
    if name.endswith(CENTER_SUFFIXES):
        return 'center'
    return 'spatiotemporal'

def learning_rates(config):
    optim_cfg = config.get('optim', {})
    return {
        'spatial': optim_cfg.get('lr_sp', 0.01),
        'temporal': optim_cfg.get('lr_t', 0.01),
        'spatiotemporal': optim_cfg.get('lr_spt', 0.01),
        'range': optim_cfg.get('lr_r', 0.005),
        'center': optim_cfg.get('lr_center', 0.01),
        'kernel': optim_cfg.get('lr_kernel', 1e-3),
        'lut': optim_cfg.get('lr_lut', 1e-3),
        'network': optim_cfg.get('lr_network', 1e-4)
    }

def build_optimizer(model, config):
    lr = learning_rates(config)
    groups = model.parameter_groups(lr)
    for g in groups:
        print(f'optimizer group {g['name']}: {sum(p.numel() for p in g['params'])} values, lr {g['lr']}')
    return torch.optim.Adam(groups)

@torch.no_grad()
def clamp_parameters(model: torch.nn.Module, config: Dict):
    clamp_cfg = config.get('clamp', {})
    sigma_min = clamp_cfg.get('sigma_min', 0.01)
    sigma_max = clamp_cfg.get('sigma_max', 2.0)
    range_min = clamp_cfg.get('range_min', 1e-4)
    range_max = clamp_cfg.get('range_max', 2.0)
    center_min = clamp_cfg.get('center_min', 0.0)
    center_max = clamp_cfg.get('center_max', 10.0)
    kernel_min = clamp_cfg.get('kernel_min', 0.0)

    if isinstance(model, CNN):
        return

    for name, param in model.named_parameters():
        if name.rsplit('.', 1)[-1].startswith('log_'):
            continue
        group = group_of(name)
        if group == 'range':
            param.clamp_(min=range_min, max=range_max)
        elif group == 'spatiotemporal':
            param.clamp_(min=sigma_min, max=sigma_max)
        elif group == 'center':
            param.clamp_(min=center_min, max=center_max)
        elif group == 'kernel':
            param.clamp_(min=kernel_min)
        elif group == 'skip':
            continue

def preview_grid(model, validation_dataset, device, crop=192):
    """Centre slice of ground truth, prediction and input, side by side."""
    clean, noisy = validation_dataset.get_whole_slice()
    H, W = clean.shape[-2:]
    y0, x0 = max((H - crop) // 2, 0), max((W - crop) // 2, 0)
    clean, noisy = clean[..., y0:y0 + crop, x0:x0 + crop].to(device), noisy[..., y0:y0 + crop, x0:x0 + crop].to(device)
    forward = getattr(model, 'forward_volume', model)
    pred = forward(noisy)
    t = pred.shape[2] // 2
    d = pred.shape[3] // 2

    panel = torch.cat([clean[:, :, t, d], pred[:, :, t, d], noisy[:, :, t, d]], dim=0)
    grid_lung = make_grid(to_display(panel, lo=LUNG_LO, hi=LUNG_HI), nrow=3, normalize=False, pad_value=1.0)
    grid_soft = make_grid(to_display(panel, lo=SOFT_LO, hi=SOFT_HI), nrow=3, normalize=False, pad_value=1.0)
    return (grid_lung * 255).round().to(torch.uint8).cpu(), (grid_soft * 255).round().to(torch.uint8).cpu()

def validate(model, dataloader, dataset, criterion, psnr_metric, ssim_metric, device, amp_dtype, epoch, epochs, margin, chunk=8):
    model.eval()
    psnr_metric.reset()
    ssim_metric.reset()
    val_loss, n_scans, grid_lung, grid_soft = 0.0, 0, None, None
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs} [VALIDATION]')

    with torch.no_grad():
        for i, (clean, noisy) in enumerate(pbar):
            scan_loss, scan_crops = 0.0, 0
            for c, n in zip(clean.split(chunk), noisy.split(chunk)):
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    c, n = interior(c.to(device), margin), n.to(device)
                    prediction = interior(model(n), margin)

                    scan_loss += criterion(prediction, c).item() * c.shape[0]
                    scan_crops += c.shape[0]
                    psnr_metric.update(prediction, c)

                    B, C, T, D, H, W = prediction.shape
                    ssim_metric.update(prediction.reshape(B * T * D, C, H, W),
                                    c.reshape(B * T * D, C, H, W))
                    
                del c, n, prediction
            val_loss += scan_loss / scan_crops
            n_scans += 1
            pbar.set_postfix(Image=i + 1, Patches=scan_crops)

            if grid_lung is None and grid_soft is None:
                grid_lung, grid_soft = preview_grid(model, dataset, device)

    model.train()
    return (val_loss / max(n_scans, 1),
            psnr_metric.compute().item(),
            ssim_metric.compute().item(),
            grid_lung, grid_soft)

def resolve_amp(config: Dict):
    amp_cfg = config.get('amp', False)
    if amp_cfg is False:
        return None
    
    stage_types = {stage['type'] for stage in config['model']['stages']}
    if stage_types & {'learnable_spt4d', 'lut4d', 'ndlut', 'nd'}:
        print('amp disabled: a selected stage has no half-precision CUDA dispatch.')
        return None
    
    return {'float16': torch.float16, 'bfloat16': torch.bfloat16}[
        config.get('amp_dtype', 'float16')]

def build(model, path, DEVICE):
    params = torch.load(path, map_location=DEVICE)
    model.load_state_dict(params)
    return model

def train(config: Dict):
    wandb.init(entity=config.get('wandb_entity', 'ipmi'),
               project=config.get('wandb_project', '4DCT-Denoising'),
               name=config['name'], config=config)
    
    device = torch.device('cuda')
    model_cfg = config.get('model')
    if isinstance(model_cfg, dict):
        model = CNN(model_cfg['stages']).to(device)
    else:
        model = FilterBank(model_cfg['stages']).to(device)
    print(model)

    transform_params = config.get('crop')
    margin = transform_params.get('margin')
    dataframe = CTTrainDataset.generate_dataframe(config['train_data'])
    dataframe = dataframe.sample(frac=1.0, random_state=config.get('split_seed', 0))
    n_train = int(config.get('num_train_scans', 30))

    train_dataset = CTTrainDataset(dataframe=dataframe.iloc[:n_train],
                                   transforms=CTTrainDataset.get_train_transforms(**transform_params))
    
    validation_dataset = CTValidationDataset(dataframe=dataframe[n_train:],
                                             transforms=CTValidationDataset.get_validation_transforms(**transform_params),
                                             scan_transforms=CTValidationDataset.get_scan_transforms())
    wandb.config.update({'validation_scans': list(dataframe.index[n_train:])})

    batch_size = config.get('batch_size', 1)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=config.get('num_workers', 4),
                                  collate_fn=CTTrainDataset.collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=1, collate_fn=CTValidationDataset.collate_fn)


    optimizer = build_optimizer(model, config)
    amp_dtype = resolve_amp(config)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_dtype == torch.float16)
    criterion = torch.nn.HuberLoss().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    reg_weight = config.get('reg_weight', 0.0)
    clip_norm = config.get('clip_grad_norm', 1.0)
    validate_every = config.get('validate_every', 2)
    epochs = config.get('epochs', 10)
    parameter_dir = config['parameter_dir']
    os.makedirs(parameter_dir, exist_ok=True)

    best_loss, best_checkpoint = float('inf'), ''
    model.train()

    for epoch in range(epochs):

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [TRAINING]')
        epoch_loss = 0.0

        for i, (clean, noisy) in enumerate(pbar):
            clean, noisy = clean.to(device), noisy.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                prediction = model(noisy)
                loss = criterion(interior(prediction, margin), interior(clean, margin))
                if reg_weight:
                    penalty = model.regularisation()
                    if penalty is not None:
                        loss = loss + reg_weight * penalty

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
            clamp_parameters(model, config)

            logged = {f'Parameters/{k}': v for k, v in model.scalar_parameters().items()}
            logged['Train/Loss'] = loss.item()
            wandb.log(logged)

            pbar.set_postfix(Loss=f'{loss.item():.6f}')
            epoch_loss += loss.item()

        wandb.log({'Train/Mean-Loss-Epoch': epoch_loss / len(train_dataloader),
                   'Train/Epoch': epoch + 1})
    
        if epoch % validate_every and epoch != epochs - 1:
            continue

        val_loss, val_psnr, val_ssim, grid_lung, grid_soft = validate(
            model, validation_dataloader, validation_dataset, criterion,
            psnr_metric, ssim_metric, device, amp_dtype, epoch, epochs, margin
        )

        wandb.log({'Validation/Loss': val_loss, 'Validation/psnr': val_psnr,
                   'Validation/ssim': val_ssim, 'Validation/Epoch': epoch + 1,
                   'Validation/Lung': wandb.Image(grid_lung, caption=f'Epoch {epoch + 1} | Lung Window'),
                   'Validation/SoftTissue': wandb.Image(grid_soft, caption=f'Epoch {epoch + 1} | Soft Tissue Window')})
        
        print(f'\nEpoch {epoch + 1}: val loss {val_loss:.6f} | '
              f'PSNR {val_psnr:.6f} | SSIM {val_ssim:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            new_checkpoint = os.path.join(
                parameter_dir, f'{config.get('name')}_epoch{epoch + 1}_loss{val_loss:.6f}.pth'
            )
            torch.save({
                'model': model.state_dict(), 'config': config,
                'epoch': epoch, 'val_loss': val_loss,
            }, new_checkpoint)
            if best_checkpoint and os.path.exists(best_checkpoint):
                os.remove(best_checkpoint)
            best_checkpoint = new_checkpoint
    
    wandb.finish()
    return best_checkpoint
