from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
import utils2
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from torchmetrics import MeanSquaredError as MSE
from typing import Union, Any, Optional, Callable, Mapping
import torch, math
import torch.nn.functional as F
from shapely.geometry import box
import geopandas as gpd
import os, fiona



class Compute_metrics(Callback):
    def __init__(self, ram_norm:Optional[torch.nn.Module], 
                 input_norm:Optional[torch.nn.Module], scale: int):
        super().__init__()
        # self.psnr = PSNR(data_range=1.0).to('cuda').float()
        # self.ssim = SSIM(data_range=1.0).to('cuda').float()
        self.mse_bic = MSE().to('cuda')
        self.mse_m = MSE().to('cuda')
        self.mse_nn = MSE().to('cuda')
        self.ram_norm = ram_norm
        self.input_norm = input_norm
        self.scale = scale
        #teste bicubico
        self._psnr_bic_sum = 0.0; self._ssim_bic_sum = 0.0;  self._n = 0
        self._psnr_m_sum = 0.0; self._ssim_m_sum = 0.0
        self._psnr_nn_sum = 0.0; self._ssim_nn_sum = 0.0
    
    def _compute_metrics(self, hr, sr, lr):
        # self.psnr.update(sr,hr)
        # self.ssim.update(sr,hr)
        # self.mse.update(sr,hr)
        if self.ram_norm:
            hr_m, sr_m, lr_m = self._to_metros(hr, sr, lr)
        else:
            hr_m, sr_m, lr_m = hr, sr, lr
            
        data_range = float((hr_m.max() - hr_m.min()).clamp_min(1e-6))
        self._psnr_m_sum += psnr_fn(sr_m, hr_m, data_range=data_range).item()
        self._ssim_m_sum += ssim_fn(sr_m, hr_m, data_range=data_range).item()   # type: ignore
        self.mse_m.update(sr_m,hr_m)
    
    # if lr is not None:
        bic = F.interpolate(lr_m, scale_factor=self.scale, mode='bicubic', align_corners=False, antialias=False)
        nn = F.interpolate(lr_m, scale_factor=self.scale, mode='nearest')
        # data_range = float((hr_m.max() - hr_m.min()).clamp_min(1e-6))
        self._psnr_bic_sum += psnr_fn(bic, hr_m, data_range=data_range).item()
        self._ssim_bic_sum += ssim_fn(bic, hr_m, data_range=data_range).item()  # type: ignore
        self._psnr_nn_sum += psnr_fn(nn, hr_m, data_range=data_range).item()
        self._ssim_nn_sum += ssim_fn(nn, hr_m, data_range=data_range).item()    # type: ignore
        self.mse_bic.update(bic, hr_m); self.mse_nn.update(nn, hr_m)
        self._n += 1
        
    
    def _restore_01(self, pl_module, hr, sr, lr):
        # desfaz apenas z-score → volta para [0,1]
        return pl_module._unormalize(hr), pl_module._unormalize(sr), pl_module._unormalize(lr) 
        
    def _to_metros(self, hr_01, sr_01, lr_01):
        # if self.ram_norm is None or self.input_norm is None:
        #     return None, None, None
        batch = { 'hr': hr_01.clone(), 'sr': sr_01.clone(), 'lr': lr_01.clone()}
        self.ram_norm.revert(batch, 'sr')                                       # type: ignore
        self.ram_norm.revert(batch, 'hr')                                       # type: ignore
        self.input_norm.revert(batch, 'lr')                                     # type: ignore
        return batch['hr'], batch['sr'], batch['lr']
    
    def _log_epoch_metrics(self, pl_module, prefix):
        mse_m = self.mse_m.compute()
        metrics={
            # f'{prefix}/mse' : mse_m,
            f'{prefix}/rmse': math.sqrt(mse_m),
            # f'{prefix}/psnr': self.psnr.compute(),
            # f'{prefix}/ssim': self.ssim.compute(),
            # f'{prefix}/mse' : self.mse.compute(),
            }
        
        if self._n > 0:
            mse_bic = self.mse_bic.compute()
            mse_nn = self.mse_nn.compute()
            metrics[f'{prefix}/rmse_bic'] = math.sqrt(mse_bic)
            metrics[f'{prefix}/rmse_nn'] = math.sqrt(mse_nn)

            metrics[f'{prefix}/psnr_bic'] = self._psnr_bic_sum / self._n
            metrics[f'{prefix}/psnr_nn'] = self._psnr_nn_sum / self._n
            metrics[f'{prefix}/psnr'] = self._psnr_m_sum/self._n
            
            metrics[f'{prefix}/ssim_bic'] = self._ssim_bic_sum / self._n
            metrics[f'{prefix}/ssim_nn'] = self._ssim_nn_sum / self._n
            metrics[f'{prefix}/ssim'] = self._ssim_m_sum/self._n
        pl_module.log_dict(metrics, prog_bar=True)
        # self.psnr.reset(); self.ssim.reset(); self.mse.reset(); 
        self.mse_m.reset()
        self._psnr_bic_sum = self._ssim_bic_sum = 0.0; self._n = 0
        self._psnr_m_sum = self._ssim_m_sum = 0.0
        
    ##### NÃO PRECISO CALCULAR ESSAS MÉTRICAS PARA A VALIDAÇÃO. APENAS NO TESTE.
    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
    #     hr, sr = self._restore_real_range(pl_module, outputs['hr'],outputs['sr'] )
    #     hr_01, sr_01, lr_01 = self._restore_01(pl_module, outputs['hr'],outputs['sr'], outputs['lr'])
    #     self._compute_metrics(hr = hr_01 , sr = sr_01, lr= lr_01 )
    
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     self._log_epoch_metrics(pl_module, 'val')
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        hr_01, sr_01, lr_01 = self._restore_01(pl_module, outputs['hr'],outputs['sr'], outputs['lr'])
        self._compute_metrics(hr = hr_01 , sr = sr_01, lr= lr_01 )
            
    def on_test_epoch_end(self, trainer, pl_module):
        self._log_epoch_metrics(pl_module, 'test')
    
class Plot_results(Callback):
    def __init__(self,val_batch_interval:int, test_batch_interval:int, 
                 cmap:Optional[str]=None, ram_norm:Optional[Callable[...,torch.nn.Module]]=None, 
                 input_norm: Optional[Callable[...,torch.nn.Module]]=None,
                 prefix=''):
        super().__init__()
        self.val_batch_interval = val_batch_interval
        self.test_batch_interval = test_batch_interval        
        self.ram_norm = ram_norm
        self.input_norm = input_norm
        self.cmap = cmap
        self.prefix = prefix

    
    def _restore_real_range(self, pl_module, outputs):
        hr = pl_module._unormalize(outputs['hr'])
        sr = pl_module._unormalize(outputs['sr'])
        lr = pl_module._unormalize(outputs['lr'])
        if self.ram_norm:
            batch = {'lr': lr, 'hr': hr, 'sr': sr}
            self.ram_norm.revert(batch, 'sr')
            self.ram_norm.revert(batch, 'hr')
            self.input_norm.revert(batch, 'lr')
            lr, sr, hr = batch['lr'], batch['sr'], batch['hr']
        return lr, sr, hr
    
    def _maybe_plot(self, pl_module, stage: str, outputs:dict, epoch:int, batch_idx:int):
        if stage == 'val':
            intervalo = self.val_batch_interval
            pasta = 'SRIU/artifacts/val/'+self.prefix
        else:
            intervalo = self.test_batch_interval
            pasta = 'SRIU/artifacts/val/'+self.prefix

        lr, sr, hr = self._restore_real_range(pl_module, outputs)

        utils2.plot_random_crops_lr_sr_hr(
            lr, sr, hr, epoch, batch_idx, pasta, self.cmap)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if batch_idx%self.val_batch_interval==0:
            # torch.save(batch, f'SRIU/artifacts/dicts/val_e_{pl_module.current_epoch}_{batch_idx}')
            # self._maybe_plot(pl_module, 'val', outputs, pl_module.current_epoch, batch_idx )
            lr, sr, hr = self._restore_real_range(pl_module, outputs)
            utils2.val_plot_dem_hr_sr_dif(sr_imgs=sr, hr_imgs=hr, batch_idx=batch_idx,
                                          epoch= pl_module.current_epoch,
                                          save_img_path='SRIU/artifacts/val/')

            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if batch_idx%self.test_batch_interval==0:
            # torch.save(batch, f'SRIU/artifacts/dicts/test_{batch_idx}')
            lr, sr, hr = self._restore_real_range(pl_module, outputs)
                        
            if sr.shape == hr.shape:
                utils2.plot_dem_predictions_mopa(lr=lr, sr=sr, hr=hr, batch_idx=batch_idx,
                                         save_img_path='SRIU/artifacts/test/')
    
    # def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # hr_img = self.hr_ds.__getitem__(batch['bounds'][0])['image'].to('cuda')
        # outputs['hr'] = hr_img.detach()
        # lr, sr, hr = self._restore_real_range(pl_module, outputs)
        # if sr.shape == hr.shape:
        #     utils2.plot_dem_predictions_mopa(lr=lr, sr=sr, hr=hr, batch_idx=batch_idx,
        #                                  save_img_path='SRIU/artifacts/pred/')
        
class Save_bbox(Callback):
    def __init__(self, gpkg_path, layer_name) -> None:
        super().__init__()
        self.gpkg_path = gpkg_path
        self.layer_name = layer_name
        self._reset_store()
        
    def _reset_store(self):
        self._bounds = {'epoch': [], 'batch': [], 'disc': [], 'bounds': []}
        
    def _save_bounds(self, bounds: list, phase: str, epoch:int, batch_idx:int):
        for idx in range(len(bounds)):
            self._bounds['epoch'].append(epoch)
            self._bounds['batch'].append(batch_idx)
            self._bounds['bounds'].append(bounds[idx])
            self._bounds['disc'].append(phase)
    
    def _create_save_gpkg_layer(self, dict_bounds: dict,gpkg_path: str ,layer_name:str):
        if 'bounds' not in dict_bounds or len(dict_bounds['bounds']) == 0:
            return
        local = {k: v[:] if isinstance(v, list) else v for k, v in dict_bounds.items()}
        geoms = [box(b.minx, b.miny, b.maxx, b.maxy) for b in local['bounds']]
        gdf = gpd.GeoDataFrame(
            {k: v for k, v in local.items() if k != 'bounds'},
            geometry=geoms,
            crs='EPSG:4326'
        )
        if os.path.exists(gpkg_path) and (layer_name in fiona.listlayers(gpkg_path)):
            mode = 'a'
        else:
            mode = 'w'
        gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode=mode, index=False)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        batch_filtered = pl_module._filter_nan_batch(batch)
        self._save_bounds(batch_filtered['bounds'], 'train', pl_module.current_epoch, batch_idx)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        if trainer.sanity_checking:
            return
        batch_filtered = pl_module._filter_nan_batch(batch)
        self._save_bounds(batch_filtered['bounds'], 'val', pl_module.current_epoch, batch_idx)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        self._save_bounds(batch['bounds'], 'test', pl_module.current_epoch, batch_idx)
    
    # def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     if trainer.sanity_checking:
    #         return
    #     self._create_save_gpkg_layer(self._bounds, self.gpkg_path, self.layer_name)
    #     self._reset_store()
        
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._create_save_gpkg_layer(self._bounds, self.gpkg_path, self.layer_name)
        # self._reset_store()
    
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mask = [d == "test" for d in self._bounds['disc']]
        # Filtrar mantendo apenas onde disc == "test"
        filtered_bounds = {k: [v for v, m in zip(vals, mask) if m]
                   for k, vals in self._bounds.items()}
        self._create_save_gpkg_layer(filtered_bounds, self.gpkg_path, self.layer_name )
        self._reset_store()


class ResetBestScoresCallback(Callback):
    def __init__(self, reset_epoch: int):
        super().__init__()
        self.reset_epoch = reset_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        # reseta assim que termina a época `reset_epoch`
        if trainer.current_epoch == self.reset_epoch+1:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    cb.best_model_score = torch.tensor(float('inf'))
                if isinstance(cb, EarlyStopping):
                    cb.best_score = torch.tensor(float('inf'))
                    cb.wait_count = 0

