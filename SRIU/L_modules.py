import torch
import torch.nn as nn
import lightning as L
from typing import List, Tuple, Any, Dict, Optional
import utils2, losses, modules
import torch.nn.functional as F


class BaseGAN(L.LightningModule):
    def __init__(self, generator: nn.Module, discriminator: Optional[nn.Module],
                 rgb:bool, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator= discriminator
        self.rgb = rgb
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to('cuda')
        self.std = torch.tensor(std).view(1, -1, 1, 1).to('cuda')
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=[
                                'generator',
                                'discriminator'
                                'perceptual_loss', 
                                'input_normalizer', 
                                'ram_normalizer',
                                'mse',
                                'slope_fn'
                            ])
    
    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        return self.generator(lr_img)
    
    def training_step(self, batch:Any, batch_idx:int) -> Dict:
        lr, hr = self._prep_batch(batch)
        sr = self.forward(lr) #self(lr)
        
        if self.discriminator:
            # GAN step
            gen_opt, disc_opt = self.optimizers(use_pl_optimizer=True)  # type: ignore
            if self.lr_schedulers():
                gen_scheduler, disc_schedule = self.lr_schedulers() # type: ignore
            else: 
                gen_scheduler, disc_schedule = None, None
            disc_loss, d_real = self._disc_loss(hr, sr)
            self._optimize(self.discriminator, disc_opt, 
                           disc_schedule, disc_loss)
            gen_losses = self._gen_loss(hr, sr, d_real)
            gen_loss = sum(gen_losses)
            self._optimize(self.generator, gen_opt, 
                           gen_scheduler, gen_loss)
            lrs = {'gen_lr': gen_opt.param_groups[0]['lr'],
                   'disc_lr': disc_opt.param_groups[0]['lr']}
            self.log_dict(lrs, on_step=True,  prog_bar=False)
            losses = {"t_g_loss": gen_loss,
                      "t_d_loss": disc_loss}
            self.log_dict(losses,on_step=False, on_epoch=True, prog_bar=True)
            return losses
        else:
            # só PSNR (L1) pretreino
            [gen_loss] = self._gen_loss(hr, sr)
            gen_opt = self.optimizers()
            gen_scheduler= self.lr_schedulers()
            self._optimize(self.generator, gen_opt, 
                           gen_scheduler, gen_loss)
            gen_lr = gen_opt.param_groups[0]['lr']  # type: ignore
            self.log("lr", gen_lr, on_step=True, prog_bar=False)
            self.log("t_g_loss", gen_loss, on_epoch=True, prog_bar=True)
            return {"t_g_loss": gen_loss}
        
        
    def validation_step(self, batch: Dict, batch_idx: int)-> Dict:
        lr, hr = self._prep_batch(batch)
        sr     = self(lr)
        losses = self._val_loss(lr, hr, sr, batch_idx)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "lr": lr.detach(),
            "hr": hr.detach(),
            "sr": sr.detach()}
    
    def test_step(self, batch: Any, batch_idx: int)-> Dict:
        lr, hr = self._prep_batch(batch)
        sr     = self(lr)
        # sr = utils2.stich_by_clip(img=lr,
        #                     n =5, generator=self.generator, 
        #                     pad=40, scale=2)
        
        return {
            "lr": lr.detach(),
            "hr": hr.detach(),
            "sr": sr.detach()}
    
    # def predict_step(self, batch):
    #     lr_img = batch[0]  
    #     name = batch[1]
    #     sr_img = utils2.stich_by_clip(img=lr_img,
    #                         n =2, generator=self.generator, 
    #                         pad=40, scale=2)
            
    #     return lr_img, sr_img, name
    
    def _prep_batch(self, batch: Any) -> Tuple[torch.Tensor,torch.Tensor]:
        lr, hr = batch
        
        if self.rgb:
            lr = lr.flatten(0,1).expand(-1, 3, -1, -1)
            hr = hr.flatten(0,1).expand(-1, 3, -1, -1)
            return lr, hr
        else:
            return lr.flatten(0,1), hr.flatten(0,1)
    
    
    def _optimize(self, module, opt, lr_sched, loss, max_norm: float = 1.0):
        opt.zero_grad()
        self.manual_backward(loss)
        # total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
        
    #     scale = float(max_norm / (total_norm + 1e-12)) if total_norm > max_norm else 1.0
    #     clipped_flag = float(total_norm > max_norm)

    #     # logue por módulo (gen/disc) e por step
    #     prefix = "gen" if module is self.generator else "disc"
    #     self.log_dict(
    #     {
    #         f"{prefix}/grad_total_norm": float(total_norm),
    #         f"{prefix}/grad_clip_scale": scale,
    #         f"{prefix}/grad_clipped": clipped_flag,  # 0 ou 1
    #     },
    #     on_step=True, on_epoch=True, prog_bar=False
    # )
        
        opt.step()
        if lr_sched is not None:
            lr_sched.step()
    
    def _gen_loss(self, hr, sr, d_real=None):
        raise NotImplementedError()

    def _disc_loss(self, hr, sr):
        raise NotImplementedError()

    def _val_loss(self,lr, hr, sr, batch_idx):
        raise NotImplementedError()
    def _unormalize(self, x:torch.Tensor) ->torch.Tensor:
        return x *self.std+ self.mean
    def _normalize(self, x):
        return (x-self.mean)/self.std
    def _filter_outliers(self, batch, mse_limit: float = 18.70)-> Dict:
        lr = batch['lr']   # [B, 1, h, w]
        hr = batch['hr']   # [B, 1, H, W]

        # 1) Upsample do LR para a resolução do HR (lote inteiro)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear", align_corners=False)

        # 2) Máscara de finitos
        finite = torch.isfinite(lr_up) & torch.isfinite(hr)
        if not finite.any():
            # tudo inválido → não filtra nada para não quebrar o pipeline
            return batch

        # 3) MSE por amostra com máscara (nanmean)
        diff2 = (lr_up - hr) ** 2
        diff2 = diff2.masked_fill(~finite, float('nan'))
        # torch.nanmean disponível no PyTorch 2.3
        mse_per = torch.nanmean(diff2, dim=(1, 2, 3))  # [B]

        # 4) Índices válidos pelo limite
        valid_mask = torch.isfinite(mse_per) & (mse_per < mse_limit)
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()

        # Segurança: se nada passa no filtro, não derruba o batch
        if len(valid_idx) == 0:
            return batch

        # 5) Reconstrói o dict filtrado
        out = {
            'lr': lr[valid_idx],
            'hr': hr[valid_idx],
            'bounds': [batch['bounds'][i] for i in valid_idx],
            'crs':    [batch['crs'][i]    for i in valid_idx],
        }
        return out
    def _filter_nan_batch(self, batch: Dict)-> Dict:
        """Check and Filter NaN values from the HR samples and MSE outliers.

        Args:
            batch (Dict): Batch of samples

        Returns:
            Dict: Filtered batch of sample.
        """
        if utils2.check_hr_on_dict(batch):
            # batch = self._filter_outliers(batch) 
            valid_indices = [i for i in range(batch['hr'].shape[0]) 
                            if not torch.isnan(batch['hr'][i]).any()]
            return {
                'lr': batch['lr'][valid_indices],
                'hr': batch['hr'][valid_indices],
                'bounds': [batch['bounds'][i] for i in valid_indices],
                'crs': [batch['crs'][i] for i in valid_indices]}
            
        else: 
            valid_indices = [i for i in range(batch['lr'].shape[0]) 
                            if not torch.isnan(batch['lr'][i]).any()]
            return {
                'lr': batch['lr'][valid_indices],
                'bounds': [batch['bounds'][i] for i in valid_indices],
                'crs': [batch['crs'][i] for i in valid_indices]}
    
    
class MSRResNet(BaseGAN):
    def __init__(self, generator, lr:float, lr_decay:float,
                 rgb:bool, **kwargs):
        kwargs.pop("discriminator", None)
        super().__init__(generator=generator, rgb=rgb, discriminator=None, **kwargs)
        self.content_loss = nn.L1Loss()
        self.lr = lr
        self.lr_decay = lr_decay

        
    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, int(self.lr_decay), .5)
        return ([gen_opt], [gen_scheduler])
    
    
    def _gen_loss(self, hr:torch.Tensor, sr:torch.Tensor)-> List[torch.Tensor]:
        return [self.content_loss(hr, sr)]
    
    def _disc_loss(self, hr:torch.Tensor, sr:torch.Tensor):
        return super()._disc_loss(hr, sr)
    
    def _val_loss(self, lr:torch.Tensor, hr:torch.Tensor, sr:torch.Tensor, batch_idx: int)-> Dict:
        v_loss = self._gen_loss(hr, sr)
        loss = {'v_g_loss':v_loss[0]}
        return loss

class ESRGAN(BaseGAN):
    def __init__(self, generator, discriminator,
                 perceptual_loss, rgb, 
                 lr,  step_milestones, **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         rgb=rgb, **kwargs)
        
        self.perceptual_loss = perceptual_loss
        self.content_loss = nn.L1Loss()
        self.gen_rela_loss = losses.RelativisticAdversarialLoss(mode="generator")
        self.disc_rela_loss = losses.RelativisticAdversarialLoss(mode="discriminator")
        self.step_milestones = step_milestones
        self.lr = lr
        
    def configure_optimizers(self):
        if self.step_milestones:
            gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
            disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr/10)  #type: ignore
            gen_scheduler = torch.optim.\
                            lr_scheduler.MultiStepLR(gen_opt, 
                                                    milestones=self.step_milestones, 
                                                    gamma=0.5)
            disc_scheduler = torch.optim.\
                            lr_scheduler.MultiStepLR(disc_opt, 
                                                    milestones=self.step_milestones, 
                                                    gamma=0.5)
            return ([gen_opt, disc_opt], [gen_scheduler, disc_scheduler])
        else:
            gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
            disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr/10) #type: ignore
            return ([gen_opt, disc_opt], [])
    
    def _gen_loss(self, hr, sr, d_real) -> List[torch.Tensor]:
        d_fake_gen = self.discriminator(sr)                         #type: ignore
        loss_adv = self.gen_rela_loss(d_fake_gen, d_real.detach()) #d_real was used on the discriminator opt
        loss_percep = self.perceptual_loss(hr, sr)
        loss_content = self.content_loss(sr, hr)
        return [1e-2*loss_percep, 1e-3*loss_adv, 5e-2*loss_content]
        
    def _disc_loss(self, hr, sr) -> List[torch.Tensor]: 
        d_real = self.discriminator(hr)
        d_fake = self.discriminator(sr.detach()) #important to separate from gen optimazation
        disc_loss = self.disc_rela_loss(d_fake,d_real)
        return [disc_loss, d_real]

    def _val_loss(self, lr, hr, sr, batch_idx) -> Dict:
        disc_loss, d_real = self._disc_loss(hr,sr)
        loss_percep, loss_adv,loss_content = self._gen_loss(hr, sr, d_real)
        losses = {'v_g_loss':loss_percep + loss_adv + loss_content,
                  'v_d_loss':disc_loss,
                  'v_loss_percep': loss_percep,
                  'v_loss_adv':loss_adv,
                  'v_loss_content': loss_content}

        return losses
    

class DEM_MSRResNet(MSRResNet):
    def __init__(self, generator, lr, lr_decay, rgb,
                 input_normalizer, ram_normalizer, **kwargs):
        super().__init__(generator, lr, lr_decay, rgb, **kwargs)
        self.input_normalizer = input_normalizer
        self.ram_normalizer = ram_normalizer
        
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # teste proposto pelo gpt
                out[k] = v.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last) #v.to(device)
            else:
                out[k] = v  # deixa BoundingBox, crs etc. no CPU
        return out
    
    def _prep_batch(self, batch):
        batch = self._filter_nan_batch(batch)
        if utils2.check_hr_on_dict(batch):
            self.input_normalizer(batch, 'lr')
            self.ram_normalizer(batch, 'hr')
            lr_img, hr_img = batch['lr'], batch['hr']
            lr_img = self._normalize(lr_img)
            hr_img = self._normalize(hr_img)
            if self.rgb:
                lr_img = lr_img.expand(-1, 3, -1, -1)
                hr_img = hr_img.expand(-1, 3, -1, -1)
            # ativando o compile
            lr_img = lr_img.to(memory_format=torch.channels_last, non_blocking=True)
            hr_img = hr_img.to(memory_format=torch.channels_last, non_blocking=True)
            return lr_img, hr_img
        else:
            self.input_normalizer(batch, 'lr')
            lr_img = batch['lr']
            lr_img = self._normalize(lr_img) 
            if self.rgb:
                lr_img = lr_img.expand(-1, 3, -1, -1)
            lr_img = lr_img.to(memory_format=torch.channels_last, non_blocking=True)
            return lr_img
            
class DEM_ESRGAN(ESRGAN):
    def __init__(self, generator, discriminator,
                 perceptual_loss, rgb, lr, step_milestones,
                 input_normalizer, ram_normalizer, **kwargs):
        super().__init__(generator = generator, discriminator = discriminator,
                         perceptual_loss = perceptual_loss, rgb = rgb, lr=lr, 
                         step_milestones=step_milestones, **kwargs)
        
        self.input_normalizer = input_normalizer
        self.ram_normalizer = ram_normalizer
        
    # def predict_step(self, batch):
    #     lr, hr = self._prep_batch(batch)
    #     # name = batch[1]
    #     sr = utils2.stich_by_clip(img=lr,
    #                         n =5, generator=self.generator, 
    #                         pad=40, scale=2)
            
    #     return {
    #         "lr": lr.detach(),
    #         "sr": sr.detach(),
    #         "hr": hr.detach()}
    
 
    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # teste proposto pelo gpt
                out[k] = v.to(device, non_blocking=True)# .contiguous(memory_format=torch.channels_last) #v.to(device)
            else:
                out[k] = v  # deixa BoundingBox, crs etc. no CPU
        return out
    
    
    def _prep_batch(self, batch):
        batch = self._filter_nan_batch(batch)
        if utils2.check_hr_on_dict(batch):
            # batch = self._filter_outliers(batch)
            # from real values to [0,1]
            self.input_normalizer(batch, 'lr')
            self.ram_normalizer(batch, 'hr')
            lr_img, hr_img = batch['lr'], batch['hr']
            # same values as the pretrain gen
            lr_img = self._normalize(lr_img)
            hr_img = self._normalize(hr_img)
            if self.rgb:
                lr_img = lr_img.expand(-1, 3, -1, -1)
                hr_img = hr_img.expand(-1, 3, -1, -1)
            lr_img = lr_img.to(memory_format=torch.channels_last, non_blocking=True)
            hr_img = hr_img.to(memory_format=torch.channels_last, non_blocking=True)
            return lr_img, hr_img
        else:
            self.input_normalizer(batch, 'lr')
            lr_img = batch['lr']
            lr_img = self._normalize(lr_img) 
            if self.rgb:
                lr_img = lr_img.expand(-1, 3, -1, -1)
            lr_img = lr_img.to(memory_format=torch.channels_last, non_blocking=True)
            return lr_img

class DEM_ESRGAN_slope(DEM_ESRGAN):
    def __init__(self, generator, discriminator,
                 perceptual_loss, rgb, lr, step_milestones,
                 input_normalizer, ram_normalizer, slope_fn, alfa, **kwargs):
        super().__init__(generator = generator, discriminator = discriminator,
                         perceptual_loss = perceptual_loss, rgb = rgb, lr=lr, 
                         step_milestones=step_milestones,input_normalizer = input_normalizer ,
                         ram_normalizer = ram_normalizer, **kwargs)
        self.slope_fn = slope_fn
        self.alfa =alfa
        self.mse = nn.MSELoss()
        
    def _loss_slope_metric(self, hr, sr):
        with torch.cuda.amp.autocast(enabled=False):
            hr_01 = self._unormalize(hr)
            sr_01 = self._unormalize(sr)
            batch = { 'hr_m': hr_01.clone(), 'sr_m': sr_01.clone()}
            self.ram_normalizer.revert(batch, 'sr_m')
            self.ram_normalizer.revert(batch, 'hr_m')
            sr_phys = batch['sr_m']
            hr_phys = batch['hr_m']
            slope_sr = self.slope_fn(sr_phys)
            slope_hr = self.slope_fn(hr_phys)
            return  self.mse(slope_sr, slope_hr)

    def _gen_loss(self, hr, sr, d_real) -> List[torch.Tensor]:
        d_fake_gen = self.discriminator(sr)
        loss_adv = self.gen_rela_loss(d_fake_gen, d_real.detach()) 
        loss_percep = self.perceptual_loss(sr, hr)
        loss_content = self.content_loss(sr, hr)
        loss_slope = self._loss_slope_metric(hr, sr)
        
        return [loss_percep, 5e-3*loss_adv, 1e-2*loss_content, self.alfa*loss_slope]
    
    def _val_loss(self, lr, hr, sr, batch_idx) -> Dict:
        disc_loss, d_real = self._disc_loss(hr,sr)
        loss_percep, loss_adv,loss_content, loss_slope = self._gen_loss(hr, sr, d_real)
        gen_loss = loss_percep + loss_adv + loss_content + loss_slope
        losses = {'v_g_loss':gen_loss,
                  'v_d_loss':disc_loss,
                  'loss_percep': loss_percep,
                  'loss_adv':loss_adv,
                  'loss_content': loss_content,
                  'loss_slope': loss_slope}

        return losses
    
class DEM_MSRResNet_terrain(DEM_MSRResNet):
    def __init__(self, generator, lr, lr_decay, rgb, input_normalizer, 
                 ram_normalizer, terrain_fn, alfa, **kwargs):
        super().__init__(generator, lr, lr_decay, rgb, input_normalizer, 
                         ram_normalizer, **kwargs)
        self.terrain_fn = terrain_fn
        self.alfa = alfa
        self.mse = nn.MSELoss()
    
    def _terrain_loss(self, sr, hr):
        with torch.cuda.amp.autocast(enabled=False):
            hr_01 = self._unormalize(hr)
            sr_01 = self._unormalize(sr)
            batch = { 'hr_m': hr_01.clone(), 'sr_m': sr_01.clone()}
            self.ram_normalizer.revert(batch, 'sr_m')
            self.ram_normalizer.revert(batch, 'hr_m')
            sr_phys = batch['sr_m']
            hr_phys = batch['hr_m']
            terrain_sr = self.terrain_fn(sr_phys)
            terrain_hr = self.terrain_fn(hr_phys)
            return  self.mse(terrain_sr, terrain_hr)
        
    def _gen_loss(self, hr: torch.Tensor, sr: torch.Tensor) -> List[torch.Tensor]:
        loss_content = self.content_loss(sr, hr)
        loss_terrain = self._terrain_loss(sr, hr)
        
        return [loss_content + self.alfa*loss_terrain]

