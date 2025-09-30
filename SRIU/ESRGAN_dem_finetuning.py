import os
import torch
from torch import optim, nn, Tensor
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import models, modules, losses, utils2
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from torchgeo.datasets import RasterDataset, random_bbox_assignment, VectorDataset, BoundingBox, GeoDataset
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
import rasterio.features
from rasterio.transform import rowcol
from rtree import index as rtree_index
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from typing import Any, Dict
import numpy as np
from shapely.geometry import box
import torch._dynamo





# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)
L.seed_everything(42, workers=True)

# Classes
class DEM(RasterDataset):
    is_image = True

class InMemoryGeoRaster(RasterDataset):
    """
    Subclasse de RasterDataset que carrega todo o GeoTIFF em memória,
    para que __getitem__(query) retorne patches sem abrir o arquivo a cada vez.
    """
    is_image = True

    def __init__(self, path: str):
        # Chama o __init__ do RasterDataset (configura CRS, bounds etc.)
        super().__init__(paths=path)

        # Abre apenas UMA vez e armazena o array e transform
        with rasterio.open(path) as src:
            self._array     = src.read(1)          # numpy array de shape (H, W)
            self._transform = src.transform        # Affine transform
            self.crs        = src.crs              # necessário para GeoDataset
            self.bbox     = src.bounds           # (minx, miny, maxx, maxy)
        self._height, self._width = self._array.shape       
     
    def __len__(self):
        return 1
    
    @property
    # def bounds(self):
    #     return self.bbox
    def bounds(self) -> BoundingBox:
        # agora retorna um BoundingBox com tempo definido (mint=0, maxt=0)
        left, bottom, right, top = self.bbox
        return BoundingBox(
            minx=left, maxx=right,
            miny=bottom, maxy=top,
            mint=0,    maxt=0
        )
        
    def __getitem__(self, query: BoundingBox) -> dict[str, torch.Tensor]:
        """
        Recebe um BoundingBox (minx, miny, maxx, maxy, mint, maxt),
        converte para índices de pixel e fatia self._array na RAM.
        """

        # 1) Extrai limites em coordenadas de mundo
        try:
            minx, miny, maxx, maxy = query.left, query.bottom, query.right, query.top
        except:
            minx, miny, maxx, maxy = query.minx, query.miny, query.maxx, query.maxy

        # 2) Converte (x, y) → (row, col)
        row0, col0 = rowcol(self._transform, minx, maxy)  # canto superior-esq
        row1, col1 = rowcol(self._transform, maxx, miny)  # canto inferior-dir

        # 3) Ordena índices para garantir slice correto
        row_start, row_stop = sorted((row0, row1))
        col_start, col_stop = sorted((col0, col1))

        # 4) Fatia o array em memória: shape resultante (h, w)
        patch_np = self._array[row_start:row_stop, col_start:col_stop]

        # 5) Converte para Tensor e adiciona canal único: (1, h, w)
        patch_t = torch.from_numpy(patch_np).unsqueeze(0).float()

        return {
            "image": patch_t,
            "crs":   self.crs,
            "bounds": query,
        }
    
class CustomDEMDataset(GeoDataset):
    def __init__(self, dataset, ram_ds):
        super().__init__(transforms=None)
        # composto em 30 m
        self.dataset = dataset 
        # referência em 15 m
        self.hr_ds = ram_ds           
        # grade espacial total (CRS + bounds) vinda do LR
        self.crs = dataset.crs     
        self.bbox = dataset.bounds
        self.index = dataset.index
        self.res = dataset.res  
        
    @property
    def bounds(self):
        return self.bbox  

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        # LR já veio em 30 m sem rampear o próprio ram_ds
        lr_patch = self.dataset[query]["image"]
        # HR em 15 m, diretamente do ram_ds
        if self.hr_ds:
            hr_patch = self.hr_ds[query]["image"]
            return {
                "lr": lr_patch,
                "hr": hr_patch,
                "bounds": query,
                "crs": self.crs,
            }
        else:
            return {
                "lr": lr_patch,
                "bounds": query,
                "crs": self.crs,
            }
            

class VectorDataset_GDF(VectorDataset):
    def __init__(self, path, layer, res=0.0001,  crs=None):
        GeoDataset.__init__(self, transforms=None)
        self.gdf = gpd.read_file(path, layer=layer)
        self.crs = crs or self.gdf.crs
        self.res = res

        # monta um R-tree 3D (x, y, t)
        prop = rtree_index.Property()
        prop.dimension = 3
        self.index = rtree_index.Index(properties=prop)

        for i, geom in enumerate(self.gdf.geometry):
            minx, miny, maxx, maxy = geom.bounds
            # para t usamos 0…0
            coords3d = (minx, miny, 0, maxx, maxy, 0)
            self.index.insert(i, coords3d, obj=i)
            
    @property
    def bounds(self) -> BoundingBox:
        minx, miny, maxx, maxy = self.gdf.total_bounds
        # mint e maxt são temporais. Como não estamos usando tempo, definimos como 0
        return BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=0, maxt=0)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        # recupera índices das geometrias que intersectam a janela
        hits = list(self.index.intersection((query.minx, query.miny,
                                              query.maxx, query.maxy,
                                              query.mint, query.maxt),
                                             objects=True))
        if not hits:
            # retorna máscara vazia
            h = round((query.maxy - query.miny) / self.res)
            w = round((query.maxx - query.minx) / self.res)
            mask = torch.zeros((1, h, w), dtype=torch.uint8)
            return {'aoi': mask, 'crs': self.crs, 'bounds': query}

        shapes = []
        for hit in hits:
            idx = hit.object
            geom = self.gdf.geometry.iloc[idx]
            props =  1 #self.gdf.iloc[idx][self.label_name]
            # converte shapely → geojson para rasterio
            # geojson = mapping(geom.to_crs(self.crs)) todas as entradas estão no mesmo CRS
            geojson = mapping(geom)
            shapes.append((geojson, props))

        # rasterização
        height = round((query.maxy - query.miny) / self.res)
        width  = round((query.maxx - query.minx) / self.res)
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy,
            width, height
        )
        mask = rasterio.features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {'aoi': mask, 'crs': self.crs, 'bounds': query}
    

# Funções 
def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        else:
            # assume metadados como crs, bbox etc., só replica o primeiro
            # collated[key] = batch[0][key]
            collated[key] = [sample[key] for sample in batch]
    # tensor of size [A,1,patch_size,patch_size]
    # collated['image'] = collated['image'].reshape(-1,patch_size,patch_size).unsqueeze(1)
    return collated

def inicialize_normalizer(ds: RasterDataset, label: str):
    ds_dict = ds.__getitem__(ds.bounds)
        # filtering NaN pixels values.
    tensor = ds_dict['image']
    tensor[tensor < 0] = float('nan') 
    # mean = torch.nanmean(tensor,dim=(1,2))
    # std = utils2.nanstd(tensor, dim=(1,2))
    # return utils2.Normalizer(mean,std) 
    if tensor.isnan().any():    
        flat = tensor.view(1,-1)
        mask = torch.isnan(flat)

        filled_min = flat.clone()
        filled_min[mask] = float('inf')
        min_global,_ = filled_min.min(dim=1)

        filled_max = flat.clone()
        filled_max[mask] = float('-inf')
        max_global, _ = filled_max.max(dim=1)
    else:
        min_global = torch.min(tensor).unsqueeze(0)
        max_global = torch.max(tensor).unsqueeze(0)
    
        #create normalizer
    
    return utils2.Normalizer_minmax(min_global, max_global)

def log_model_state_dict(run_id: str)-> None:
    # get a model saved in mlflow and saves only the weigths
    model_uri = f"runs:/{run_id}/best_model"  # Substitua pelo seu run_id
    mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")
    best_model = mlflow.pytorch.load_model(model_uri)
    
    torch.save(best_model.generator.state_dict(), "generator_weights.pth")
    torch.save(best_model.discriminator.state_dict(), "discriminator_weights.pth")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact("generator_weights.pth", artifact_path="weights")
        mlflow.log_artifact("discriminator_weights.pth", artifact_path="weights")

def create_save_gpkg_layer(dict_bounds,gpkg_path: str ,layer_name:str):
    dict_bounds['geometry'] = [box(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy) for bbox in dict_bounds['bounds']]
    del dict_bounds['bounds']
    gdf = gpd.GeoDataFrame(dict_bounds, crs = 'EPSG:4326')
    gdf.to_file(gpkg_path,layer=layer_name, driver="GPKG")



def ESRGAN_DEM_FineTuning(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="ESRGAN_DEM_FineTuning",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')

    # losses
    if args.unet_local_weigths:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}, #wang et al 21, usa múltiplas camadas aqui.
                        weights_path=args.unet_local_weigths)
    else:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0})
    gen_rela_loss = losses.RelativisticAdversarialLoss(mode="generator")
    disc_rela_loss = losses.RelativisticAdversarialLoss(mode="discriminator")
    content_loss = nn.L1Loss()
    
    class DEM_finetuning(L.LightningModule):
        def __init__(self, generator, discriminator, input_normalizer, ram_normalizer):
            super().__init__()
            self.generator = generator
            self.discriminator= discriminator
            self.input_normalizer = input_normalizer
            self.ram_normalizer = ram_normalizer
            
            self.perceptual_loss = perceptual_loss
            self.gen_rela_loss = gen_rela_loss
            self.disc_rela_loss = disc_rela_loss
            self.content_loss = content_loss
            self.scale = 2
            self.resize_transform = Resize((224, 224))
            
            self._bounds = {'epoch':[],
                            'batch':[],
                            'disc':[],
                            'bounds':[]}
            # this is the line that enables manual optmization
            self.automatic_optimization = False
            
        def on_fit_start(self):
            #teste de otimização
            self.generator.to(memory_format=torch.channels_last)
            self.discriminator.to(memory_format=torch.channels_last)
            self.perceptual_loss.to(memory_format=torch.channels_last)
            
            torch._dynamo.config.suppress_errors = True
            self.generator = torch.compile(
                self.generator,
                mode="default",           
                backend="aot_eager",        
            )

            self.discriminator = torch.compile(
                self.discriminator,
                mode="default",
                backend="aot_eager",
            )

            self.perceptual_loss = torch.compile(
                self.perceptual_loss,
                mode="default",
                backend="aot_eager",
            )
        
        def configure_optimizers(self):
            gen_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
            disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
            # gen_scheduler = torch.optim.\
            #                 lr_scheduler.MultiStepLR(gen_opt, 
            #                                         milestones=[5000, 10000, 20000, 30000], 
            #                                         gamma=0.5 )
            # disc_scheduler = torch.optim.\
            #                 lr_scheduler.MultiStepLR(disc_opt, 
            #                                         milestones=[5000, 10000, 20000, 30000], 
            #                                         gamma=0.5 )
            # return ([gen_opt, disc_opt], [gen_scheduler, disc_scheduler])
            return gen_opt, disc_opt
        
        def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
            out = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # teste proposto pelo gpt
                    out[k] = v.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last) #v.to(device)
                else:
                    out[k] = v  # deixa BoundingBox, crs etc. no CPU
            return out
        
        def filter_nan_batches(self, batch: Dict)-> Dict:
            """Check and Filter NaN values from the HR samples

            Args:
                batch (Dict): Batch og samples

            Returns:
                Dict: Filtered batch of sample.
            """
            if utils2.check_hr_on_dict(batch):
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
                    

        def batch2img(self, batch: Dict, epoch: int, idx:int )-> list[torch.Tensor]:
            batch = self.filter_nan_batches(batch)
            if utils2.check_hr_on_dict(batch):
                self.input_normalizer(batch, 'lr')
                self.ram_normalizer(batch, 'hr')
                lr_img, hr_img = batch['lr'], batch['hr']
                # lr_img = lr_img.reshape(-1,int(args.crop_size),int(args.crop_size))
                if not args.unet_local_weigths:
                    lr_img = lr_img.repeat(1,3,1,1)
                    hr_img = hr_img.repeat(1,3,1,1)
                assert not torch.isnan(lr_img).any(), f"valor NaN nas imagens lr epoca {epoch}_{idx}"
                assert not torch.isnan(hr_img).any(), f"valor NaN nas imagens hr epoca {epoch}_{idx}" 
                return lr_img, hr_img, batch['bounds']
            else: 
                self.input_normalizer(batch, 'lr')
                lr_img = batch['lr']
                # lr_img = lr_img.reshape(-1,int(args.crop_size),int(args.crop_size))
                if not args.unet_local_weigths:
                    lr_img = lr_img.repeat(1,3,1,1)
                assert not torch.isnan(lr_img).any(), f"valor NaN nas imagens lr epoca {epoch}_{idx}"
                return lr_img, batch['bounds']
                
        
        def save_bounds(self, bounds: list, phase: str, epoch:int, batch_idx:int):
            for idx in range(len(bounds)):
                self._bounds['epoch'].append(epoch)
                self._bounds['batch'].append(batch_idx)
                self._bounds['bounds'].append(bounds[idx])
                self._bounds['disc'].append(phase)                
            
            
        def training_step(self, batch, batch_idx):
        # training_step defines the train loop.    
            lr_img, hr_img, bounds = self.batch2img(batch, self.current_epoch, batch_idx)
            # torch.compiler.cudagraph_mark_step_begin()
            sr_img = self.generator(lr_img)
            batch_size = lr_img.shape[0]   

            # SR images
            gen_opt, disc_opt = self.optimizers()
            # gen_scheduler, disc_scheduler = self.lr_schedulers()
            
            
            # Discriminator optimization
            # torch.compiler.cudagraph_mark_step_begin()
            d_real = self.discriminator(hr_img)
            d_fake = self.discriminator(sr_img.detach()) #important to separate from gen optimazation
            
            disc_loss = self.disc_rela_loss(d_fake,d_real)
            disc_opt.zero_grad(set_to_none=True)
            self.manual_backward(disc_loss)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0) #prevent gradient explosion
            disc_opt.step()
            # disc_scheduler.step()
            
            # Generator optmization
            # torch.compiler.cudagraph_mark_step_begin()
            d_fake_gen = self.discriminator(sr_img)
            loss_adv = self.gen_rela_loss(d_fake_gen, d_real.detach()) #d_real was used on the discriminator opt
            # with torch.no_grad:
            #     loss_percep = self.perceptual_loss(sr_img, hr_img)
            loss_percep = self.perceptual_loss(self.resize_transform(sr_img),self.resize_transform(hr_img))
            loss_content = self.content_loss(hr_img, sr_img)
            gen_loss = loss_percep + 5e-3*loss_adv + 1e-2*loss_content
            
            gen_opt.zero_grad(set_to_none=True)
            self.manual_backward(gen_loss)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0) #prevent gradient explosion
            gen_opt.step()
            # gen_scheduler.step()
            
            self.save_bounds(bounds, 'train', self.current_epoch, batch_idx)

            
            utils2.log_metrics(self, {
                        "t_g_loss": gen_loss,
                        "t_d_loss": disc_loss,
                        "t_pixel_loss": loss_content,
                        "t_percep_loss": loss_percep,
                        "t_adv_loss": loss_adv,
                        'batch_size': batch_size
                    }, on_step=False, on_epoch=True, prog_bar=True)
        
        def validation_step(self, batch, batch_idx):
        # validation_step defines validation loop
            gen_loss = disc_loss = 0
            lr_img, hr_img, bounds = self.batch2img(batch, self.current_epoch, batch_idx)
            batch_size = lr_img.shape[0]
            # assert batch_size == 2*args.batch_size, f'{batch_size} e {args.batch_size} teste de batch_size'
            # torch.compiler.cudagraph_mark_step_begin()
            sr_img = self.generator(lr_img)

            # revert the normalizer in the SR imgs
            # batch['sr'] = sr_img
            # self.input_normalizer.revert(batch, 'sr')
            # sr_img = batch['sr']
            
            # torch.compiler.cudagraph_mark_step_begin()
            d_real = self.discriminator(hr_img)
            d_fake = self.discriminator(sr_img)
            disc_loss = self.disc_rela_loss(d_fake,d_real)
            loss_adv = self.gen_rela_loss(d_fake, d_real) #d_real was used on the discriminator opt
            # with torch.no_grad:
            #     loss_percep = self.perceptual_loss(sr_img, hr_img)
            loss_percep = self.perceptual_loss(self.resize_transform(sr_img),self.resize_transform(hr_img))
            loss_content = self.content_loss(hr_img, sr_img)
            gen_loss = loss_percep + 5e-3*loss_adv + 1e-2*loss_content
            
            # desnormalizar os valores para os cálculos das métricas
            batch['sr'] = sr_img
            ram_norm.revert(batch, 'sr')
            batch['hr'] = hr_img
            ram_norm.revert(batch, 'hr')
            batch['lr'] = lr_img
            input_norm.revert(batch, 'lr')       
            
            psnr_batch, ssim_batch, mse_batch = utils2.calculate_metrics_dem_batch(batch['hr'], batch['sr'], batch_size)
            if batch_idx%15==0:
                utils2.plot_random_dem_crops_lr_sr_hr(lr_img, sr_img, hr_img,
                                                  self.current_epoch, batch_idx,
                                                  'SRIU/artifacts/val/')
                # batch['sr'] = sr_img
                # batch['hr'] = hr_img
                # batch['lr'] = lr_img
                torch.save(batch, 
                            f'SRIU/artifacts/dicts/val_e_{self.current_epoch}_{batch_idx}')
                
            self.save_bounds(bounds, 'val', self.current_epoch, batch_idx)
            
            utils2.log_metrics(self, {
                                "v_g_loss": gen_loss,
                                "v_d_loss": disc_loss,
                                "psnr": psnr_batch,
                                "ssim": ssim_batch,
                                "mse": mse_batch,
                                'batch_size': batch_size
                                }, on_step=False, on_epoch=True, prog_bar=True)    
            
        def test_step(self, batch, batch_idx):
            lr_img, hr_img, bounds = self.batch2img(batch, self.current_epoch, batch_idx)
            batch_size = lr_img.shape[0]
            
            # sr_img = utils2.stich_by_average(lr_img, self.scale,args.crop_size,
            #                                  args.crop_size//2, self.generator)
            
            sr_img = utils2.stich_by_clip(lr_img, self.scale, 2,
                                          self.generator, 40)
            
            
            assert not torch.isnan(sr_img).any(), f"valor NaN gerado pelo generator epoca {self.current_epoch}_{batch_idx}"
            # desnormalizar os valores para os cálculos das métricas
            batch['sr'] = sr_img
            ram_norm.revert(batch, 'sr')
            batch['hr'] = hr_img
            ram_norm.revert(batch, 'hr')
            batch['lr'] = lr_img
            input_norm.revert(batch, 'lr')       
            psnr_batch, ssim_batch, mse_batch = utils2.calculate_metrics_dem_batch(batch['hr'], batch['sr'], batch_size)
            if batch_idx%5==0:
                utils2.plot_dem_predictions_2(batch, batch_size,
                                              batch_idx, 'SRIU/artifacts/val/')
            self.save_bounds(bounds, 'test', self.current_epoch, batch_idx)
            utils2.log_metrics(self, {
                                "test_psnr": psnr_batch,
                                "test_ssim": ssim_batch,
                                "test_mse": mse_batch
                                }, on_step=False, on_epoch=True, prog_bar=True)

        # def predict_step(self, batch):
        #     lr_img = batch[0]  # [B, C, H, W]
        #     name = batch[1]
        #     b, c, h, w = lr_img.shape
        #     scale = self.scale
        #     patch_size = args.crop_size // 2
        #     stride = patch_size // 2  # 50% sobreposição

        #     # Padding
        #     pad_h = (stride - h % stride) % stride
        #     pad_w = (stride - w % stride) % stride
        #     lr_img_padded = F.pad(lr_img, (0, pad_w, 0, pad_h), mode='reflect')
        #     _, _, h_pad, w_pad = lr_img_padded.shape

        #     sr_img_accum = torch.zeros((b, c, h_pad * scale, w_pad * scale), device=lr_img.device)
        #     weight_mask = torch.zeros_like(sr_img_accum)

        #     for i in range(0, h_pad - patch_size + 1, stride):
        #         for j in range(0, w_pad - patch_size + 1, stride):
        #             patch = lr_img_padded[:, :, i:i+patch_size, j:j+patch_size]
        #             with torch.no_grad():
        #                 sr_patch = self.generator(patch)  # [B, C, H', W']
                    
        #             i_sr, j_sr = i * scale, j * scale
        #             sr_img_accum[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += sr_patch
        #             weight_mask[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += 1

        #     sr_img = sr_img_accum / weight_mask
        #     sr_img = sr_img[:, :, :h * scale, :w * scale]  # remove padding
        #     return lr_img, sr_img, name
    
    input_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/{args.input_ds}_finetuning_al.tif"
    input_ds = InMemoryGeoRaster(input_ds_path)
    predict_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/{args.input_ds}_predict_al.tif"
    predict_ds_raster = InMemoryGeoRaster(predict_ds_path)
    
    ram = InMemoryGeoRaster(r"C:\Users\Eduardo JR\Fast\SRIU\ram_finetuning_15m.tif")
    
    
    #importante criar os normalizaers do dem_ds e do R
    input_norm = inicialize_normalizer(input_ds,'image')
    ram_norm = inicialize_normalizer(ram, 'image')
    
    ########## EDITAR AQUI PARA ABRIR AS CAMADAS GPKG JÁ CRIADAS E DIVIDIDAS. ########
    # aoi_finetuning_50k = VectorDataset_GDF(
    #             path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
    #             layer='esrgan_finetuning_50k', 
    #             crs="EPSG:4326")
    # train_aoi, val_aoi, test_aoi = random_bbox_assignment(aoi_finetuning_50k, [0.6, 0.2, 0.2])

    train_aoi = VectorDataset_GDF(
                path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
                layer='esrgan_finetuning_50k_train', 
                crs="EPSG:4326")
    
    val_aoi = VectorDataset_GDF(
                path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
                layer='esrgan_finetuning_50k_val', 
                crs="EPSG:4326")
    
    test_aoi = VectorDataset_GDF(
                path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
                layer='esrgan_finetuning_50k_test', 
                crs="EPSG:4326")
    
    predict_aoi = VectorDataset_GDF(
                path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
                layer='esrgan_finetunig_predict', 
                crs="EPSG:4326")
    
    train_ds =  CustomDEMDataset(input_ds & train_aoi,ram)
    val_ds = CustomDEMDataset(input_ds & val_aoi, ram)
    test_ds = CustomDEMDataset(input_ds & test_aoi, ram)
    predict_ds = CustomDEMDataset(predict_ds_raster, None) #no need to filter 
    
    train_sampler = RandomGeoSampler(
                    train_ds,
                    size=args.crop_size,
                    # length=args.batch_size,
                    roi=train_ds.bounds)
     
    train_dataload = DataLoader(dataset=train_ds,
                                batch_size=args.batch_size,
                                sampler=train_sampler,
                                pin_memory=True,
                                # persistent_workers=True,
                                # num_workers= 4,
                                collate_fn= custom_collate_fn)
    
    val_sampler = RandomGeoSampler(
                val_ds,
                size=args.crop_size,
                # stride=args.crop_size/2,
                # length=args.batch_size,
                roi=val_ds.bounds)
    
    val_dataload = DataLoader(dataset=val_ds, 
                              batch_size=args.batch_size, 
                              sampler=val_sampler,
                              pin_memory=True,
                            #   persistent_workers=True,
                            #   num_workers= 4,
                              collate_fn=custom_collate_fn)
        
    test_sampler = RandomGeoSampler(
                test_ds,
                size=800,
                # stride=args.crop_size/2,
                # length=args.batch_size,
                roi=val_ds.bounds)
    
    test_dataload = DataLoader(dataset=test_ds, 
                              batch_size=4, 
                              sampler=test_sampler,
                              pin_memory=True,
                            #   persistent_workers=True,
                            #   num_workers= 4,
                              collate_fn=custom_collate_fn)
    
    # predict_sampler = RandomGeoSampler(dataset=predict_ds,
    #                                  size=,
    #                                  stride=,
    #                                  )
    
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    adv_profiler = L.pytorch.profilers.AdvancedProfiler(
        filename="adv_profiler",  # Não inclua a extensão .txt aqui
        dirpath=r'D:\Documentos\OneDrive\Documentos\Mestrado\Super Resolução Imagem Única\SRIU\artifacts\profiler',  # Especifica explicitamente o diretório
    )
    
    early_stop_callback = EarlyStopping(
                        monitor="v_g_loss",
                        patience=10,
                        mode="min",
                        verbose=True
                    )

    checkpoint_callback = ModelCheckpoint(
                        monitor="v_g_loss",
                        save_top_k=1,
                        mode="min",
                        filename="best-finetuning-{args.input_ds}-e{epoch:02d}",
                        dirpath="SRIU/saved_ckpt/",
                        auto_insert_metric_name=False
                    )
    
    trainer = L.Trainer(max_epochs=args.max_epochs, 
                    precision="16-mixed",
                    accelerator="gpu",
                    devices=1,
                    limit_train_batches=.25,
                    limit_val_batches=.25,
                    logger=mlflow_logger,
                    enable_progress_bar=True,
                    profiler='simple',
                    # limit_predict_batches=.5,
                    callbacks=[early_stop_callback, checkpoint_callback],
                    )
    
    
    mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")
    # ESRGAN/teste_lr_ativado
    model_uri = "runs:/46f0c76bfc2d4607b4a6cd23ade30d7c/best_model" 
    trained_esrgan = mlflow.pytorch.load_model(model_uri)
    dem_finetuning = DEM_finetuning(trained_esrgan.generator, 
                                    trained_esrgan.discriminator, 
                                    input_norm, ram_norm)
    
    if args.unet_local_weigths:
        new_conv_gen_input = utils2.single_band_model(dem_finetuning.generator.encoder.blocks[0], True)
        dem_finetuning.generator.encoder.blocks[0] = new_conv_gen_input
        new_conv_gen_output = utils2.single_band_model(dem_finetuning.generator.decoder.blocks[1][2], True, False)
        dem_finetuning.generator.decoder.blocks[1][2] = new_conv_gen_output
        new_conv_disc = utils2.single_band_model(dem_finetuning.discriminator.encoder.net[0][0])
        dem_finetuning.discriminator.encoder.net[0][0] = new_conv_disc
    
    #ESRGAN_DEM_FineTuning/SRTM_ESRGAN_DEM_finetuning
    # gen_encoder = models.ESREncoder(in_channels=3, out_channels=64, growth_channels=32,
    #                      num_basic_blocks=23, num_dense_blocks=3, num_residual_blocks=5,
    #                      conv=modules.Conv2d, activation=modules.LeakyReLU, residual_scaling=0.2)
    # gen_decoder = models.ESRNetDecoder(in_channels=64, out_channels=3, scale_factor=2,
    #                             conv=modules.Conv2d, activation=modules.LeakyReLU)
    # generator = models.EncoderDecoderNet(gen_encoder, gen_decoder)

    # disc_encoder = modules.StridedConvEncoder(layers=(3,64,128,128,256,256,512,512),
    #                                 layer_order=("conv","activation"),
    #                                 conv=modules.Conv2d, activation=modules.LeakyReLU)
    # disc_pool    = torch.nn.AdaptiveAvgPool2d(1)
    # disc_head    = modules.LinearHead(in_channels=512, out_channels=1, latent_channels=[1024],
    #                         layer_order=("linear","activation"))
    # discriminator = models.DiscriminatorVGG(disc_encoder, disc_pool, disc_head)
    # dem_finetuning = DEM_finetuning(generator, discriminator, input_norm, ram_norm)
    # state_dict_uri = "runs:/46c7a868fb8749f0b8ee56b9bfcdcdb6/state_dict/dem_finetuning_full.pth"
    # local_path = mlflow.artifacts.download_artifacts(state_dict_uri)
    # state_dict = torch.load(local_path, map_location="cuda")
    # gen_sd, disc_sd = {}, {}
    # for k, v in state_dict.items():
    #     if "generator" in k:
    #         # retira prefixo generator. ou _orig_mod.generator.
    #         new_k = k.split("generator._orig_mod.", 1)[-1]
    #         gen_sd[new_k] = v
    #     elif "discriminator" in k:
    #         new_k = k.split("discriminator._orig_mod.", 1)[-1]
    #         disc_sd[new_k] = v
    # dem_finetuning.generator.load_state_dict(gen_sd, strict=True)
    # dem_finetuning.discriminator.load_state_dict(disc_sd, strict=True)    
    

    trainer.fit(model=dem_finetuning, 
                train_dataloaders=train_dataload, 
                val_dataloaders=val_dataload)
    
    trainer.test(model= dem_finetuning,
                 dataloaders=test_dataload,
                 ckpt_path='best'
                 )
    
    # trainer.predict()

    # utils2.ordena_filtra_profiler('SRIU/artifacts/profiler/fit-adv_profiler.txt')
    # utils2.ordena_filtra_profiler('SRIU/artifacts/profiler/test-adv_profiler.txt')
    
    create_save_gpkg_layer(dem_finetuning._bounds,
                           'C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg',
                           f'ROI_{args.run_name}')
    
    
        
    torch.save(dem_finetuning.state_dict(),
                "SRIU/artifacts/dem_finetuning_full.pth")
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(vars(args))            
        mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
        mlflow.log_artifacts("SRIU/artifacts/pred", artifact_path="prediction_imgs")
        mlflow.log_artifacts("SRIU/artifacts/profiler", artifact_path="profiler_report")
        mlflow.log_artifacts('SRIU/artifacts/dicts', artifact_path='tensors')
        mlflow.log_artifact("SRIU/artifacts/dem_finetuning_full.pth", artifact_path='state_dict')

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=' Train a ESRGAN PSNR oriented model')
    parser.add_argument(
        '--crop_size',
        type= int,
        help= 'Crop size of each hr image in the batch',
        default= 128
    )
    parser.add_argument(
        '--batch_size',
        type= int,
        help='Number of item from the dataloader',
        default= 11 #reduzir para 4 ou 6 em testes
    )
    parser.add_argument(
        '--max_epochs',
        type= int,
        help='Maximum number os epochs in training',
        default=50
    )
    parser.add_argument(
        '--run_name',
        type= str,
        help='name for the mlflow run',
        default= 'COP_ESRGAN_DEM_finetuning'
    )
    parser.add_argument(
        '--input_ds',
        type= str,
        help='SRTM ou COP',
        default= 'COP'
    )
    parser.add_argument(
        '--unet_local_weigths',
        type= str,
        help='weights of a 1band unet',
        default= r'SRIU\saved_ckpt\unet_1band_30epochs.ckpt'
    )
    
    args = parser.parse_args()

    # erase files before a new experiments
    pasta_imagens = 'SRIU/artifacts'
    for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(pasta_raiz, arquivo)
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)
    ESRGAN_DEM_FineTuning(args)