# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # ou ":4096:8" se houver erro de memória
# os.environ["PYTHONHASHSEED"] = "42"
# import torch
# from torch.utils.data import DataLoader
# import lightning as L
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import MLFlowLogger
# import models, modules, losses, utils2
# import argparse
# # matplotlib.use('Agg')
# import mlflow
# import os, sys, glob
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)
# from scripts.terrain_features import Slope, Aspect, CurvatureTotal
# from torchgeo.datasets import RasterDataset
# from torchgeo.samplers import RandomGeoSampler, GridGeoSampler, Units
# from torchgeo.datasets.utils import BoundingBox
# from shapely.geometry import box
# from L_modules import DEM_ESRGAN, DEM_MSRResNet, MSRResNet, DEM_ESRGAN_slope
# from L_callbacks import *
# from torchgeo_modules import DEMDataset, CustomDEMDataset, VectorDataset_GDF, StrictGridGeoSampler
# from modules import custom_collate_fn
# from typing import Any, Dict, Optional, List
# from ESRGAN_DEM import esrgan_generator_setup, esrgan_discriminator_setup




# # garante a reprodutibilidade do experimento
# utils2.set_seed(42)
# seed = torch.Generator().manual_seed(42)
# L.seed_everything(42, workers=True)


# def inicialize_normalizer_by_tiles(
#     ds,
#     mode: str = "min_max",             # 'min_max' ou 'mean_std'
#     *,
#     roi: BoundingBox | None = None,    # por padrão usa ds.bounds
#     nodata: float | None = None,       # valor NoData (ex.: -32768); None => só filtra não-finito
#     max_tiles: int | None = None,      # limite opcional de tiles para acelerar
#     clip_outliers: float | None = None # ex.: 5.0 => clip a ±5σ (apenas p/ mean_std)
# ):
#     """
#     Calcula estatísticas varrendo *cada tile* retornado pelo índice espacial do RasterDataset.
#     Evita GridGeoSampler e qualquer canvas grande.

#     Retorna:
#       - utils2.Normalizer_minmax(...) se mode='min_max'
#       - utils2.Normalizer(...)       se mode='mean_std'
#     """

#     roi = roi or ds.bounds
#     idx = ds.index
#     dim = getattr(idx.properties, "dimension", 2)

#     # Monta tupla de consulta compatível com a dimensão do índice (2D ou 3D)
#     if dim == 3:
#         query = (roi.minx, roi.miny, roi.maxx, roi.maxy, roi.mint, roi.maxt)
#     else:
#         query = (roi.minx, roi.miny, roi.maxx, roi.maxy)

#     # Coleta itens (tiles) que intersectam o ROI
#     items = list(idx.intersection(query, objects=True))
#     if len(items) == 0:
#         raise RuntimeError("Nenhum tile encontrado no ROI ao consultar o índice espacial.")

#     # Acumuladores globais
#     has_data = False
#     g_min = torch.tensor(float("inf"))
#     g_max = torch.tensor(float("-inf"))

#     tot_count = 0
#     tot_sum = 0.0
#     tot_sumsq = 0.0

#     # Itera tile a tile
#     for i, it in enumerate(items):
#         if max_tiles is not None and i >= max_tiles:
#             break

#         # BBox do tile (respeita dimensão do índice)
#         bb = it.bounds  # rtree IndexItem: tuple com 4 ou 6 números
#         if dim == 3:
#             minx, maxx, miny, maxy, mint, maxt = bb
#         else:
#             minx, maxx, miny, maxy = bb
#             mint, maxt = ds.bbox.mint, ds.bbox.maxt

#         tile_bb = BoundingBox(minx, maxx, miny, maxy, 0, 0)

#         # Interseção com o ROI (segurança)
#         ix_minx = max(tile_bb.minx, roi.minx)
#         ix_maxx = min(tile_bb.maxx, roi.maxx)
#         ix_miny = max(tile_bb.miny, roi.miny)
#         ix_maxy = min(tile_bb.maxy, roi.maxy)
#         if not (ix_minx < ix_maxx and ix_miny < ix_maxy):
#             continue  # sem sobreposição espacial

#         inter = BoundingBox(ix_minx, ix_maxx, ix_miny, ix_maxy, mint, maxt)

#         # Lê apenas a área de interseção via RasterDataset (abrirá o GeoTIFF certo)
#         out = ds[inter]               # {'image': Tensor [1,H,W], ...}
#         x = out["image"].to(torch.float32)
#         x[x<0] = float('nan')

#         # Máscara de inválidos
#         if nodata is not None:
#             invalid = torch.isclose(x, torch.tensor(float(nodata), dtype=x.dtype))
#         else:
#             invalid = ~torch.isfinite(x)

#         if invalid.all():
#             continue

#         # Atualiza min/máx ignorando inválidos
#         xv = x.masked_fill(invalid, float('nan'))
        
#         flat = x.view(1,-1)
#         mask = torch.isnan(flat)

#         filled_min = flat.clone()
#         filled_min[mask] = float('inf')
#         bmin,_ = filled_min.min(dim=1)

#         filled_max = flat.clone()
#         filled_max[mask] = float('-inf')
#         bmax, _ = filled_max.max(dim=1)
        
#         if torch.isfinite(bmin):
#             g_min = torch.minimum(g_min, bmin)
#             has_data = True
#         if torch.isfinite(bmax):
#             g_max = torch.maximum(g_max, bmax)
#             has_data = True

#         # Média/variância incremental (opcional)
#         if mode == "mean_std":
#             xv = x[~invalid]
#             if xv.numel() == 0:
#                 continue
#             if clip_outliers is not None and clip_outliers > 0:
#                 mu = float(xv.mean())
#                 sd = float(xv.std(unbiased=False)) + 1e-12
#                 lo, hi = mu - clip_outliers * sd, mu + clip_outliers * sd
#                 xv = xv.clamp(min=lo, max=hi)
#             tot_count += int(xv.numel())
#             tot_sum   += float(xv.sum())
#             tot_sumsq += float((xv * xv).sum())

#     if not has_data:
#         raise RuntimeError("Apenas NoData/NaN encontrados ao percorrer os tiles do ROI.")

#     # Constrói o normalizador
#     if mode == "min_max":
#         g_min = g_min.unsqueeze(0).unsqueeze(0)
#         g_max = g_max.unsqueeze(0).unsqueeze(0)
#         return utils2.Normalizer_minmax(min=g_min, max=g_max)

#     if mode == "mean_std":
#         if tot_count == 0:
#             raise RuntimeError("Sem pixels válidos para média/desvio.")
#         mean = tot_sum / tot_count
#         var  = max(tot_sumsq / tot_count - mean * mean, 0.0)
#         std  = math.sqrt(var) + 1e-12
#         return utils2.Normalizer(mean=[mean], stdev=[std])

#     raise ValueError("mode deve ser 'min_max' ou 'mean_std'")

# def inicialize_normalizer(
#     ds,                      # RasterDataset ou compatível com __getitem__(bbox)->{'image': Tensor}
#     mode: str = 'min_max',   # 'min_max' ou 'mean_std'
#     *,
#     patch_size: int = 4096,  # em pixels (ajuste conforme a sua RAM)
#     stride: int | None = None,
# ):
#     """
#     Calcula min/máx ou média/desvio via varredura em patches,
#     evitando ds.__getitem__(ds.bounds) e o canvas gigante.
#     """
#     stride = stride or patch_size

#     # Sampler em PIXELS pra não depender de resolução CRS

#     sampler = GridGeoSampler(
#             ds, size=patch_size, stride=stride, roi=ds.bounds, units= Units.PIXELS
#     )
    
    

#     # Acumuladores
#     has_data = False
#     g_min = torch.tensor(float("inf"))
#     g_max = torch.tensor(float("-inf"))

#     # Para média/variância (ignorando NaN) usamos somas
#     tot_count = 0
#     tot_sum = 0.0
#     tot_sumsq = 0.0

#     for k, bbox in enumerate(sampler):
#         out = ds[bbox]                # espera {'image': Tensor [1,H,W]}
#         x   = out["image"].to(torch.float32)  # CPU está bom
#         # tratar NoData como NaN (ajuste sua regra se necessário)
#         x[x < 0] = float('nan')

#         if torch.isnan(x).all():
#             continue
        
#         flat = x.view(1,-1)
#         mask = torch.isnan(flat)

#         filled_min = flat.clone()
#         filled_min[mask] = float('inf')
#         bmin,_ = filled_min.min(dim=1)

#         filled_max = flat.clone()
#         filled_max[mask] = float('-inf')
#         bmax, _ = filled_max.max(dim=1)
        
#         # min/max
#         g_min = torch.minimum(g_min, bmin)
#         g_max = torch.maximum(g_max, bmax)
#         has_data = True

#         if mode == 'mean_std':
#             # somas ignorando NaN
#             valid = ~torch.isnan(x)
#             n = int(valid.sum())
#             if n > 0:
#                 xv = x[valid]
#                 tot_count += n
#                 tot_sum   += float(xv.sum())
#                 tot_sumsq += float((xv * xv).sum())

#     if not has_data:
#         raise RuntimeError("Não foi possível calcular estatísticas: apenas NaNs foram encontrados.")

#     if mode == 'min_max':
#         # utils2.Normalizer_minmax espera tensores 1D (C,) — aqui C=1
#         return utils2.Normalizer_minmax(min=[g_min.item()], max=[g_max.item()])

#     elif mode == 'mean_std':
#         if tot_count == 0:
#             raise RuntimeError("Sem pixels válidos para média/desvio.")
#         mean = tot_sum / tot_count
#         # var populacional: E[x^2] - (E[x])^2
#         var  = max(tot_sumsq / tot_count - mean * mean, 0.0)
#         std  = math.sqrt(var) + 1e-12  # estabiliza
#         return utils2.Normalizer(mean=[mean], stdev=[std])

#     else:
#         raise ValueError("mode deve ser 'min_max' ou 'mean_std'")

# def lista_tifs(diretorio: str, pattern: str) -> List[str]:
#     return sorted(glob.glob(os.path.join(diretorio, pattern)))

# torch.set_float32_matmul_precision('high')

# generator = esrgan_generator_setup(in_channels=1, scale=2)
# discriminator = esrgan_discriminator_setup(in_channels=1)
# perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}, #wang et al 21, usa múltiplas camadas aqui.
#                         weights_path= r'SRIU\saved_ckpt\unet_1band_72.ckpt',
#                     mean = [0.4517], std=[0.2434]) # imagenet-L mean = 0.4517, std=0.2434

# # input_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/COP_finetuning_al.tif"
# input_directory = r"C:\Users\Eduardo JR\Fast\SRIU\zoom_2"

# input_train = DEMDataset(paths=lista_tifs(input_directory+'/train', 'cop*.tif'))
# input_val = DEMDataset(paths=lista_tifs(input_directory+'/val', 'cop*.tif'))
# input_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'cop*.tif'))
# # predict_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/COP_predict_al.tif"
# # predict_ds_raster = InMemoryGeoRaster(predict_ds_path)

# ram_train = DEMDataset(paths=lista_tifs(input_directory+'/train', 'ram_*_dsm.tif'))
# ram_val = DEMDataset(paths=lista_tifs(input_directory+'/val', 'ram_*_dsm.tif'))
# ram_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'ram_*_dsm.tif'))
# # ram_predict = InMemoryGeoRaster(r"C:\Users\Eduardo JR\Fast\SRIU\ram_predict_15m.tif")

# # importante criar os normalizaers do dem_ds e do R

# input_norm = inicialize_normalizer_by_tiles(input_train|input_val|input_test, mode='min_max', nodata=None)
# ram_norm = inicialize_normalizer_by_tiles(ram_train|ram_val|ram_test, mode='min_max', nodata=-9999 )



# ########## EDITAR AQUI PARA ABRIR AS CAMADAS GPKG JÁ CRIADAS E DIVIDIDAS. ########
# # aoi_finetuning_50k = VectorDataset_GDF(
# #             path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
# #             layer='esrgan_finetuning_50k', 
# #             crs="EPSG:4326")
# # train_aoi, val_aoi, test_aoi = random_bbox_assignment(aoi_finetuning_50k, [0.6, 0.2, 0.2])

# train_aoi = VectorDataset_GDF(
#             path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
#             layer='esrgan_finetuning_50k_train', 
#             crs="EPSG:4326")

# val_aoi = VectorDataset_GDF(
#                 path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
#                 layer='esrgan_finetuning_50k_val', 
#                 crs="EPSG:4326")


# train_ds =  CustomDEMDataset(input_train,ram_train)
# val_ds = CustomDEMDataset(input_val, ram_val)

# # predict_ds = CustomDEMDataset(predict_ds_raster, ram_predict) #no need to filter 

# train_sampler = RandomGeoSampler(
#                 train_ds,
#                 size=128,
#                 roi=train_ds.bounds)
    
# train_dataload = DataLoader(dataset=train_ds,
#                             batch_size=6,
#                             sampler=train_sampler,
#                             generator= seed,
#                             pin_memory=False,
#                             num_workers=0,
#                             collate_fn= custom_collate_fn)

# val_sampler = StrictGridGeoSampler(
#                 dataset=val_ds,
#                 size=128,
#                 stride=128,
#                 roi=val_ds.bounds,
#                 cover= val_aoi) 

# val_dataload = DataLoader(dataset=val_ds, 
#                               batch_size=12, 
#                               sampler=val_sampler,
#                             #   generator= seed,
#                               pin_memory=False,
#                               num_workers=0,   
#                               collate_fn=custom_collate_fn)

# save_bounds = Save_bbox(gpkg_path= r"C:\Users\Eduardo JR\Fast\SRIU\\bd_srmde.gpkg",
#                             layer_name='mopa_teste')

# trainer_2 = L.Trainer(max_epochs= 10, 
#                 precision="16-mixed", # "32-true" ,
#                 accelerator="gpu",
#                 deterministic= True,
#                 limit_train_batches=0.20, #.20,
#                 # limit_val_batches=.50,
#                 # limit_test_batches=10,
#                 # fast_dev_run=20,
#                 # logger=mlflow_logger,
#                 enable_progress_bar=True,
#                 # profiler='simple',
#                 # limit_predict_batches=.5,
#                 callbacks=save_bounds,
#                 )


# generator = utils2.compile_module(generator)
# discriminator = utils2.compile_module(discriminator)
# perceptual_loss = utils2.compile_module(perceptual_loss)

# model_uri = "mlflow-artifacts:/908603184323111774/c5ad478b03dd47d39e60a8b1e10b750a/artifacts/model/MSRResNet_COP_e_18-v1.ckpt" 
# mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")
# local_ckpt = mlflow.artifacts.download_artifacts(model_uri)

# pre_train_gen = DEM_MSRResNet.load_from_checkpoint(checkpoint_path= local_ckpt, map_location='cpu', 
#                         generator = generator, input_normalizer=input_norm, ram_normalizer=ram_norm)

# dem_esrgan = DEM_ESRGAN(generator=pre_train_gen.generator, discriminator=discriminator,
#                             perceptual_loss=perceptual_loss, rgb=False, lr=1e-4, step_milestones=[1e3, 3e3],
#                             input_normalizer=input_norm, ram_normalizer=ram_norm,
#                             mean = 0.0854 , std=0.0355)

# trainer_2.fit(dem_esrgan, train_dataloaders=train_dataload, 
#                   val_dataloaders=val_dataload)


from ESRGAN_DEM import lista_tifs
from torchgeo_modules import DEMDataset,CustomDEMDataset, VectorDataset_GDF, StrictGridGeoSampler
from modules import custom_collate_fn
from torch.utils.data import DataLoader
from torchgeo.samplers import  Units

input_directory = f"C:\\Users\\Eduardo JR\\Fast\\SRIU\\zoom_4"

input_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'cop*.tif'))
ram_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'ram_*_dsm.tif'))    

test_aoi = VectorDataset_GDF(
                path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
                layer='esrgan_finetuning_50k_test', 
                crs="EPSG:4326",
                res= input_test.res)

test_ds = CustomDEMDataset(input_test, ram_test)

test_sampler = StrictGridGeoSampler(
                dataset=test_ds,
                size=900,
                stride=900,
                roi=test_ds.bounds,
                cover= test_aoi)

test_dataload = DataLoader(dataset=test_ds, 
                              batch_size=1, 
                              sampler=test_sampler,
                            #   generator= seed,
                              pin_memory=False,
                            #   persistent_workers=True,
                              num_workers= 0,
                              collate_fn=custom_collate_fn)

data1 = next(iter(test_dataload))
print(data1)