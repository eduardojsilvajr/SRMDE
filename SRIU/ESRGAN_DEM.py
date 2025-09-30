import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # ou ":4096:8" se houver erro de memória
os.environ["PYTHONHASHSEED"] = "42"
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import modules, losses, utils2
import argparse
# matplotlib.use('Agg')
import mlflow
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scripts.terrain_features import Slope, Aspect, CurvatureTotal
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomGeoSampler, Units
from torchgeo.datasets.utils import BoundingBox
import glob
from L_modules import DEM_ESRGAN, DEM_MSRResNet, MSRResNet, DEM_ESRGAN_slope, DEM_MSRResNet_terrain
from L_callbacks import *
from torchgeo_modules import DEMDataset, CustomDEMDataset, VectorDataset_GDF, StrictGridGeoSampler
from modules import custom_collate_fn
from models import ESREncoder, ESRNetDecoder, EncoderDecoderNet, DiscriminatorVGG
from typing import List


# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)
L.seed_everything(42, workers=True)

def inicialize_normalizer_by_tiles(
    ds,
    mode: str = "min_max",             # 'min_max' ou 'mean_std'
    roi: BoundingBox | None = None,    # por padrão usa ds.bounds
    nodata: float | None = None,       # valor NoData (ex.: -32768); None => só filtra não-finito
    max_tiles: int | None = None,      # limite opcional de tiles para acelerar
    clip_outliers: float | None = None # ex.: 5.0 => clip a ±5σ (apenas p/ mean_std)
):
    """
    Calcula estatísticas varrendo *cada tile* retornado pelo índice espacial do RasterDataset.
    Evita GridGeoSampler e qualquer canvas grande.

    Retorna:
      - utils2.Normalizer_minmax(...) se mode='min_max'
      - utils2.Normalizer(...)       se mode='mean_std'
    """

    roi = roi or ds.bounds
    idx = ds.index
    dim = getattr(idx.properties, "dimension", 2)

    # Monta tupla de consulta compatível com a dimensão do índice (2D ou 3D)
    if dim == 3:
        query = (roi.minx, roi.miny, roi.maxx, roi.maxy, roi.mint, roi.maxt)
    else:
        query = (roi.minx, roi.miny, roi.maxx, roi.maxy)

    # Coleta itens (tiles) que intersectam o ROI
    items = list(idx.intersection(query, objects=True))
    if len(items) == 0:
        raise RuntimeError("Nenhum tile encontrado no ROI ao consultar o índice espacial.")

    # Acumuladores globais
    has_data = False
    g_min = torch.tensor(float("inf"))
    g_max = torch.tensor(float("-inf"))

    tot_count = 0
    tot_sum = 0.0
    tot_sumsq = 0.0

    # Itera tile a tile
    for i, it in enumerate(items):
        if max_tiles is not None and i >= max_tiles:
            break

        # BBox do tile (respeita dimensão do índice)
        bb = it.bounds  # rtree IndexItem: tuple com 4 ou 6 números
        if dim == 3:
            minx, maxx, miny, maxy, mint, maxt = bb
        else:
            minx, maxx, miny, maxy = bb
            mint, maxt = ds.bbox.mint, ds.bbox.maxt

        tile_bb = BoundingBox(minx, maxx, miny, maxy, 0, 0)

        # Interseção com o ROI (segurança)
        ix_minx = max(tile_bb.minx, roi.minx)
        ix_maxx = min(tile_bb.maxx, roi.maxx)
        ix_miny = max(tile_bb.miny, roi.miny)
        ix_maxy = min(tile_bb.maxy, roi.maxy)
        if not (ix_minx < ix_maxx and ix_miny < ix_maxy):
            continue  # sem sobreposição espacial

        inter = BoundingBox(ix_minx, ix_maxx, ix_miny, ix_maxy, mint, maxt)

        # Lê apenas a área de interseção via RasterDataset (abrirá o GeoTIFF certo)
        out = ds[inter]               # {'image': Tensor [1,H,W], ...}
        x = out["image"].to(torch.float32)
        x[x<0] = float('nan')

        # Máscara de inválidos
        if nodata is not None:
            invalid = torch.isclose(x, torch.tensor(float(nodata), dtype=x.dtype))
        else:
            invalid = ~torch.isfinite(x)

        if invalid.all():
            continue

        # Atualiza min/máx ignorando inválidos
        xv = x.masked_fill(invalid, float('nan'))
        
        flat = x.view(1,-1)
        mask = torch.isnan(flat)

        filled_min = flat.clone()
        filled_min[mask] = float('inf')
        bmin,_ = filled_min.min(dim=1)

        filled_max = flat.clone()
        filled_max[mask] = float('-inf')
        bmax, _ = filled_max.max(dim=1)
        
        if torch.isfinite(bmin):
            g_min = torch.minimum(g_min, bmin)
            has_data = True
        if torch.isfinite(bmax):
            g_max = torch.maximum(g_max, bmax)
            has_data = True

        # Média/variância incremental (opcional)
        if mode == "mean_std":
            xv = x[~invalid]
            if xv.numel() == 0:
                continue
            if clip_outliers is not None and clip_outliers > 0:
                mu = float(xv.mean())
                sd = float(xv.std(unbiased=False)) + 1e-12
                lo, hi = mu - clip_outliers * sd, mu + clip_outliers * sd
                xv = xv.clamp(min=lo, max=hi)
            tot_count += int(xv.numel())
            tot_sum   += float(xv.sum())
            tot_sumsq += float((xv * xv).sum())

    if not has_data:
        raise RuntimeError("Apenas NoData/NaN encontrados ao percorrer os tiles do ROI.")

    # Constrói o normalizador
    if mode == "min_max":
        g_min = g_min
        g_max = g_max
        return utils2.Normalizer_minmax(min=g_min, max=g_max)

    if mode == "mean_std":
        if tot_count == 0:
            raise RuntimeError("Sem pixels válidos para média/desvio.")
        mean = tot_sum / tot_count
        var  = max(tot_sumsq / tot_count - mean * mean, 0.0)
        std  = math.sqrt(var) + 1e-12
        return utils2.Normalizer(mean=[mean], stdev=[std]) #type: ignore

    raise ValueError("mode deve ser 'min_max' ou 'mean_std'")

def inicialize_normalizer(ds: RasterDataset, label: str, ):
    ds_dict = ds.__getitem__(ds.bounds)
        # filtering NaN pixels values.
    tensor = ds_dict['image']
    tensor[tensor < 0] = float('nan') 

    if tensor.isnan().any():
        mean = torch.nanmean(tensor)
        std = utils2.nanstd(tensor)
        
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
        mean = torch.mean(tensor)
        std = torch.std(tensor)
    if label =='min_max':    
        return utils2.Normalizer_minmax(min_global, max_global)
    elif label =='mean_std':
        return utils2.Normalizer(mean=mean, stdev=std)

def lista_tifs(diretorio: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(diretorio, pattern)))


def esrgan_generator_setup(in_channels:int, basic_blocks:int, scale:int):
    gen_encoder = ESREncoder(in_channels=in_channels, out_channels=64, growth_channels=32,
                            num_basic_blocks=basic_blocks, num_dense_blocks=3, num_residual_blocks=5, 
                            residual_scaling= 0.2) #num_basic_blocks=23 ou 10
    gen_decoder =ESRNetDecoder(in_channels=64, out_channels=in_channels, scale_factor=scale)

    return EncoderDecoderNet(encoder=gen_encoder, decoder=gen_decoder)

def esrgan_discriminator_setup(in_channels:int):
    disc_encoder = modules.StridedConvEncoder(layers=(in_channels, 64,
                128, 128, 256, 256, 512, 512, 1024),
                strides= None,
               )
    disc_pool = torch.nn.AdaptiveAvgPool2d(1) 
    disc_head = modules.LinearHead( in_channels=1024, out_channels=1,
                latent_channels=[1024],layer_order=("linear", "activation"))

    return DiscriminatorVGG(encoder=disc_encoder, pool= disc_pool, head=disc_head)


def ESRGAN_DEM(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="D-ESRGAN",                 # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')

    
    ### creating the generator
    generator = esrgan_generator_setup(in_channels=1, basic_blocks=args.bblock, scale=args.zoom)
    
    ### creating de discriminator
    discriminator = esrgan_discriminator_setup(1)
    

    
    ### Setting perceptual loss for 1 band images
    if args.unet_local_weigths:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}, #wang et al 21, usa múltiplas camadas aqui.
                        weights_path=args.unet_local_weigths,
                    mean = [0.4517], std=[0.2434]) # imagenet-L mean = 0.4517, std=0.2434
    else:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0})

    ### usado quando a imagem era única.
    
        # input_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/{args.input_ds}_finetuning_al.tif"
        # input_ds = InMemoryGeoRaster(input_ds_path)
        # predict_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/{args.input_ds}_predict_al.tif"
        # # predict_ds_raster = InMemoryGeoRaster(predict_ds_path)
        
        # ram = InMemoryGeoRaster(r"C:\Users\Eduardo JR\Fast\SRIU\ram_finetuning_15m.tif")
        # # ram_predict = InMemoryGeoRaster(r"C:\Users\Eduardo JR\Fast\SRIU\ram_predict_15m.tif")
        
        # # importante criar os normalizaers do dem_ds e do R
        # input_norm = inicialize_normalizer(input_ds,'min_max')
        # ram_norm = inicialize_normalizer(ram, 'min_max')
    
    input_directory = f"C:\\Users\\Eduardo JR\\Fast\\SRIU\\zoom_{args.zoom}"
    input_train = DEMDataset(paths=lista_tifs(input_directory+'/train', 'cop*.tif'))
    input_val = DEMDataset(paths=lista_tifs(input_directory+'/val', 'cop*.tif'))
    input_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'cop*.tif'))
    
    ram_train = DEMDataset(paths=lista_tifs(input_directory+'/train', 'ram_*_dsm.tif'))
    ram_val = DEMDataset(paths=lista_tifs(input_directory+'/val', 'ram_*_dsm.tif'))
    ram_test = DEMDataset(paths=lista_tifs(input_directory+'/test', 'ram_*_dsm.tif'))    
    
    # input_norm = inicialize_normalizer_by_tiles(input_train|input_val|input_test, mode='min_max', nodata=None)
    # ram_norm = inicialize_normalizer_by_tiles(ram_train|ram_val|ram_test, mode='min_max', nodata=-9999 )
    input_norm = inicialize_normalizer_by_tiles(input_train, mode='min_max', nodata=None)
    ram_norm = inicialize_normalizer_by_tiles(ram_train, mode='min_max', nodata=-9999 )


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
    

    train_ds =  CustomDEMDataset(input_train,ram_train)
    val_ds = CustomDEMDataset(input_val, ram_val)
    test_ds = CustomDEMDataset(input_test, ram_test)
    
        # train_ds =  CustomDEMDataset(input_ds & train_aoi,ram)
        # val_ds = CustomDEMDataset(input_ds & val_aoi, ram)
        # test_ds = CustomDEMDataset(input_ds & test_aoi, ram)
        # predict_ds = CustomDEMDataset(predict_ds_raster, ram_predict) #no need to filter 
    
    train_sampler = RandomGeoSampler(
                    train_ds,
                    size=args.crop_size,
                    roi=train_aoi.bounds)
     
    train_dataload = DataLoader(dataset=train_ds,
                                batch_size=args.batch_size,
                                sampler=train_sampler,
                                generator= seed,
                                pin_memory=False,
                                num_workers=0,
                                collate_fn= custom_collate_fn)
    
    val_sampler = StrictGridGeoSampler(
                dataset=val_ds,
                size=128,
                stride=128,
                roi=val_ds.bounds,
                cover= val_aoi) 


    
    val_dataload = DataLoader(dataset=val_ds, 
                              batch_size=args.batch_size, 
                              sampler=val_sampler,
                              generator= seed,
                              pin_memory=False,
                              num_workers=0,   
                              collate_fn=custom_collate_fn)
        
    test_sampler = StrictGridGeoSampler(
                dataset=test_ds,
                size=910,
                stride=910,
                roi=test_ds.bounds,
                cover= test_aoi)
    
    test_dataload = DataLoader(dataset=test_ds, 
                              batch_size=1, 
                              sampler=test_sampler,
                              generator= seed,
                              pin_memory=False,
                            #   persistent_workers=True,
                              num_workers= 0,
                              collate_fn=custom_collate_fn)
    

    
    save_bounds = Save_bbox(gpkg_path= r"C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg",
                            layer_name=args.run_name)
    
    early_stop_callback = EarlyStopping(
                        monitor="v_g_loss",
                        patience=10,
                        mode="min",
                        verbose=True
                    )
    compute_metrics = Compute_metrics(ram_norm=ram_norm, input_norm=ram_norm, scale=args.zoom)
    
    plot_results = Plot_results(val_batch_interval=40, test_batch_interval=3, 
                                ram_norm=ram_norm, input_norm=ram_norm, cmap='terrain')
    
    checkpoint_callback = ModelCheckpoint(
                        monitor="v_g_loss",
                        save_top_k=1,
                        mode="min",
                        filename= 'DEM_GEN_train_COP_e_{epoch:02d}', #"DEM_GEN_train_e_{epoch:02d}",
                        dirpath="SRIU/saved_ckpt/",
                        auto_insert_metric_name=False
                    )
    
    callbacks_2 = [checkpoint_callback, early_stop_callback, 
                   compute_metrics, plot_results, 
                    # save_bounds
                  ]
    
    trainer_2 = L.Trainer(max_epochs= args.max_epochs, 
                precision="16-mixed", # "32-true" ,
                accelerator="gpu",
                deterministic= True,
                limit_train_batches=0.20,
                # limit_val_batches=.50,
                # fast_dev_run=20,
                logger=mlflow_logger,
                enable_progress_bar=True,
                # profiler='simple',
                # limit_predict_batches=.5,
                callbacks=callbacks_2,
                )
    

    
    ### Necessário quando se usa o compilador
    torch._dynamo.config.suppress_errors = True                         #type: ignore
    generator = utils2.compile_module(generator)
    discriminator = utils2.compile_module(discriminator)
    # perceptual_loss = utils2.compile_module(perceptual_loss)
    
    ### pre load model
    ### scale = 2
    ## 23RRDB
    # model_uri = "mlflow-artifacts:/908603184323111774/6dffeef130074a928cf4d64c0f303553/artifacts/model/MSRResNet_COP_e_20.ckpt"
    ## 10 RRDB
    # model_uri = "mlflow-artifacts:/908603184323111774/c5ad478b03dd47d39e60a8b1e10b750a/artifacts/model/MSRResNet_COP_e_18-v1.ckpt" 
    
    ### scale =4
    ## 23RRDB
    # model_uri = "mlflow-artifacts:/838611645833230282/d1fbf6f9db75401e9620c2ddbd06ff73/artifacts/model/DEM_ESRGAN_COP_e_21.ckpt"
    ## 10 RRDB
    # model_uri = "mlflow-artifacts:/838611645833230282/04305b059a024401a6c81acea56d9fe6/artifacts/model/DEM_ESRGAN_COP_e_09-v1.ckpt"
    

    # mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")           #type: ignore
    # local_ckpt = mlflow.artifacts.download_artifacts(model_uri)         #type: ignore 
    # local_ckpt = r'SRIU\saved_ckpt\DEM_ESRGAN_COP_e_08.ckpt'

    # pre_train_gen = DEM_MSRResNet.load_from_checkpoint(checkpoint_path= local_ckpt, map_location='cpu', 
    #                 generator = generator, input_normalizer=ram_norm, ram_normalizer=ram_norm)
    
    
    sr_mde = DEM_MSRResNet(generator=generator, lr= 5e-4, lr_decay=1e3, rgb=False, 
                               input_normalizer=ram_norm, ram_normalizer=ram_norm,
                               mean = 0.0854 , std=0.0355)
    
    # sr_mde = DEM_MSRResNet_terrain(generator=pre_train_gen.generator, lr= 5e-4, lr_decay=1e3, rgb=False, 
    #                            input_normalizer=ram_norm, ram_normalizer=ram_norm,
    #                            mean = 0.0854 , std=0.0355, terrain_fn=Aspect(), alfa=args.alfa_aspect)
    
    

    # sr_mde = DEM_ESRGAN(generator=pre_train_gen.generator, discriminator=discriminator,
    #                         perceptual_loss=perceptual_loss, rgb=False, lr=1e-4, step_milestones=[1e3, 3e3],
    #                         input_normalizer=ram_norm, ram_normalizer=ram_norm,
    #                         mean = 0.0854 , std=0.0355)
    
    # sr_mde = DEM_ESRGAN_slope(generator=pre_train_gen.generator, discriminator=discriminator,
    #                         perceptual_loss=perceptual_loss, rgb=False, lr=1e-4, step_milestones=[1e3, 3e3],
    #                         input_normalizer=input_norm, ram_normalizer=ram_norm,
    #                         mean = 0.0854 , std=0.0355, slope_fn=Slope(), alfa=args.alfa_slope)
    
    trainer_2.fit(sr_mde, train_dataloaders=train_dataload, 
                  val_dataloaders=val_dataload)
    trainer_2.test(sr_mde, dataloaders=test_dataload, ckpt_path='best'
                   )       
        

    with mlflow.start_run(run_id=mlflow_logger.run_id):    #type: ignore
        mlflow.log_params(vars(args))            
        # mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
        mlflow.log_artifacts("SRIU/artifacts/test", artifact_path="test_imgs")
        mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path='model')
        # mlflow.log_artifact(local_ckpt, artifact_path='model')

    
    

if __name__ == '__main__':
    # alfas =  [1e-4 , 5e-3, 1e-3, 5e-2, 1e-2, 1e-1, 1]
    for blocks in range(1,10):
    #     # alfa_slope = 1
        
        parser = argparse.ArgumentParser(
            description=' Train a ESRGAN PSNR oriented model')
        parser.add_argument('--crop_size',
            type= int,
            help= 'Crop size of each hr image in the batch',
            default= 128
        )
        parser.add_argument('--batch_size',
            type= int,
            help='Number of item from the dataloader',
            default=6 #6 
        )
        parser.add_argument('--max_epochs',
            type= int,
            help='Maximum number os epochs in training',
            default=100
        )
        parser.add_argument('--zoom',
            type= int,
            help='SR zoom',
            default=4
        )
        parser.add_argument('--bblock',
            type= int,
            help='number o RRDB in generator architeture.',
            default=blocks
        )
        # parser.add_argument('--alfa_aspect',
        #     type= int,
        #     help='slope parameter',
        #     default= 0#alfa_slope
        # )
        parser.add_argument(
            '--run_name',
            type= str,
            help='name for the mlflow run',
            default= f'Generator_{blocks}RRDB'  
        ) #'ESRGAN_DEM_COP_10RRDB_slope'  'MSRResNet_DEM_COP_10RRDB'
        parser.add_argument('--input_ds',
            type= str,
            help='SRTM ou COP',
            default= 'COP'
        )
        parser.add_argument('--unet_local_weigths',
            type= str,
            help='weights of a 1band unet',
            default= r'SRIU\saved_ckpt\VGG_1band_72.ckpt'
        )
    
    
        args = parser.parse_args()

        # erase files before a new experiments
        pasta_imagens = 'SRIU/artifacts'
        for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
            for arquivo in arquivos:
                caminho_arquivo = os.path.join(pasta_raiz, arquivo)
                if os.path.isfile(caminho_arquivo):
                    os.remove(caminho_arquivo)
        ESRGAN_DEM(args)