import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # ou ":4096:8" se houver erro de memória
os.environ["PYTHONHASHSEED"] = "42"
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import models, modules, losses, utils2
import argparse
# matplotlib.use('Agg')
import mlflow
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scripts.terrain_features import Slope, Aspect, CurvatureTotal
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from shapely.geometry import box
from L_modules import DEM_ESRGAN, DEM_MSRResNet, MSRResNet, DEM_ESRGAN_slope
from L_callbacks import *
from torchgeo_modules import InMemoryGeoRaster, CustomDEMDataset, VectorDataset_GDF, StrictGridGeoSampler
from modules import custom_collate_fn
from models import SRResNetDecoder, SRResNetEncoder, EncoderDecoderNet, DiscriminatorVGG


# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)
L.seed_everything(42, workers=True)

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


def srgan_generator_setup(in_channels:int, scale:int):
    gen_encoder = SRResNetEncoder(in_channels=in_channels, out_channels=64,
                             num_residual_blocks=20) #D-SRGAN basic blocks
    gen_decoder =SRResNetDecoder(in_channels=64, out_channels=in_channels, scale_factor=scale)

    return EncoderDecoderNet(encoder=gen_encoder, decoder=gen_decoder)

def srgan_discriminator_setup(in_channels:int):
    disc_encoder = modules.StridedConvEncoder(
        layers=(in_channels, 64, 128, 128, 256, 256, 512, 512, 512, 512),
        strides=[1,2,1,2,1,2,1,2,2]
        )
    disc_pool = torch.nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d(kernel_size=X)
    disc_head = modules.LinearHead( in_channels=512, out_channels=1,
                latent_channels=[1024],layer_order=("linear", "activation"))

    return DiscriminatorVGG(encoder=disc_encoder, pool= disc_pool, head=disc_head)

def D_SRGAN(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="D-SRGAN",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')

    
     # creating the generator
    generator = srgan_generator_setup(in_channels=1, scale=args.zoom)
    
    # creating de discriminator
    discriminator = srgan_discriminator_setup(1)
    
    # losses
    if args.unet_local_weigths:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}, #wang et al 21, usa múltiplas camadas aqui.
                        weights_path=args.unet_local_weigths,
                    mean = [0.4517], std=[0.2434]) # imagenet-L mean = 0.4517, std=0.2434
    else:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0})

    
    input_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/{args.input_ds}_finetuning_al.tif"
    input_ds = InMemoryGeoRaster(input_ds_path)
    predict_ds_path = f"C:/Users/Eduardo JR/Fast/SRIU/{args.input_ds}_predict_al.tif"
    # predict_ds_raster = InMemoryGeoRaster(predict_ds_path)
    
    ram = InMemoryGeoRaster(r"C:\Users\Eduardo JR\Fast\SRIU\ram_finetuning_15m.tif")
    # ram_predict = InMemoryGeoRaster(r"C:\Users\Eduardo JR\Fast\SRIU\ram_predict_15m.tif")
    
    # importante criar os normalizaers do dem_ds e do R
    input_norm = inicialize_normalizer(input_ds,'min_max')
    ram_norm = inicialize_normalizer(ram, 'min_max')
    
    
    
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
    
    # predict_aoi = VectorDataset_GDF(
    #             path='C:\\Users\\Eduardo JR\\Fast\\SRIU\\bd_srmde.gpkg', 
    #             layer='esrgan_finetunig_predict', 
    #             crs="EPSG:4326")
    
    train_ds =  CustomDEMDataset(input_ds & train_aoi,ram)
    val_ds = CustomDEMDataset(input_ds & val_aoi, ram)
    test_ds = CustomDEMDataset(input_ds & test_aoi, ram)
    # predict_ds = CustomDEMDataset(predict_ds_raster, ram_predict) #no need to filter 
    
    train_sampler = RandomGeoSampler(
                    train_ds,
                    size=args.crop_size,
                    roi=train_ds.bounds)
     
    train_dataload = DataLoader(dataset=train_ds,
                                batch_size=args.batch_size,
                                sampler=train_sampler,
                                # generator= seed,
                                pin_memory=False,
                                num_workers=0,
                                collate_fn= custom_collate_fn)
    
    val_sampler = StrictGridGeoSampler(
                dataset=val_ds,
                size=128,
                stride=128,
                roi=val_ds.bounds,
                cover= val_aoi) 
    # RandomGeoSampler(
    #             val_ds,
    #             size=args.crop_size,
    #             roi=val_ds.bounds)
        

    
    val_dataload = DataLoader(dataset=val_ds, 
                              batch_size=args.batch_size, 
                              sampler=val_sampler,
                            #   generator= seed,
                              pin_memory=False,
                              num_workers=0,   
                              collate_fn=custom_collate_fn)
        
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
    
    # predict_sampler = GridGeoSampler(
    #             dataset=predict_ds,
    #             size=1000,
    #             stride=1000,
    #             roi=predict_ds.bounds)
    
    # predict_dataload = DataLoader(dataset=predict_ds, 
    #                           batch_size=1, 
    #                           sampler=predict_sampler,
    #                           generator= seed,
    #                           pin_memory=False,
    #                         #   persistent_workers=True,
    #                           num_workers= 0,
    #                           collate_fn=custom_collate_fn)
    
    # save_bounds = Save_bbox(gpkg_path= r"C:\Users\Eduardo JR\Fast\SRIU\\bd_srmde.gpkg",
    #                         layer_name=args.run_name)
    
    early_stop_callback = EarlyStopping(
                        monitor="v_g_loss",
                        patience=10,
                        mode="min",
                        verbose=True
                    )
    compute_metrics = Compute_metrics(ram_norm=ram_norm, input_norm=input_norm)
    
    plot_results = Plot_results(val_batch_interval=40, test_batch_interval=5, 
                                ram_norm=ram_norm, input_norm=input_norm, cmap='terrain')
    
    checkpoint_callback = ModelCheckpoint(
                        monitor="v_g_loss",
                        save_top_k=1,
                        mode="min",
                        filename= 'D-SRGAN_COP_e_{epoch:02d}', #"DEM_GEN_train_e_{epoch:02d}",
                        dirpath="SRIU/saved_ckpt/",
                        auto_insert_metric_name=False
                    )
    
    callbacks_2 = [checkpoint_callback, early_stop_callback, compute_metrics, plot_results]
    
    trainer_2 = L.Trainer(max_epochs= args.max_epochs, 
                precision="16-mixed", # "32-true" ,
                accelerator="gpu",
                deterministic= True,
                limit_train_batches=0.20, #.20,
                # limit_val_batches=.50,
                # limit_test_batches=10,
                # fast_dev_run=20,
                logger=mlflow_logger,
                enable_progress_bar=True,
                # profiler='simple',
                # limit_predict_batches=.5,
                callbacks=callbacks_2,
                )
    
    
    ### PSNR_ESRGAN/teste_gen_gray_Norm
    # gen_ckp_path = r'SRIU\saved_ckpt\MSRResNet_COP_e_08.ckpt' 
    # pre_train_gen = DEM_MSRResNet.load_from_checkpoint(generator=generator, 
    #                                             checkpoint_path=gen_ckp_path, 
    #                         input_normalizer=input_norm, ram_normalizer=ram_norm)
    
     ### Necessário quando se usa o compilador
    torch._dynamo.config.suppress_errors = True 
    generator = utils2.compile_module(generator)
    # discriminator = utils2.compile_module(discriminator)
    # perceptual_loss = utils2.compile_module(perceptual_loss)
    
    ### Testar apenas o predict
    ## 23RRDB
    # model_uri = "mlflow-artifacts:/908603184323111774/6dffeef130074a928cf4d64c0f303553/artifacts/model/MSRResNet_COP_e_20.ckpt"
    ## 10 RRDB
    # model_uri = "mlflow-artifacts:/908603184323111774/c5ad478b03dd47d39e60a8b1e10b750a/artifacts/model/MSRResNet_COP_e_18-v1.ckpt" 
    # mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")
    # local_ckpt = mlflow.artifacts.download_artifacts(model_uri)
 
    pre_train_gen = DEM_MSRResNet.load_from_checkpoint(checkpoint_path= local_ckpt, map_location='cpu', 
                            generator = generator, input_normalizer=input_norm, ram_normalizer=ram_norm)
    
    

    dem_esrgan = DEM_ESRGAN(generator=pre_train_gen.generator, discriminator=discriminator,
                            perceptual_loss=perceptual_loss, rgb=False, lr=1e-4, step_milestones=[1e3, 3e3],
                            input_normalizer=input_norm, ram_normalizer=ram_norm,
                            mean = 0.0854 , std=0.0355)
    
    # dem_esrgan = DEM_ESRGAN_slope(generator=pre_train_gen.generator, discriminator=discriminator,
    #                         perceptual_loss=perceptual_loss, rgb=False, lr=1e-4, step_milestones=[1e3, 3e3],
    #                         input_normalizer=input_norm, ram_normalizer=ram_norm,
    #                         mean = 0.0854 , std=0.0355, slope_fn=Slope(), alfa=args.alfa_slope)
    
    trainer_2.fit(dem_esrgan, train_dataloaders=train_dataload, 
                  val_dataloaders=val_dataload)
    trainer_2.test(dem_esrgan, dataloaders=test_dataload, ckpt_path='best')
        
        

    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(vars(args))            
        mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
        mlflow.log_artifacts("SRIU/artifacts/test", artifact_path="test_imgs")
        mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path='model')

    
    

if __name__ == '__main__':
    # alfas =  [5e-2, 1e-2, 1e-1]#[1e-4 , 5e-3, 1e-3, 5e-2, 1e-2, 1e-1, 1]
    # for alfa_slope in alfas:
        # alfa_slope = 1
        
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
            default=2
        )
        parser.add_argument(
            '--run_name',
            type= str,
            help='name for the mlflow run',
            default= 'ESRGAN_DEM_COP'
        )
        parser.add_argument('--input_ds',
            type= str,
            help='SRTM ou COP',
            default= 'COP'
        )
        parser.add_argument('--unet_local_weigths',
            type= str,
            help='weights of a 1band unet',
            default= r'SRIU\saved_ckpt\unet_1band_72.ckpt'
        )
    
    
        args = parser.parse_args()

        # erase files before a new experiments
        pasta_imagens = 'SRIU/artifacts'
        for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
            for arquivo in arquivos:
                caminho_arquivo = os.path.join(pasta_raiz, arquivo)
                if os.path.isfile(caminho_arquivo):
                    os.remove(caminho_arquivo)
        D_SRGAN(args)