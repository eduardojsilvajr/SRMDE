import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import models, modules, losses, utils2
import argparse
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'WebAgg' (escolha um backend interativo)
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from torchgeo.datasets import RasterDataset
import tqdm
# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)


######parâmetros
class DEM(RasterDataset):
    is_image = True

class RAM(RasterDataset):
    is_image = False

def inicialize_normalizer(ds: RasterDataset, label: str):
    ds_dict = ds.__getitem__(ds.bounds)
    mean = ds_dict[label].float().mean(dim=(1,2))
    std =ds_dict[label].float().std(dim=(1,2))
    return utils2.Normalizer(mean, std)

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


def ESRGAN_dem_prediction(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="ESRGAN",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')
    
    # creating the generator
    gen_encoder = models.ESREncoder(in_channels=3, out_channels=64, growth_channels=32,
                            num_basic_blocks=23, num_dense_blocks=3, num_residual_blocks=5,
                            conv=modules.Conv2d, activation=modules.LeakyReLU, residual_scaling= 0.2)
    gen_decoder = models.ESRNetDecoder(in_channels=64, out_channels=3, scale_factor=2,
                                conv=modules.Conv2d, activation=modules.LeakyReLU)

    generator = models.EncoderDecoderNet(encoder=gen_encoder, decoder=gen_decoder)
    
    class Esrgan_predictor(L.LightningModule):
        def __init__(self, generator, normalizer):
            super().__init__()
            self.generator = generator
            self.scale = 2
            self.normalizer = normalizer          
        
        def predict_step(self, batch):
            batch_normalized = self.normalizer(batch, 'image')
            lr_img = batch_normalized["image"]
            lr_img = lr_img.reshape(-1,args.patch_size,args.patch_size)\
                .unsqueeze(1)
            lr_img = lr_img.repeat(1, 3, 1, 1).to('cuda')
                        
          
            b, c, h, w = lr_img.shape
            scale = self.scale
            patch_size_lr = args.crop_size // 2
            stride = patch_size_lr // 2  # 50% sobreposição

            # Padding
            pad_h = (stride - h % stride) % stride
            pad_w = (stride - w % stride) % stride
            lr_img_padded = F.pad(lr_img, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, h_pad, w_pad = lr_img_padded.shape

            sr_img_accum = torch.zeros((b, c, h_pad * scale, w_pad * scale),
                                       device=lr_img.device)
            weight_mask = torch.zeros_like(sr_img_accum)

            for i in range(0, h_pad - patch_size_lr + 1, stride):
                for j in range(0, w_pad - patch_size_lr + 1, stride):
                    patch = lr_img_padded[:, :, i:i+patch_size_lr, j:j+patch_size_lr]
                    
                    sr_patch = self.generator(patch)  # [B, C, H', W']
                    
                    
                    i_sr, j_sr = i * scale, j * scale
                    sr_img_accum[:, :, i_sr:i_sr + patch_size_lr*scale,
                            j_sr:j_sr + patch_size_lr*scale] += sr_patch
                    weight_mask[:, :, i_sr:i_sr + patch_size_lr*scale, 
                            j_sr:j_sr + patch_size_lr*scale] += 1

            sr_img = sr_img_accum / weight_mask
            sr_img = sr_img[:, :, :h * scale, :w * scale]  # remove padding
            sr_img_mean = sr_img.mean(dim=1, keepdim=True)
            lr_img_mean = lr_img.mean(dim=1, keepdim=True)
            
            return  {
                    "lr": lr_img_mean.cpu(),
                    "sr": sr_img_mean.cpu(),
                    'crs':batch['crs'],
                    'bounds':batch['bounds']
                }
    
    
    cop = DEM(r"C:\Users\Eduardo JR\Fast\SRIU\COP_alinhado.tif")
    srtm = DEM(r"C:\Users\Eduardo JR\Fast\SRIU\SRTM_alinhado.tif")
    ram = DEM(r"C:\Users\Eduardo JR\Fast\SRIU\ram_dsm_15m.tif")
    dem_ds = cop & srtm
    
    #importante criar os normalizaers do dem_ds e do R
    input_norm = inicialize_normalizer(dem_ds,'image')
    ram_norm = inicialize_normalizer(ram, 'image')

    
    run_id = '46f0c76bfc2d4607b4a6cd23ade30d7c'
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    artifact_path = "weights/generator_weights.pth"
    
    local_path = client.download_artifacts(run_id, artifact_path)
    generator.load_state_dict(torch.load(local_path))
    generator.eval().cuda()
    esrgan_predictor = Esrgan_predictor(generator, input_norm)
    
    dm_dem = modules.DM_DEM(dataset=dem_ds,
                            batch_size=args.batch_size,
                            patch_size=args.patch_size,
                            num_workers=0)
    
    
    
    dm_dem.setup("predict")
    dataloader = dm_dem.predict_dataloader()
    
    with torch.no_grad():
        for batch in dataloader:
            predictions = esrgan_predictor.predict_step(batch)

    # abrir a imagem do RAM e recortar as mesmas áreas que em predictions
    ram_data = []
    for bound in predictions['bounds']:
        ram_data.append(ram.__getitem__(bound)['image'])
    predictions['hr'] = torch.stack(ram_data)
        # calcular as métricas de referência
    predictions['hr'] = torch.cat([predictions['hr'][i].repeat(2, 1, 1, 1) 
                                   for i in range(args.batch_size)], dim=0)
    ram_norm(predictions, 'hr')
    metricas = utils2.calculate_metrics_batch(predictions['hr'],
                                              predictions['sr'],
                                              2*args.batch_size)
    print({ 'psnr': metricas[0],
            'ssim': metricas[1],
            'mse': metricas[2]})
    

    # desnormalizando as lr, sr e ram
    predictions['lr'] = predictions['lr'].reshape(args.batch_size,2,
                                        args.patch_size,args.patch_size)
    input_norm.revert(predictions, 'lr')
    predictions['lr'] = predictions['lr'].reshape(2*args.batch_size,1,
                                        args.patch_size,args.patch_size)
    
    predictions['sr'] = predictions['sr'].reshape(args.batch_size,2,
                                    2*args.patch_size,2*args.patch_size)
    input_norm.revert(predictions, 'sr')
    predictions['sr'] = predictions['sr'].reshape(2*args.batch_size,1,
                                    2*args.patch_size,2*args.patch_size)
    
    ram_norm.revert(predictions, 'hr')
    
    utils2.plot_dem_predictions(predictions,
                                'SRIU/artifacts/pred/',
                                args.batch_size)    
    
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(vars(args))
    
        mlflow.log_artifacts("SRIU/artifacts/pred", 
                             artifact_path="prediction_imgs")
        mlflow.log_metrics({'psnr': metricas[0],
                            'ssim': metricas[1],
                            'mse': metricas[2]})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=' Predict DEM SR with a ESRGAN  pre trained model')
    parser.add_argument(
        '--crop_size',
        type= int,
        help= 'Crop size of each hr image in the batch',
        default= 128
    )

    parser.add_argument(
        '--batch_size',
        type= int,
        help='Number of item from the dataloader\
            (the real batch size is batch_size*num_crops)',
        default = 30
    )
    parser.add_argument(
        '--patch_size',
        type= int,
        help='Image size in each batch',
        default=1000
    )   
    parser.add_argument(
        '--run_name',
        type= str,
        help='name for the mlflow run',
        default= 'esrgan_treinado_teste_MDE'
    )
    args = parser.parse_args()

    # erase the imgs before a new experiments
    pasta_imagens = 'SRIU/artifacts'
    for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(pasta_raiz, arquivo)
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)
    ESRGAN_dem_prediction(args)