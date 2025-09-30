import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # ou ":4096:8" se houver erro de memória
os.environ["PYTHONHASHSEED"] = "42"
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import v2
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback
import models, modules, losses, utils2
from L_callbacks import *
from L_modules import *
import argparse
import mlflow
from models import ESREncoder, ESRNetDecoder, EncoderDecoderNet, DiscriminatorVGG



# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)
L.seed_everything(42, workers=True)

def esrgan_generator_setup(in_channels:int, scale:int):
    gen_encoder = ESREncoder(in_channels=in_channels, out_channels=64, growth_channels=32,
                            num_basic_blocks=23, num_dense_blocks=3, num_residual_blocks=5, 
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

######parâmetros
def ESRGAN_training(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="ESRGAN",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000" 
        )
    torch.set_float32_matmul_precision('high')

    # Creating datasets
    mean_train =  [0.4492]     # [0.4679, 0.4488, 0.4033] ou [0.4492]
    std_train =   [0.2510]     # [0.2671, 0.2557, 0.2819] ou [0.2510]
    
    basic_transforms = v2.Compose([v2.ToImage(), 
                       v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean = mean_train, 
                                    std = std_train)
                       ]) 
    #mean = [0.4679, 0.4488, 0.4033], std = [0.2671, 0.2557, 0.2819] para RGB
    
    
    DM_div2_flickr = modules.DM_div2_flickr(train_dataset_file=args.train_dataset_file,
                                            test_dataset_file=args.test_dataset_file,
                                            root_dir=r"C:\Users\Eduardo JR\Fast",
                                            num_crops= args.num_crops, 
                                            crop_size_hr=args.crop_size,
                                            batch_size=args.batch_size,
                                            train_augment= args.train_set_augmantation, 
                                            transforms=basic_transforms,
                                            seed = seed,
                                            rgb=False)
    

    # creating the generator
    generator = esrgan_generator_setup(in_channels=1, scale=args.zoom)

    # creating de discriminator
    discriminator = esrgan_discriminator_setup(in_channels=1)

    # losses
    if args.unet_local_weigths:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}, #wang et al 21, usa múltiplas camadas aqui.
                        weights_path=args.unet_local_weigths,
                    mean = [0.4517] , std=[0.2434]) # imagenet-L mean = 0.4517, std=0.2434
    else:
        perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0})
    #wang et al 21, usa múltiplas camadas aqui.

    
    compute_metrics = Compute_metrics(ram_norm=None, input_norm=None, scale=args.zoom)
    # plot_val_results = Plot_results(val_batch_interval=60, test_batch_interval=60 )
    
    early_stop_callback = EarlyStopping(
                        monitor="v_g_loss",
                        patience=5,
                        mode="min",
                        verbose=True
                    )

    checkpoint_callback = ModelCheckpoint(
                        monitor="v_g_loss",
                        save_top_k=1,
                        mode="min",
                        filename="SRIU/saved_ckpt/esrgan_{epoch:02d}",
                        auto_insert_metric_name=False
                    )
    
    # reset_callback = ResetBestScoresCallback(reset_epoch=9)
    
    torch._dynamo.config.suppress_errors = True                         #type: ignore
    generator = utils2.compile_module(generator)
    discriminator = utils2.compile_module(discriminator)
    perceptual_loss = utils2.compile_module(perceptual_loss)

    trainer = L.Trainer(max_epochs=args.max_epochs, 
                        precision="16-mixed",
                        accelerator="gpu",
                        # fast_dev_run=5,
                        limit_train_batches=.3,
                        logger=mlflow_logger,
                        callbacks=[compute_metrics,
                                   checkpoint_callback, early_stop_callback])
     
    
    #### RGB
    # model_uri = "mlflow-artifacts:/908603184323111774/6dffeef130074a928cf4d64c0f303553/artifacts/model/MSRResNet_COP_e_20.ckpt"
    # psnr_esrgan = mlflow.pytorch.load_model(model_uri)
    #### Gray
    gen_ckp_path = r'SRIU\saved_ckpt\MSRResnet_gray_09.ckpt' 
    psnr_esrgan = MSRResNet.load_from_checkpoint(checkpoint_path=gen_ckp_path,
                                                     generator=generator)
    
    # psnr_esrgan = MSRResNet(generator= generator, lr= 5e-4, lr_decay=1e3,
    #                         rgb=False, mean = mean_train, std = std_train )
    
    esrgan = ESRGAN(generator=psnr_esrgan.generator, discriminator=discriminator,
                    perceptual_loss=perceptual_loss, rgb=False, lr=1e-4, 
                    step_milestones=[5000 , 10000, 20000, 30000],
                    mean = mean_train, std = std_train
                    )
    
    trainer.fit(model=esrgan, datamodule=DM_div2_flickr)
    trainer.test(model=esrgan, dataloaders=DM_div2_flickr, ckpt_path="best")
    
    # preds = trainer.predict(model=psnr_esrgan, dataloaders=DM_div2_flickr)
    # utils2.plot_prediction(preds,'SRIU/artifacts/pred/', cmap='Greys')

    
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(vars(args))    
        mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
        mlflow.log_artifacts("SRIU/artifacts/pred", artifact_path="prediction_imgs")
        mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path='model')
        # mlflow.log_artifact(gen_ckp_path, artifact_path='model')

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
        '--num_crops',
        type= int,
        help='Number of random crops in each image of the dataset',
        default = 4
    )
    parser.add_argument(
        '--batch_size',
        type= int,
        help='Number of item from the dataloader\
            (the real batch size is batch_size*num_crops)',
        default= 4
    )
    parser.add_argument(
        '--max_epochs',
        type= int,
        help='Maximum number os epochs in training',
        default=50
    )
    parser.add_argument(
        '--train_set_augmantation',
        type= bool,
        help='Enables data augmantation during training',
        default=True
    )
    parser.add_argument('--zoom',
            type= int,
            help='SR zoom',
            default=2
    )
    parser.add_argument(
        '--train_dataset_file',
        type= str,
        help='path to txt file with paired image',
        default=r'SRIU\datasets\train_DIV2K_Flickr_pair.txt'
    )
    parser.add_argument(
        '--test_dataset_file',
        type= str,
        help='path to txt file with paired image',
        default=r'SRIU\datasets\test_DIV2K_Flickr_pair.txt'
    )
    parser.add_argument(
        '--run_name',
        type= str,
        help='name for the mlflow run',
        default= 'ESRGAN_gray'
    )
    parser.add_argument(
        '--unet_local_weigths',
        type= str,
        help='weights of a 1band unet',
        default= r'SRIU\saved_ckpt\unet_1band_72.ckpt'
    )
    args = parser.parse_args()

    # erase the imgs before a new experiments
    pasta_imagens = 'SRIU/artifacts'
    for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(pasta_raiz, arquivo)
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)
    ESRGAN_training(args)