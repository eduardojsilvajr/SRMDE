import os
import torch
from torch import  nn
from torchvision.transforms import v2, ToTensor
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import models, modules, losses, utils2
import argparse
import mlflow
from L_modules import PSNR_ESRGAN
from L_callbacks import Compute_metrics, Plot_results


# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)


def main(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="PSNR_ESRGAN",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')

    # Creating datasets
    basic_transforms = v2.Compose([v2.ToImage(), 
                       v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean=[0.4492],
                                    std=[0.2510])
                                  ])
    to_tensor = ToTensor() 
    # mean = 0.4492, std= 0.2510 abrindo o DIV2K e Flickr na banda L
    # mean = [0.4679, 0.4488, 0.4033], std = [0.2671, 0.2557, 0.2819] para RGB
    DM_div2_flickr = modules.DM_div2_flickr(train_dataset_file=args.train_dataset_file,
                                            test_dataset_file=args.test_dataset_file,
                                            root_dir=r"C:\Users\Eduardo JR\Fast",
                                            num_crops= args.num_crops, 
                                            crop_size_hr=args.crop_size,
                                            batch_size=args.batch_size,
                                            train_augment= args.train_set_augmantation, 
                                            transforms=to_tensor,
                                            seed=seed,
                                            rgb=args.rgb)

    # creating the generator
    generator = models.generator_setup(1)
    generator = utils2.compile_module(generator)
    
    compute_metrics = Compute_metrics()
    
    plot_results = Plot_results(val_batch_interval=100, test_batch_interval=100, cmap='Greys')
    
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
                        dirpath="SRIU/saved_ckpt",
                        filename="best-psnr-epoch_{epoch:02d}",
                        auto_insert_metric_name=False
                    )
    callbacks = [early_stop_callback, checkpoint_callback,
                 compute_metrics, plot_results]
    
    trainer = L.Trainer(max_epochs=args.max_epochs, 
                        precision="16-mixed",
                        accelerator="gpu",
                        limit_train_batches=0.25,
                        # fast_dev_run=5,
                        # limit_val_batches=10,
                        logger=mlflow_logger,
                        # limit_predict_batches=.5,
                        callbacks=callbacks)
     
    esrgan = PSNR_ESRGAN(generator=generator, mean = [0.4492], std = [0.2510],
                         lr=2e-4, lr_decay=5e2, rgb=args.rgb)
    
    trainer.fit(esrgan, datamodule=DM_div2_flickr)
    trainer.test(model=esrgan, dataloaders=DM_div2_flickr, ckpt_path="best") #, ckpt_path="best"
    preds = trainer.predict(model=esrgan, dataloaders=DM_div2_flickr) #, ckpt_path="best"
    
    # esrgan = PSNR_ESRGAN.load_from_checkpoint(
    # checkpoint_path='SRIU/saved_ckpt/best-psnr-epoch_15.ckpt',
    # generator=generator, content_loss=content_loss,
    # lr=2e-4, lr_decay=10, rgb=args.rgb, cmap='Greys')
    # preds = trainer.predict(model=esrgan, dataloaders=DM_div2_flickr)

    utils2.plot_prediction(preds,'SRIU/artifacts/pred/', cmap='Greys')
    
    torch.save(esrgan.state_dict(),
                "SRIU/artifacts/model.pth")
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(vars(args))    
        mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
        mlflow.log_artifacts("SRIU/artifacts/pred", artifact_path="prediction_imgs")
        mlflow.log_artifact("SRIU/artifacts/model.pth", artifact_path='state_dict')



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
        default = 2
    )
    parser.add_argument(
        '--batch_size',
        type= int,
        help='Number of item from the dataloader\
            (the real batch size is batch_size*num_crops)',
        default= 12
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
        default= 'teste_gen_gray_Norm'
    )
    parser.add_argument(
        '--rgb',
        type= bool,
        help='True - RGB/ False - gray ',
        default= False
    )
    args = parser.parse_args()

    # erase the imgs before a new experiments
    pasta_imagens = 'SRIU/artifacts'
    for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(pasta_raiz, arquivo)
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)
    main(args)
