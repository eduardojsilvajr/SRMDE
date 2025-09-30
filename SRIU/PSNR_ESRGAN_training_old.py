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
import random


# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)

class PSNR_Esrgan(L.LightningModule):
    def __init__(self, generator, crop, rgb):
        super().__init__()
        self.generator = generator
        self.content_loss = nn.L1Loss()
        self.scale = 2
        self.crop = crop
        self.rgb = rgb
        # this is the line that enables manual optmization
        self.automatic_optimization = False
    
    def _prep_batch(self, batch):
        lr_img, hr_img = batch
        # lr_img = lr_img.reshape(-1,3,int(self.crop/2),int(self.crop/2))
        # hr_img = hr_img.reshape(-1,3,self.crop,self.crop)
        # return lr_img, hr_img
        if self.rgb:
            lr = lr_img.flatten(0,1).expand(-1, 3, -1, -1)
            hr = hr_img.flatten(0,1).expand(-1, 3, -1, -1)
            return lr, hr
        else:
            return lr_img.flatten(0,1), hr_img.flatten(0,1)
    
    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, 5, .5)
        return ([gen_opt], [gen_scheduler])

    def training_step(self, batch, batch_idx):
        # training only the generator with L1loss
        
        lr_img, hr_img = self._prep_batch(batch)          
        sr_img = self.generator(lr_img)
        assert not torch.isnan(sr_img).any(), f"valor NaN gerado pelo generator epoca {self.current_epoch}_{batch_idx}"
        
        gen_opt = self.optimizers()
        gen_scheduler= self.lr_schedulers()
        
        loss_content = self.content_loss(hr_img, sr_img)
        gen_loss = loss_content
        assert not torch.isnan(gen_loss).any(), "gen_loss NaN"
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        #important to prevent the model to crash
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0) #prevent gradient explosion
        gen_opt.step()
        gen_scheduler.step()
        current_lr = gen_opt.param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)        
        utils2.log_metrics(self, {
                    "t_g_loss": gen_loss,
                }, on_step=False, on_epoch=True, prog_bar=True)
        return {"t_g_loss": gen_loss}

    def validation_step(self, batch, batch_idx):
    # validation_step defines validation loop
        gen_loss = disc_loss = 0
        lr_img, hr_img = self._prep_batch(batch)
        batch_size = lr_img.shape[0]
        sr_img = self.generator(lr_img)
        assert not torch.isnan(sr_img).any(), f"valor NaN gerado pelo generator epoca {self.current_epoch}_{batch_idx}"

                    
        loss_content = self.content_loss(hr_img, sr_img)
        gen_loss = loss_content
        #metrics calculation    
        metrics_batch = utils2.calculate_metrics_batch(hr_img, sr_img)
        #plot and save images during validation
        if batch_idx%50==0:
            utils2.plot_random_crops_lr_sr_hr(lr_img, sr_img, hr_img,
                                                self.current_epoch, batch_idx,
                                                'SRIU/artifacts/val/')
        
        self.log_dict(metrics_batch, on_step=False, on_epoch=True, prog_bar=True )
        self.log("v_g_loss", gen_loss, on_step=False, on_epoch=True, prog_bar=True)    
        
    def test_step(self, batch, batch_idx):
        lr_img, hr_img = self._prep_batch(batch)
        batch_size = lr_img.shape[0]
        sr_img = self.generator(lr_img)
        assert not torch.isnan(sr_img).any(), f"valor NaN gerado pelo generator epoca {self.current_epoch}_{batch_idx}"
        metrics_batch = utils2.calculate_metrics_batch(hr_img, sr_img)
        self.log_dict(metrics_batch, on_step=False, on_epoch=True, prog_bar=True )


    def predict_step(self, batch):
        lr_img = batch[0]  # [B, C, H, W]
        name = batch[1]
        b, c, h, w = lr_img.shape
        scale = self.scale
        patch_size = self.crop // 2
        stride = patch_size // 2  # 50% sobreposição

        # Padding
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        lr_img_padded = F.pad(lr_img, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, h_pad, w_pad = lr_img_padded.shape

        sr_img_accum = torch.zeros((b, c, h_pad * scale, w_pad * scale), device=lr_img.device)
        weight_mask = torch.zeros_like(sr_img_accum)

        for i in range(0, h_pad - patch_size + 1, stride):
            for j in range(0, w_pad - patch_size + 1, stride):
                patch = lr_img_padded[:, :, i:i+patch_size, j:j+patch_size]
                with torch.no_grad():
                    sr_patch = self.generator(patch)  # [B, C, H', W']
                
                i_sr, j_sr = i * scale, j * scale
                sr_img_accum[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += sr_patch
                weight_mask[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += 1

        sr_img = sr_img_accum / weight_mask
        sr_img = sr_img[:, :, :h * scale, :w * scale]  # remove padding
        return lr_img, sr_img, name

def main(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="PSNR_ESRGAN",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')

    # Creating datasets
    to_tensor = ToTensor()
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
    
    # creating de discriminator
    # disc_encoder = modules.StridedConvEncoder(layers=(3, 64, 128, 128, 256, 256, 512, 512),
    #         layer_order = ("conv", "activation"), conv = modules.Conv2d,
    #         activation = modules.LeakyReLU)
    # disc_pool = nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d(kernel_size=X)
    # disc_head = modules.LinearHead( in_channels=512, out_channels=1,
    #             latent_channels=[1024],layer_order=("linear", "activation"))

    # discriminator = models.DiscriminatorVGG(encoder=disc_encoder,
    #                                         pool= disc_pool, head=disc_head)

    # losses
    # perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}) #wang et al 21, usa múltiplas camadas aqui.
    # gen_rela_loss = losses.RelativisticAdversarialLoss(mode="generator")
    # disc_rela_loss = losses.RelativisticAdversarialLoss(mode="discriminator")

    
    class ESRPredictor(L.LightningModule):
        def __init__(self, generator, scale=2, patch_size=64):
            super().__init__()
            self.generator = generator.eval()
            self.scale = scale
            self.patch_size = patch_size

        def predict_step(self, batch):
            lr_img = batch[0]  # [B, C, H, W]
            name = batch[1]
            b, c, h, w = lr_img.shape
            scale = self.scale
            patch_size = self.patch_size
            stride = patch_size // 2  # 50% sobreposição

            # Padding
            pad_h = (stride - h % stride) % stride
            pad_w = (stride - w % stride) % stride
            lr_img_padded = F.pad(lr_img, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, h_pad, w_pad = lr_img_padded.shape

            sr_img_accum = torch.zeros((b, c, h_pad * scale, w_pad * scale), device=lr_img.device)
            weight_mask = torch.zeros_like(sr_img_accum)

            for i in range(0, h_pad - patch_size + 1, stride):
                for j in range(0, w_pad - patch_size + 1, stride):
                    patch = lr_img_padded[:, :, i:i+patch_size, j:j+patch_size]
                    with torch.no_grad():
                        sr_patch = self.generator(patch)  # [B, C, H', W']
                    
                    i_sr, j_sr = i * scale, j * scale
                    sr_img_accum[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += sr_patch
                    weight_mask[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += 1

            sr_img = sr_img_accum / weight_mask
            sr_img = sr_img[:, :, :h * scale, :w * scale]  # remove padding
            return lr_img, sr_img, name


    
    early_stop_callback = EarlyStopping(
                        monitor="v_g_loss",
                        patience=2,
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

    trainer = L.Trainer(max_epochs=args.max_epochs, 
                        precision="16-mixed",
                        accelerator="gpu",
                        # fast_dev_run=True,
                        limit_train_batches=.25,
                        # limit_val_batches=10,
                        logger=mlflow_logger,
                        # limit_predict_batches=.5,
                        callbacks=[early_stop_callback])
     
    esrgan = PSNR_Esrgan(generator, args.crop_size, args.rgb)
    trainer.fit(esrgan, DM_div2_flickr)
      
    trainer.test(model=esrgan, dataloaders=DM_div2_flickr, ckpt_path="best")
    
    preds = trainer.predict(model=esrgan, dataloaders=DM_div2_flickr, ckpt_path="best")
    utils2.plot_prediction(preds,
                           'SRIU/artifacts/pred/')
    
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(vars(args))
        mlflow.log_artifact("SRIU/datasets/train_DIV2K_Flickr_pair.txt", artifact_path="datasets")
    
        mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
        mlflow.log_artifacts("SRIU/artifacts/pred", artifact_path="prediction_imgs")
       #substituir para salvar apenas os pesos do generator, não o modelo completo.
        # mlflow.pytorch.log_model(
        #     esrgan.generator,
        #     artifact_path="best_model",
        #     registered_model_name="PSNR_ESRGAN_Model",
        #     # code_paths=code_paths
        # ) 

    
    # model_uri = "runs:/14d33ccf46a44c6cb47a2b47be001f19/best_model"  # Substitua pelo seu run_id
    # mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")
    # model = mlflow.pytorch.load_model(model_uri)

    # predictor = ESRPredictor(model.generator)

    # # Use seu dataloader com imagens completas
    # preds = trainer.predict(predictor, DM_div2_flickr)
    # utils2.plot_prediction(preds,
    #                        'SRIU/artifacts/pred/')
    
    
   

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
        default=10
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
        default= 'teste_rotina_old_gray'
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
