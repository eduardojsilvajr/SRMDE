import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import v2
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback
import models, modules, losses, utils2
from L_callbacks import Compute_metrics, Visualize_results
from L_modules import ESRGAN
import argparse
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'WebAgg' (escolha um backend interativo)
import matplotlib.pyplot as plt
import mlflow


# garante a reprodutibilidade do experimento
utils2.set_seed(42)
seed = torch.Generator().manual_seed(42)


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


######parâmetros
def PSNR_training(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="ESRGAN",  # nome do experimento,
        run_name=args.run_name,
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    torch.set_float32_matmul_precision('high')

    # Creating datasets
    basic_transforms = v2.Compose([v2.ToImage(), 
                       v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean = [0.4679, 0.4488, 0.4033], 
                                    std = [0.2671, 0.2557, 0.2819])
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
    gen_encoder = models.ESREncoder(in_channels=3, out_channels=64, growth_channels=32,
                            num_basic_blocks=23, num_dense_blocks=3, num_residual_blocks=5,
                            conv=modules.Conv2d, activation=modules.LeakyReLU, residual_scaling= 0.2)
    gen_decoder = models.ESRNetDecoder(in_channels=64, out_channels=3, scale_factor=2,
                                conv=modules.Conv2d, activation=modules.LeakyReLU)

    generator = models.EncoderDecoderNet(encoder=gen_encoder, decoder=gen_decoder)

    # creating de discriminator
    disc_encoder = modules.StridedConvEncoder(layers=(3, 64, 128, 128, 256, 256, 512, 512),
            layer_order = ("conv", "activation"), conv = modules.Conv2d,
            activation = modules.LeakyReLU)
    disc_pool = nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d(kernel_size=X)
    disc_head = modules.LinearHead( in_channels=512, out_channels=1,
                latent_channels=[1024],layer_order=("linear", "activation"))

    discriminator = models.DiscriminatorVGG(encoder=disc_encoder,
                                            pool= disc_pool, head=disc_head)

    # losses
    perceptual_loss = losses.PerceptualLoss(layers={"conv5_4": 1.0}) #wang et al 21, usa múltiplas camadas aqui.
    gen_rela_loss = losses.RelativisticAdversarialLoss(mode="generator")
    disc_rela_loss = losses.RelativisticAdversarialLoss(mode="discriminator")
    content_loss = nn.L1Loss()

    class Esrgan(L.LightningModule):
        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator = generator
            self.discriminator= discriminator
            self.scale = 2
            self.perceptual_loss = perceptual_loss
            self.gen_rela_loss = gen_rela_loss
            self.disc_rela_loss = disc_rela_loss
            self.content_loss = content_loss
            self.mean = torch.Tensor([0.4493])[:, None, None]
            self.std = torch.Tensor([0.2540])[:, None, None]
            # this is the line that enables manual optmization
            self.automatic_optimization = False
            
        def unormalize(self, x):
            return x*self.std +self.mean
        
        def configure_optimizers(self):
            gen_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
            disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
            gen_scheduler = torch.optim.\
                            lr_scheduler.MultiStepLR(gen_opt, 
                                                    milestones=[5000, 10000, 20000, 30000], 
                                                    gamma=0.5 )
            disc_scheduler = torch.optim.\
                            lr_scheduler.MultiStepLR(disc_opt, 
                                                    milestones=[5000, 10000, 20000, 30000], 
                                                    gamma=0.5 )
            return ([gen_opt, disc_opt], [gen_scheduler, disc_scheduler])
        
        def training_psnr_generator(self, sr_img, hr_img):
            gen_opt, _ = self.optimizers()
            gen_scheduler, _ = self.lr_schedulers()
            loss_content = self.content_loss(hr_img, sr_img)
            gen_loss = loss_content
            assert not torch.isnan(gen_loss).any(), "gen_loss NaN"
            gen_opt.zero_grad()
            self.manual_backward(gen_loss)
            #important to prevent the model to crash
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0) #prevent gradient explosion
            gen_opt.step()
            gen_scheduler.step()
            
            
            
            utils2.log_metrics(self, {
                        "t_g_loss": gen_loss,
                    }, on_step=False)
            
            
        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            lr_img, hr_img = batch
            lr_img = lr_img.flatten(0,1)
            hr_img = hr_img.flatten(0,1)
            sr_img = self.generator(lr_img)
            
            if self.current_epoch<10:
                #pretraning the generator with Content_loss
                self.training_psnr_generator(sr_img, hr_img)
            else:
                
                gen_opt, disc_opt = self.optimizers()
                gen_scheduler, disc_scheduler = self.lr_schedulers()
                 
                # Discriminator optimization
                d_real = self.discriminator(hr_img)
                d_fake = self.discriminator(sr_img.detach()) #important to separate from gen optimazation
                disc_loss = self.disc_rela_loss(d_fake,d_real)
                disc_opt.zero_grad()
                self.manual_backward(disc_loss)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0) #prevent gradient explosion
                disc_opt.step()
                disc_scheduler.step()
                
                # Generator optmization
                d_fake_gen = self.discriminator(sr_img)
                loss_adv = self.gen_rela_loss(d_fake_gen, d_real.detach()) #d_real was used on the discriminator opt
                loss_percep = self.perceptual_loss(hr_img, sr_img)
                loss_content = self.content_loss(sr_img, hr_img)
                gen_loss = loss_percep + 5e-3*loss_adv + 1e-2*loss_content
                gen_opt.zero_grad()
                self.manual_backward(gen_loss)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0) #prevent gradient explosion
                gen_opt.step()
                gen_scheduler.step()
                
                
                utils2.log_metrics(self, {
                            "t_g_loss": gen_loss,
                            "t_d_loss": disc_loss,
                            "t_pixel_loss": loss_content,
                            "t_percep_loss": loss_percep,
                            "t_adv_loss": loss_adv
                        }, on_step=False, on_epoch=True, prog_bar=True)
        
        def validation_psnr_generator(self, lr_img, sr_img, hr_img, batch_idx):
            loss_content = self.content_loss(hr_img, sr_img)
            gen_loss = loss_content
            #metrics calculation
            hr_img = self.unormalize(hr_img)
            sr_img = self.unormalize(sr_img)
            
            psnr_batch, ssim_batch, mse_batch = utils2.calculate_metrics_batch(hr_img, sr_img)
            #plot and save images during validation
            if batch_idx%50==0:
                utils2.plot_random_dem_crops_lr_sr_hr(lr_img, sr_img, hr_img,
                                                  self.current_epoch, batch_idx,
                                                  'SRIU/artifacts/val/','Greys')
            
            utils2.log_metrics(self, {
                                "v_g_loss": gen_loss,
                                "psnr": psnr_batch,
                                "ssim": ssim_batch,
                                "mse": mse_batch
                                }, on_step=False, on_epoch=True, prog_bar=True)  
        
        def validation_step(self, batch, batch_idx):
        # validation_step defines validation loop
            gen_loss = disc_loss = 0
            lr_img, hr_img = batch
            lr_img = lr_img.flatten(0,1)
            hr_img = hr_img.flatten(0,1)
            batch_size = lr_img.shape[0]
            sr_img = self.generator(lr_img)
            if self.current_epoch<10:
                self.validation_psnr_generator(lr_img, sr_img,hr_img,batch_idx)
            else:
                d_real = self.discriminator(hr_img)
                d_fake = self.discriminator(sr_img)
                disc_loss = self.disc_rela_loss(d_fake,d_real)
                loss_adv = self.gen_rela_loss(d_fake, d_real) #d_real was used on the discriminator opt
                loss_percep = self.perceptual_loss(hr_img, sr_img)
                loss_content = self.content_loss(hr_img, sr_img)
                gen_loss = loss_percep + 5e-3*loss_adv + 1e-2*loss_content

                hr_img = self.unormalize(hr_img)
                sr_img = self.unormalize(sr_img)
                psnr_batch, ssim_batch, mse_batch = utils2.calculate_metrics_batch(hr_img, sr_img)
                if batch_idx%50==0:
                    utils2.plot_random_dem_crops_lr_sr_hr(lr_img, sr_img, hr_img,
                                                    self.current_epoch, batch_idx,
                                                    'SRIU/artifacts/val/', 'Greys')
                
                utils2.log_metrics(self, {
                                    "v_g_loss": gen_loss,
                                    "v_d_loss": disc_loss,
                                    "psnr": psnr_batch,
                                    "ssim": ssim_batch,
                                    "mse": mse_batch
                                    }, on_step=False, on_epoch=True, prog_bar=True)    
        
        def test_step(self, batch, batch_idx):
            lr_img, hr_img = batch
            lr_img = lr_img.flatten(0,1)
            hr_img = hr_img.flatten(0,1)            
            batch_size = lr_img.shape[0]
            
            sr_img = self.generator(lr_img)
            assert not torch.isnan(sr_img).any(), f"valor NaN gerado pelo generator epoca {self.current_epoch}_{batch_idx}"
            
            hr_img = self.unormalize(hr_img)
            sr_img = self.unormalize(sr_img)            
            psnr_batch, ssim_batch, mse_batch = utils2.calculate_metrics_batch(hr_img, sr_img)
            utils2.log_metrics(self, {
                                "test_psnr": psnr_batch,
                                "test_ssim": ssim_batch,
                                "test_mse": mse_batch
                                }, on_step=False, on_epoch=True, prog_bar=True)
        
        def predict_step(self, batch):
            lr_img = batch[0] 
            name = batch[1]
            sr_img = utils2.stich_by_clip(lr_img, self.scale, 3,
                                          self.generator, 40)
            
            return lr_img, sr_img, name
    
    compute_metrics = Compute_metrics()
    plot_val_results = Plot_results(batch_interval=2, cmap='Greys')
    
    early_stop_callback = EarlyStopping(
                        monitor="v_loss",
                        patience=10,
                        mode="min",
                        verbose=True
                    )

    checkpoint_callback = ModelCheckpoint(
                        monitor="v_g_loss",
                        save_top_k=1,
                        mode="min",
                        filename="SRIU/saved_ckpt/1band_esrgan_{epoch:02d}",
                        auto_insert_metric_name=False
                    )
    
    reset_callback = ResetBestScoresCallback(reset_epoch=9)

    trainer = L.Trainer(max_epochs=args.max_epochs, 
                        precision="16-mixed",
                        accelerator="gpu",
                        fast_dev_run=5,
                        limit_train_batches=.3,
                        # logger=mlflow_logger,
                        callbacks=[compute_metrics, plot_val_results])
     
    
    model_uri = "runs:/45e00013657d4d35957e43c4432322d7/best_model" 
    mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000")
    psnr_esrgan = mlflow.pytorch.load_model(model_uri)
    esrgan = ESRGAN(generator=psnr_esrgan.generator, discriminator=discriminator,
                    perceptual_loss=perceptual_loss, rgb=True, lr=1e-4, 
                    step_milestones=[5000, 10000, 20000, 30000],
                    mean = [0.4679, 0.4488, 0.4033], std = [0.2671, 0.2557, 0.2819]
                    )
    
    trainer.fit(esrgan, DM_div2_flickr)

    # trainer.test(model=esrgan, dataloaders=DM_div2_flickr, ckpt_path="best")
    # preds = trainer.predict(model=esrgan, dataloaders=DM_div2_flickr, ckpt_path="best")
    
    trainer.test(model=esrgan, dataloaders=DM_div2_flickr)
    preds = trainer.predict(model=esrgan, dataloaders=DM_div2_flickr)

    
    utils2.plot_prediction(preds,'SRIU/artifacts/pred/')
    
    
    
    # with mlflow.start_run(run_id=mlflow_logger.run_id):
    #     mlflow.log_params(vars(args))
    #     mlflow.log_artifact("SRIU/datasets/train_DIV2K_Flickr_pair.txt", artifact_path="datasets")
    
    #     mlflow.log_artifacts("SRIU/artifacts/val", artifact_path="validation_imgs")
    #     mlflow.log_artifacts("SRIU/artifacts/pred", artifact_path="prediction_imgs")
    #     mlflow.pytorch.log_model(
    #         esrgan,
    #         artifact_path="best_model",
    #         registered_model_name="ESRGAN_Model",
    #     ) 


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
        default=100
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
        default= 'ESRGAN_lr_ativado_grayscale'
    )
    args = parser.parse_args()

    # erase the imgs before a new experiments
    pasta_imagens = 'SRIU/artifacts'
    for pasta_raiz, _, arquivos in os.walk(pasta_imagens):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(pasta_raiz, arquivo)
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)
    PSNR_training(args)