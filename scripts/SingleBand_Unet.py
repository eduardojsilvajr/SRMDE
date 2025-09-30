import torch
import torchvision
from torch import nn 
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import  utils2
from typing import Any, Callable, Dict, List
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from torchmetrics import Accuracy
from ImageNet_class import ImageNet_class


# garante a reprodutibilidade do experimento
# utils2.set_seed(42)
# seed = torch.Generator().manual_seed(42)
# L.seed_everything(42, workers=True)


def prepara_dataset_imagenet(
    root_dir: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42)-> list[list]:
    
    # Lista para armazenar os arquivos encontrados
    samples = []           # lista de (caminho, label)
    inv_imagent_class = {}
    for k,v in ImageNet_class.items():
        v = v.split(',')[0]
        v = v.replace(' ', '_').lower()
        inv_imagent_class[v] = k
    # Percorre todos os arquivos e subpastas
    root = Path(root_dir)
    for label, class_dir in enumerate(sorted(root.iterdir())):
        if not class_dir.is_dir():
            continue
        label = inv_imagent_class[class_dir.name]
        for img_path in class_dir.glob('*.*'):
            samples.append((str(img_path), label))

    paths, labels = zip(*samples)
    # 2.1) retira test
    paths_trainval, paths_test, labels_trainval, labels_test = train_test_split(
        paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_trainval, labels_trainval,
        test_size=val_ratio,
        stratify=labels_trainval,
        random_state=seed
    )

    train_samples = list(zip(paths_train, labels_train))
    val_samples   = list(zip(paths_val,   labels_val))
    test_samples  = list(zip(paths_test,  labels_test))
        
    
    return [train_samples, val_samples, test_samples]

class DS_imagenet540k(Dataset):
    def __init__(self, 
                 samples: List,
                 singleband:bool,
                 transform: Callable[[Any], Any]):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.singleband = singleband
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path_img, rotulo = self.samples[idx]      
        
           # força 3 canais
        if self.singleband:
            image = Image.open(path_img).convert("L")
            norm = v2.Normalize(
                mean=([0.4517]),
                std=([0.2434]))
        else:
            image = Image.open(path_img).convert("RGB")
            norm = v2.Normalize(
                    mean=self.mean,
                    std=self.std)
        
        pipeline = v2.Compose([
            self.transform, 
            norm])

        # 3) aplica sem alterar self.transform
        image = pipeline(image)
        
        return image, rotulo
    
class DM_imagenet540k(L.LightningDataModule):
    def __init__(self,
                train_samples: List,
                val_samples: List,
                test_samples: List,
                rotulos: Dict[int, str],
                singleband:bool, 
                transform: Callable[[Any], Any],
                batch_size:int = 4):
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.rotulos = rotulos
        self.singleband = singleband
        self.batch_size = batch_size
        self.transform = transform
    
    def setup(self, stage):
        if stage == 'fit':
            self.train_ds = DS_imagenet540k(samples=self.train_samples,
                                            transform=self.transform,
                                            singleband=self.singleband)
            self.val_ds = DS_imagenet540k(samples=self.val_samples,
                                          transform=self.transform,
                                          singleband=self.singleband)
        if stage =='test':
            self.test_ds = DS_imagenet540k(samples=self.test_samples,
                                           transform=self.transform,
                                           singleband=self.singleband)
    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size= self.batch_size,
                          shuffle=True,
                          num_workers=6, 
                          pin_memory=True, 
                          persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size= self.batch_size,
                          shuffle=False,
                          num_workers=6, 
                          pin_memory=True, 
                          persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size= self.batch_size,
                          shuffle=False,
                          num_workers=6, 
                        pin_memory=True, 
                        persistent_workers=True)

class SingleBandClassifier(L.LightningModule):
        def __init__(self,  model:torch.nn.Module, singleband:bool):
            super().__init__()
            self.model = model
            if singleband:
                ### substitui o conv1 (features[0]) de 1x3x64 por 1x1x64
                conv1 = self.model.features[0]
                new_conv = utils2.single_band_model(conv=conv1, weigths=True, input=True)
                self.model.features[0] = new_conv
            
            # 1) congela *tudo*
            for p in self.model.parameters():
                p.requires_grad = False

            # 2) libera conv1 e a cabeça (classifier)
            for p in self.model.features[0].parameters():
                p.requires_grad = True
            for p in self.model.classifier.parameters():
                p.requires_grad = True

            self.loss_fn = nn.CrossEntropyLoss()
            self.val_acc =  Accuracy(task="multiclass", num_classes=1000)
            self.test_acc =  Accuracy(task="multiclass", num_classes=1000)
            self.save_hyperparameters('singleband')
        
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.model.parameters(), 
                                    lr=0.0001)
            sched = torch.optim.lr_scheduler.StepLR(opt, 10, .5,)
            return ([opt], [sched])
        
        def on_train_epoch_start(self) -> None:
        # na época == freeze_epochs, libera todo o backbone
            if self.current_epoch == 3:
                for p in self.model.features.parameters():
                    p.requires_grad = True
                self.print("info", f"Unfroze backbone at epoch {self.current_epoch}")

        
        def on_fit_start(self) -> None:
            self.model.to(memory_format=torch.channels_last) # type: ignore[call-overload]
            torch._dynamo.config.suppress_errors = True      # type: ignore[call-overload]
            self.model = torch.compile(
                self.model,
                mode="default",           
                backend="aot_eager",        
            )
            
        def forward(self, imgs):
            return self.model(imgs)

        def training_step(self, batch):
            imgs, labels = batch
            imgs = imgs.to(memory_format=torch.channels_last, non_blocking=True)            
            logits  = self(imgs)
            batch_loss = self.loss_fn(logits, labels)
            self.log("gen_loss", batch_loss, on_epoch=True, on_step=True, prog_bar=True)
            return batch_loss
        
        def validation_step(self, batch):
            imgs, labels = batch
            imgs = imgs.to(memory_format=torch.channels_last, non_blocking=True)            
            logits = self.model(imgs)
            batch_loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            self.val_acc.update(preds, labels)
            self.log("val_loss", batch_loss, on_epoch=True, on_step=True, prog_bar=True)
            

        def on_validation_epoch_end(self):
            self.log("val_acc", self.val_acc.compute(), prog_bar=True)
            self.val_acc.reset()        
                
        def test_step(self, batch):
            imgs, labels = batch
            imgs = imgs.to(memory_format=torch.channels_last, non_blocking=True)            
            logits = self.model(imgs)
            # batch_loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            self.test_acc.update(preds, labels)
            # self.log("test_loss", batch_loss, on_epoch=True, on_step=False, prog_bar=True)

        def on_test_epoch_end(self):
            self.log("test_acc", self.test_acc.compute(), prog_bar=True)
            self.test_acc.reset()
        
def main(args):
   
    
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

   
    transform = v2.Compose([v2.ToImage(), 
                    v2.RandomResizedCrop(size=224, scale=(.5,1)),
                    v2.RandomHorizontalFlip(),
                    v2.ToDtype(torch.float32, scale=True),])
    
    # dir_imagenet = r"C:\Users\Eduardo JR\Fast\imagenet_540"
    train_samples, val_samples, test_samples = prepara_dataset_imagenet(args.dir_imagenet)
    
    # create datamodules
    DM_imagenet540 = DM_imagenet540k(train_samples=train_samples,
                                     val_samples=val_samples,
                                     test_samples=test_samples,
                                     rotulos=ImageNet_class,
                                     batch_size=args.batch_size,
                                     transform=transform,
                                     singleband=True)
    
    
        
    early_stop_callback = EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        mode="min",
                        verbose=True
                    )
    
    checkpoint_callback = ModelCheckpoint(
                        monitor="val_loss",
                        save_top_k=1,
                        mode="min",
                        dirpath="SRIU/saved_ckpt",
                        filename="unet_1band_{epoch:02d}",
                        auto_insert_metric_name=False
                    )
    mlflow_logger = MLFlowLogger(
        experiment_name="Default",  # nome do experimento,
        run_name='VGG19_monocanal',
        tracking_uri="http://127.0.0.1:5000"        # ou um servidor MLflow remoto
        )
    
    L_trainer = L.Trainer(max_epochs= args.max_epochs,
                        precision="16-mixed",
                        accelerator="gpu",
                        devices=1,
                        # strategy=strategy,
                        limit_train_batches=0.2,
                        # limit_val_batches=0.2,
                        logger=mlflow_logger,
                        callbacks=[early_stop_callback, checkpoint_callback]
                        ) 
    vgg19 = torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1', num_classes=1000)
    unet_1band = SingleBandClassifier(model= vgg19,singleband=True)
    
    L_trainer.fit(unet_1band, datamodule=DM_imagenet540)
    L_trainer.test(unet_1band, datamodule=DM_imagenet540)
    
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=' Train a ESRGAN PSNR oriented model')
    parser.add_argument(
        '--batch_size',
        type= int,
        help='Number of item from the dataloader',
        default= 128 
    )
    parser.add_argument(
        '--max_epochs',
        type= int,
        help='Maximum number os epochs in training',
        default=100
    )
    parser.add_argument(
        '--dir_imagenet',
        type= str,
        help='imaget directory files',
        default=r"C:\Users\Eduardo JR\Fast\imagenet_540"
    )
    args = parser.parse_args()
    main(args)
