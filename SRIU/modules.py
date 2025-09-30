import collections
from typing import Any, Callable, List, Tuple, Iterable, Optional, Dict
import functools
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import os
import utils2
import random
from torchvision.transforms import functional as TF
import lightning as L



Conv2d: Callable[..., nn.Module] = functools.partial(
    nn.Conv2d, kernel_size=(3, 3), padding=1,# padding_mode='replicate',
)
LeakyReLU: Callable[..., nn.Module] = functools.partial(
    nn.LeakyReLU, negative_slope=0.2, inplace=True
)




class ConcatInputModule(nn.Module):
    """Module wrapper, passing outputs of all previous layers
    into each next layer. DenseNet implementation

    Args:
        module: PyTorch layer to wrap.

    """

    def __init__(self, module: Iterable[nn.Module]) -> None:
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.

        """
        output = [x]
        for module in self.module:
            z = torch.cat(output, dim=1)
            output.append(module(z))

        return output[-1]


class ResidualModule(nn.Module):
    """Residual wrapper, adds identity connection.
    ResNet implementation
    It has been proposed in `Deep Residual Learning for Image Recognition`_.

    Args:
        module: PyTorch layer to wrap.
        scale: Residual connections scaling factor.
        requires_grad: If set to ``False``, the layer will not learn
            the strength of the residual connection.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/pdf/1512.03385.pdf

    """

    def __init__(
        self,
        module: nn.Module,
        scale: float = 1.0,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.module = module
        self.scale = nn.Parameter(
            torch.tensor(scale), requires_grad=requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.

        """
        return x + self.scale * self.module(x)


class ResidualDenseBlock(ResidualModule):
    """Basic block of :py:class:`ResidualInResidualDenseBlock`.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`.
        growth_channels: Number of channels in the latent space.
        num_blocks: Number of convolutional blocks to use to form dense block.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use after convolution
            e.g., :py:class:`nn.LeakyReLU`.
        residual_scaling: Residual connections scaling factor.

    """

    def __init__(
        self,
        num_features: int,
        growth_channels: int,
        num_blocks: int = 5,
        conv: Callable[..., nn.Module] = Conv2d,
        activation: Callable[..., nn.Module] = LeakyReLU,
        residual_scaling: float = 0.2,
    ) -> None:
        in_channels = [
            num_features + i * growth_channels for i in range(num_blocks)
        ]
        out_channels = [growth_channels] * (num_blocks - 1) + [num_features]

        blocks: List[nn.Module] = []
        for in_channels_, out_channels_ in zip(in_channels, out_channels):
            block = collections.OrderedDict([
                ("conv", conv(in_channels_, out_channels_)),
                ("act", activation()),
            ])
            blocks.append(nn.Sequential(block))

        super().__init__(
            module=ConcatInputModule(nn.ModuleList(blocks)),
            scale=residual_scaling,
        )


class ResidualInResidualDenseBlock(ResidualModule):
    """Residual-in-Residual Dense Block (RRDB).

    Look at the paper: `ESRGAN: Enhanced Super-Resolution Generative
    Adversarial Networks`_ for more details.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`.
        growth_channels: Number of channels in the latent space.
        num_dense_blocks: Number of dense blocks to use to form `RRDB` block.
        residual_scaling: Residual connections scaling factor.
        **kwargs: Dense block params.

    .. _`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`:
        https://arxiv.org/pdf/1809.00219.pdf

    """

    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        num_dense_blocks: int = 3,
        residual_scaling: float = 0.2,
        **kwargs: Any,
    ) -> None:
        blocks: List[Tuple[str, nn.Module]] = []
        for i in range(num_dense_blocks):
            block = ResidualDenseBlock(
                num_features=num_features,
                growth_channels=growth_channels,
                residual_scaling=residual_scaling,
                **kwargs,
            )
            blocks.append((f"block_{i}", block))

        super().__init__(
            module=nn.Sequential(collections.OrderedDict(blocks)),
            scale=residual_scaling
        )

class StridedConvEncoder(nn.Module):
    """Generalized Fully Convolutional encoder.

    Args:
        layers: List of feature maps sizes of each block.
        layer_order: Ordered list of layers applied within each block.
            For instance, if you don't want to use normalization layer
            just exclude it from this list.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        norm: Class constructor or partial object which when called should
            return normalization layer e.g., :py:class:`.nn.BatchNorm2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use e.g., :py:class:`nn.ReLU`.
        residual: Class constructor or partial object which when called
            should return block wrapper module e.g.,
            :py:class:`esrgan.nn.ResidualModule` can be used
            to add residual connections between blocks.

    """

    def __init__(
        self,
        layers: Iterable[int] = (3, 64, 128, 128, 256, 256, 512, 512),
        strides:Optional[Iterable[int]] = [1,2,1,2,1,2,1,2],
        layer_order: Iterable[str] = ("conv", "norm", "activation"),
        conv: Callable[..., nn.Module] = Conv2d,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation: Callable[..., nn.Module] = LeakyReLU,
        residual: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        name2fn: Dict[str, Callable[..., nn.Module]] = {
            "activation": activation,
            "conv": conv,
            "norm": norm,
        }

        self._layers = list(layers)
        self.strides = strides

        net: List[Tuple[str, nn.Module]] = []

        first_conv = collections.OrderedDict([
            ("conv_0", name2fn["conv"](self._layers[0], self._layers[1])),
            ("act", name2fn["activation"]()),
        ])
        net.append(("block_0", nn.Sequential(first_conv)))

        channels = utils2.pairwise(self._layers[1:])
        for i, (in_ch, out_ch) in enumerate(channels, start=1):
            block_list: List[Tuple[str, nn.Module]] = []
            for name in layer_order:
                # `conv + 2x2 pooling` is equal to `conv with stride=2`
                if self.strides == None:
                    kwargs = {"stride": out_ch // in_ch} if name == "conv" else {}
                else:
                    kwargs = {"stride": self.strides[i]} if name == "conv" else {} # type: ignore

                module = utils2.create_layer(
                    layer_name=name,
                    layer=name2fn[name],
                    in_channels=in_ch,
                    out_channels=out_ch,
                    **kwargs
                )
                block_list.append((name, module))
            block = nn.Sequential(collections.OrderedDict(block_list))

            # add residual connection, like in resnet blocks
            if residual is not None and in_ch == out_ch:
                block = residual(block)

            net.append((f"block_{i}", block))

        self.net = nn.Sequential(collections.OrderedDict(net))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Batch of embeddings.

        """
        output = self.net(x)

        return output

    @property
    def in_channels(self) -> int:
        """The number of channels in the feature map of the input.

        Returns:
            Size of the input feature map.

        """
        return self._layers[0]

    @property
    def out_channels(self) -> int:
        """Number of channels produced by the block.

        Returns:
            Size of the output feature map.

        """
        return self._layers[-1]


class LinearHead(nn.Module):
    """Stack of linear layers used for embeddings classification.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        latent_channels: Size of the latent space.
        layer_order: Ordered list of layers applied within each block.
            For instance, if you don't want to use activation function
            just exclude it from this list.
        linear: Class constructor or partial object which when called
            should return linear layer e.g., :py:class:`nn.Linear`.
        activation: Class constructor or partial object which when called
            should return activation function layer e.g., :py:class:`nn.ReLU`.
        norm: Class constructor or partial object which when called
            should return normalization layer e.g., :py:class:`nn.BatchNorm1d`.
        dropout: Class constructor or partial object which when called
            should return dropout layer e.g., :py:class:`nn.Dropout`.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: Optional[Iterable[int]] = None,
        layer_order: Iterable[str] = ("linear", "activation"),
        linear: Callable[..., nn.Module] = nn.Linear,
        activation: Callable[..., nn.Module] = LeakyReLU,
        norm: Optional[Callable[..., nn.Module]] = None,
        dropout: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        name2fn: Dict[str, Callable[..., nn.Module]] = {
            "activation": activation,
            "dropout": dropout,
            "linear": linear, #type: ignore
            "norm": norm,
        }

        latent_channels = latent_channels or []
        channels = [in_channels, *latent_channels, out_channels]
        channels_pairs: List[Tuple[int, int]] = list(utils2.pairwise(channels))

        net: List[nn.Module] = []
        for in_ch, out_ch in channels_pairs[:-1]:
            for name in layer_order:
                module = utils2.create_layer(
                    layer_name=name,
                    layer=name2fn[name],
                    in_channels=in_ch,
                    out_channels=out_ch,
                )
                net.append(module)
        net.append(name2fn["linear"](*channels_pairs[-1]))

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs e.g. images.

        Returns:
            Batch of logits.

        """
        output = self.net(x)

        return output


class InterpolateConv(nn.Module):
    """Upsamples a given multi-channel 2D (spatial) data.

    Args:
        num_features: Number of channels in the input tensor.
        scale_factor: Factor to increase spatial resolution by.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use after convolution
            e.g., :py:class:`nn.PReLU`.

    """

    def __init__(
        self,
        num_features: int,
        scale_factor: int = 2,
        conv: Callable[..., nn.Module] = Conv2d,
        activation: Callable[..., nn.Module] = LeakyReLU,
    ) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.block = nn.Sequential(
            conv(num_features, num_features),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Upscale input -> apply conv -> apply nonlinearity.

        Args:
            x: Batch of inputs.

        Returns:
            Upscaled data.

        """
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        output = self.block(x)

        return output   

class DS_div2_flickr(Dataset):
    def __init__(self, 
                 dataset_file: str,
                 root_dir:str,
                 num_crops:int,
                 crop_size_hr:int =128, 
                 scale:int=2,
                 rgb: bool=True, 
                 transforms:Optional[Callable[..., nn.Module]] = None, 
                 augment:bool=True):
        """_summary_

        Args:
            dataset_file (str): _description_
            root_dir (_type_): _description_
            num_crops (str): _description_
            crop_size_hr (int, optional): _description_. Defaults to 128.
            scale (int, optional): _description_. Defaults to 2.
            transforms (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.
            augment (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.root_dir =root_dir
        self.dataset_file = dataset_file
        self.scale = scale
        self.crop_size_hr = crop_size_hr
        self.crop_size_lr = crop_size_hr // scale
        self.transforms = transforms
        self.num_crops = num_crops
        self.rgb = rgb
        self.augment = augment
        self.image_pairs = []
        
        # Read image pairs from text file and store in hr_path, lr_path
        with open(self.dataset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    hr_path, lr_path = [x.strip() for x in line.split(',')]
                    self.image_pairs.append((hr_path, lr_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.image_pairs[idx]
        
        if self.rgb:
            lr_image = Image.open(os.path.join(self.root_dir,lr_path)).convert('RGB')
            hr_image = Image.open(os.path.join(self.root_dir,hr_path)).convert('RGB')
        else:
            lr_image = Image.open(os.path.join(self.root_dir,lr_path)).convert('L')
            hr_image = Image.open(os.path.join(self.root_dir,hr_path)).convert('L')            
        
        hr_width, hr_height = hr_image.size
        # Listas para armazenar os crops
        lr_crops = []
        hr_crops = []
        
        # num_crops =1 <=> test_ds.
        if self.num_crops<2:
            lr_crops.append(self.transforms(lr_image))
            hr_crops.append(self.transforms(hr_image))
        else:

            for _ in range(self.num_crops):
                ### Gerar posições aleatórias para cada crop
                top_hr = random.randint(0, hr_height - self.crop_size_hr)
                left_hr = random.randint(0, hr_width - self.crop_size_hr)
                top_lr = top_hr // self.scale
                left_lr = left_hr // self.scale

                ### Fazer o crop
                hr_crop = TF.crop(hr_image, top_hr, left_hr, self.crop_size_hr, self.crop_size_hr)
                lr_crop = TF.crop(lr_image, top_lr, left_lr, self.crop_size_lr, self.crop_size_lr)
                # hr_crop = TF.center_crop(hr_image,self.crop_size_hr )
                # lr_crop = TF.center_crop(lr_image, self.crop_size_lr)
                
                ### Verificar alinhamento
                assert lr_crop.size[0] * self.scale == hr_crop.size[0], "Misaligned crop"
                
                ### Aumento de dados coerente
                if self.augment:
                    if random.random() < 0.5:
                        hr_crop = TF.hflip(hr_crop)
                        lr_crop = TF.hflip(lr_crop)
                    if random.random() < 0.5:
                        angle = random.choice([90, 180, 270])
                        hr_crop = TF.rotate(hr_crop, angle)
                        lr_crop = TF.rotate(lr_crop, angle)

                if self.transforms:
                    lr_crop = self.transforms(lr_crop)
                    hr_crop = self.transforms(hr_crop)

                lr_crops.append(lr_crop)
                hr_crops.append(hr_crop)

        ### Empilhar os crops para formar tensores com dimensão extra
        lr_stack = torch.stack(lr_crops)
        hr_stack = torch.stack(hr_crops)
        
        return lr_stack, hr_stack

class DS_set5_set14(Dataset):
    def __init__(self,
                 dataset_file: str,
                 rgb:bool = True,
                 transforms:Optional[Callable[..., nn.Module]] = None, 
):
        super().__init__()
        self.dataset_file = dataset_file
        self.img_paths = []
        self.rgb = rgb
        self.transforms = transforms
        
        with open(self.dataset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.img_paths.append(line)
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        if self.rgb:
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path).convert('L')
            # img = Image.open(img_path).convert('RGB')
            # r, g, b = img.split()
            # img = r

            
        name = img_path.split('\\')[-1][:-4]
        
        if self.transforms:
                img = self.transforms(img)
        return img, name

class DM_div2_flickr(L.LightningDataModule):
    def __init__(self,
                train_dataset_file: str,
                test_dataset_file: str,
                root_dir: str,
                num_crops:int,
                seed:torch.Generator,
                crop_size_hr:int =128, 
                batch_size:int = 4,
                transforms:Optional[Callable[..., nn.Module]] = None, 
                train_augment:bool=True,
                rgb:bool = True):
        
        super().__init__()
        self.train_dataset_file = train_dataset_file
        self.test_dataset_file = test_dataset_file
        self.predict_dataset = r'D:\Documentos\OneDrive\Documentos\Mestrado\SR - MDE\SRIU\datasets\predict_Set5_Set14.txt'
        self.root_dir = root_dir
        self.crop_size_hr = crop_size_hr
        self.num_crops = num_crops
        self.batch_size = batch_size
        self.transforms = transforms
        self.train_augment = train_augment
        self.rgb = rgb
        self.seed = seed

        
    def setup(self, stage):
        if stage =='fit':
            train_dataset = DS_div2_flickr(dataset_file=self.train_dataset_file, 
                                         root_dir=self.root_dir,
                                         num_crops= self.num_crops, 
                                         crop_size_hr=self.crop_size_hr,
                                         augment= self.train_augment,
                                         transforms=self.transforms,
                                         rgb=self.rgb)
            self.train_split, self.val_split = random_split(train_dataset, lengths=[.75,.25], generator= self.seed)
    
        if stage =='test':
            self.test_dataset = DS_div2_flickr(dataset_file=self.test_dataset_file, 
                                        root_dir=self.root_dir,
                                        num_crops= 1,
                                        crop_size_hr=self.crop_size_hr,
                                        augment= False,  
                                        transforms=self.transforms,
                                        rgb=self.rgb)
        
        if stage == 'predict':
            self.predict_dataset = DS_set5_set14(dataset_file=self.predict_dataset,
                                                 transforms=self.transforms,
                                                 rgb=self.rgb)
        
            
    def train_dataloader(self):
        return DataLoader(self.train_split, 
                        batch_size=self.batch_size, 
                        shuffle=False,
                        # num_workers=4, 
                        # pin_memory=True, 
                        # persistent_workers=True
                        )
    def val_dataloader(self):
        return DataLoader(self.val_split, 
                        batch_size=self.batch_size, 
                        shuffle=False,
                        # num_workers=4, 
                        # pin_memory=True, 
                        # persistent_workers=True
                        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                        batch_size=1,
                        shuffle=False,
                        # num_workers=4, 
                        # pin_memory=True, 
                        # persistent_workers=True
                        )
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          shuffle=False)


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

