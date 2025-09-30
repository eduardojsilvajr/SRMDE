from typing import Callable, Dict, Iterable, Union, Optional
import utils2
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
import torchvision
from torchvision.transforms import Resize, InterpolationMode


class AdversarialLoss(_Loss):
    """GAN Loss function.

    Args:
        mode: Specifies loss terget: ``'generator'`` or ``'discriminator'``.
            ``'generator'``: maximize probability that fake data drawn from
            real data distribution (it is useful when training generator),
            ``'discriminator'``: minimize probability that real and generated
            distributions are similar.

    Raises:
        NotImplementedError: If `mode` not ``'generator'``
                or ``'discriminator'``.

    """

    def __init__(self, mode: str = "discriminator") -> None:
        super().__init__()

        if mode == "generator":
            self.forward = self.forward_generator
        elif mode == "discriminator":
            self.forward = self.forward_discriminator
        else:
            raise NotImplementedError()

    def forward_generator(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (generator mode).

        Args:
            fake_logits: Predictions of discriminator for fake data.
            real_logits: Predictions of discriminator for real data.

        Returns:
            Loss, scalar.

        """
        loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )

        return loss

    def forward_discriminator(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (discriminator mode).

        Args:
            fake_logits: Predictions of discriminator for fake data.
            real_logits: Predictions of discriminator for real data.

        Returns:
            Loss, scalar.

        """
        loss_real = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits), reduction="sum"
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits), reduction="sum"
        )

        num_samples = real_logits.shape[0] + fake_logits.shape[0]
        loss = (loss_real + loss_fake) / num_samples  # mean loss

        return loss


class RelativisticAdversarialLoss(_Loss):
    """Relativistic average GAN loss function.

    It has been proposed in `The relativistic discriminator: a key element
    missing from standard GAN`_.

    Args:
        mode: Specifies loss target: ``'generator'`` or ``'discriminator'``.
            ``'generator'``: maximize probability that fake data more realistic
            than real (it is useful when training generator),
            ``'discriminator'``: maximize probability that real data more
            realistic than fake (useful when training discriminator).

    Raises:
        NotImplementedError: If `mode` not ``'generator'``
            or ``'discriminator'``.

    .. _`The relativistic discriminator: a key element missing
        from standard GAN`: https://arxiv.org/pdf/1807.00734.pdf

    """

    def __init__(self, mode: str = "discriminator") -> None:
        super().__init__()

        if mode == "generator":
            self.rf_labels, self.fr_labels = 0, 1
        elif mode == "discriminator":
            self.rf_labels, self.fr_labels = 1, 0
        else:
            raise NotImplementedError()

    def forward(
        # self, outputs: torch.Tensor, targets: torch.Tensor
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward propagation method for the relativistic adversarial loss.

        Args:
            fake_logits: Probability that generated samples are not real.
            real_logits: Probability that real (ground truth) samples are fake.

        Returns:
            Loss, scalar.

        """
        loss_rf = F.binary_cross_entropy_with_logits(
            input=(real_logits - fake_logits.mean()),
            target=torch.empty_like(real_logits).fill_(self.rf_labels),
        )
        loss_fr = F.binary_cross_entropy_with_logits(
            input=(fake_logits - real_logits.mean()),
            target=torch.empty_like(fake_logits).fill_(self.fr_labels),
        )
        loss = (loss_fr + loss_rf) / 2

        return loss

def _layer2index_vgg16(layer: str) -> int:
    """Map name of VGG layer to corresponding number in torchvision layer.

    Args:
        layer: name of the layer e.g. ``'conv1_1'``

    Returns:
        Number of layer (in network) with name `layer`.

    Examples:
        >>> _layer2index_vgg16('conv1_1')
        0
        >>> _layer2index_vgg16('pool5')
        30

    """
    block1 = ("conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1")
    block2 = ("conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2")
    block3 = ("conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3")  # noqa: E501
    block4 = ("conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "pool4")  # noqa: E501
    block5 = ("conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5")  # noqa: E501
    layers_order = block1 + block2 + block3 + block4 + block5
    vgg16_layers = {n: idx for idx, n in enumerate(layers_order)}

    return vgg16_layers[layer]


def _layer2index_vgg19(layer: str) -> int:
    """Map name of VGG layer to corresponding number in torchvision layer.

    Args:
        layer: name of the layer e.g. ``'conv1_1'``

    Returns:
        Number of layer (in network) with name `layer`.

    Examples:
        >>> _layer2index_vgg16('conv1_1')
        0
        >>> _layer2index_vgg16('pool5')
        36

    """
    block1 = ("conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1")
    block2 = ("conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2")
    block3 = ("conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3")  # noqa: E501
    block4 = ("conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4")  # noqa: E501
    block5 = ("conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5")  # noqa: E501
    layers_order = block1 + block2 + block3 + block4 + block5
    vgg19_layers = {n: idx for idx, n in enumerate(layers_order)}

    return vgg19_layers[layer]


class PerceptualLoss(_Loss):
    """The Perceptual Loss.

    Calculates loss between features of `model` (usually VGG is used)
    for input (produced by generator) and target (real) images.

    Args:
        layers: Dict of layers names and weights (to balance different layers).
        model: Model to use to extract features.
        distance: Method to compute distance between features.
        mean: List of float values used for data standartization.
            If there is no need to normalize data, please use [0., 0., 0.].
        std: List of float values used for data standartization.
            If there is no need to normalize data, please use [1., 1., 1.].

    Raises:
        NotImplementedError: `distance` must be one of:
            ``'l1'``, ``'cityblock'``, ``'l2'``, or ``'euclidean'``,
            raise error otherwise.
    """

    def __init__(
        self,
        layers: Dict[str, float],
        model: str = "vgg19",
        weights_path: Optional[str] = None,
        distance: Union[str, Callable] = "l1",
        mean: Iterable[float] = (0.485, 0.456, 0.406),
        std: Iterable[float] = (0.229, 0.224, 0.225),
        
    ) -> None:
        super().__init__()

        model_fn = torchvision.models.__dict__[model]
        layer2idx = globals()[f"_layer2index_{model}"]

        w_sum = sum(layers.values())
        self.layers = {str(layer2idx(k)): w / w_sum for k, w in layers.items()}

        last_layer = max(map(layer2idx, layers))
        if weights_path:
            model_instance = model_fn(weights=None) #false
            new_conv = utils2.single_band_model(model_instance.features[0])
            model_instance.features[0] = new_conv
            
            #se eu não me engano tem que tirar isso pelo modelo não estar compilado.
            ckpt = torch.load(weights_path, map_location="cpu")
            sd = ckpt["state_dict"]
            new_sd = { k.replace("model._orig_mod.", ""): v for k, v in sd.items() }
            model_instance.load_state_dict(new_sd)
        else: 
            model_instance = model_fn(weights='VGG19_Weights.IMAGENET1K_V1')
        
        network = nn.Sequential(*list(model_instance.features)[:last_layer + 1]).eval()
        for param in network.parameters():
            param.requires_grad = False
        # self.model = network
        self.model = network.to("cuda")

        if callable(distance):
            self.distance = distance
        elif distance.lower() in {"l1", "cityblock"}:
            self.distance = F.l1_loss
        elif distance.lower() in {"l2", "euclidean"}:
            self.distance = F.mse_loss
        else:
            raise NotImplementedError()

        mean_tensor  = torch.tensor(mean).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std).view(1, -1, 1, 1)
        self.mean = mean_tensor.to(
            "cuda", non_blocking=True, memory_format=torch.channels_last
        )
        self.std = std_tensor.to(
            "cuda", non_blocking=True, memory_format=torch.channels_last
        )
        # self.resize = Resize([224,224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        
    def forward(
        self, fake_data: torch.Tensor, real_data: torch.Tensor
    ) -> torch.Tensor:
        """Forward propagation method for the perceptual loss.

        Args:
            fake_data: Batch of input (fake, generated) images.
            real_data: Batch of target (real, ground truth) images.

        Returns:
            Loss, scalar.
        """
        fake_features = self._get_features(fake_data)
        real_features = self._get_features(real_data)

        # calculate weighted sum of distances between real and fake features
        # loss = torch.tensor(0.0, requires_grad=True).to(fake_data)
        loss = torch.zeros((), device=fake_data.device, dtype=fake_data.dtype)
        for layer, weight in self.layers.items():
            layer_loss = F.l1_loss(fake_features[layer], real_features[layer])
            loss = loss + weight * layer_loss

        return loss

    def _get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        if x.shape[2]!=224:
            x = self._resize(x)
        
        x = (x - self.mean) / self.std

        # extract net features
        features: Dict[str, torch.Tensor] = {}
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features[name] = x

        return features
    def _resize(self, x:torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=False)
    

class PerceptualLoss_2(_Loss):
    """The Perceptual Loss.

    Calculates loss between features of `model` (usually VGG is used)
    for input (produced by generator) and target (real) images.

    Args:
        layers: Dict of layers names and weights (to balance different layers).
        model: Model to use to extract features.
        distance: Method to compute distance between features.
        mean: List of float values used for data standartization.
            If there is no need to normalize data, please use [0., 0., 0.].
        std: List of float values used for data standartization.
            If there is no need to normalize data, please use [1., 1., 1.].

    Raises:
        NotImplementedError: `distance` must be one of:
            ``'l1'``, ``'cityblock'``, ``'l2'``, or ``'euclidean'``,
            raise error otherwise.
    """

    def __init__(
        self,
        layers: Dict[str, float],
        model: str = "vgg19",
        distance: Union[str, Callable] = "l1",
        mean: Iterable[float] = (0.485, 0.456, 0.406),
        std: Iterable[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()

        model_fn = torchvision.models.__dict__[model]
        layer2idx = globals()[f"_layer2index_{model}"]

        w_sum = sum(layers.values())
        self.layers = {str(layer2idx(k)): w / w_sum for k, w in layers.items()}

        last_layer = max(map(layer2idx, layers))
        model_instance = model_fn(pretrained=True)
        network = nn.Sequential(*list(model_instance.features)[:last_layer + 1]).eval()
        for param in network.parameters():
            param.requires_grad = False
        self.model = network

        if callable(distance):
            self.distance = distance
        elif distance.lower() in {"l1", "cityblock"}:
            self.distance = F.l1_loss
        elif distance.lower() in {"l2", "euclidean"}:
            self.distance = F.mse_loss
        else:
            raise NotImplementedError()

        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(
        self, fake_data: torch.Tensor, real_data: torch.Tensor
    ) -> torch.Tensor:
        """Forward propagation method for the perceptual loss.

        Args:
            fake_data: Batch of input (fake, generated) images.
            real_data: Batch of target (real, ground truth) images.

        Returns:
            Loss, scalar.
        """
        fake_features = self._get_features(fake_data)
        real_features = self._get_features(real_data)

        # calculate weighted sum of distances between real and fake features
        loss = torch.tensor(0.0, requires_grad=True).to(fake_data)
        for layer, weight in self.layers.items():
            layer_loss = F.l1_loss(fake_features[layer], real_features[layer])
            loss = loss + weight * layer_loss

        return loss

    def _get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # normalize input tensor
        x = (x - self.mean.to(x)) / self.std.to(x)

        # extract net features
        features: Dict[str, torch.Tensor] = {}
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features[name] = x

        return features
