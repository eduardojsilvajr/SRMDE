import os, re, random, copy, itertools, math
import numpy as np
import inspect
import math
from typing import Any, Iterable, Callable, Optional, Union
import torch
from torch import nn
import io
import matplotlib.pyplot as plt
import torch.nn.functional as F

# import piq



def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """Iterate sequences by pairs.

    Args:
        iterable: Any iterable sequence.

    Returns:
        Pairwise iterator.

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        ...     print(i)
        (1, 2)
        (2, 5)
        (5, -3)

    """
    a, b = itertools.tee(iterable, 2)
    next(b, None)
    return zip(a, b)

def is_power_of_two(number: int) -> bool:
    """Check if a given number is a power of two.

    Args:
        number: Nonnegative integer.

    Returns:
        ``True`` or ``False``.

    Examples:
        >>> is_power_of_two(4)
        True
        >>> is_power_of_two(3)
        False

    """
    result = number == 0 or (number & (number - 1) != 0)
    return result

def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> None:
    """Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification`_.

    Args:
        tensor: An n-dimensional tensor.
        a: The slope of the rectifier used after this layer
            (only used with ``'leaky_relu'`` and ``'prelu'``).
        mode: Either ``'fan_in'`` or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes
            in the backwards pass.
        nonlinearity: The non-linear function (`nn.functional` name).

    .. _`Delving deep into rectifiers: Surpassing human-level performance
        on ImageNet classification`: https://arxiv.org/pdf/1502.01852.pdf

    """
    base_act = "relu" if nonlinearity == "prely" else nonlinearity
    nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=base_act)

    if nonlinearity == "prelu":
        with torch.no_grad():
            std_correction = math.sqrt(1 + a ** 2)
            tensor.div_(std_correction)

def module_init_(
    module: nn.Module,
    nonlinearity: Union[str, nn.Module, None] = None,
    **kwargs: Any,
) -> None:
    """Initialize module based on the activation function.

    Args:
        module: Module to initialize.
        nonlinearity: Activation function. If LeakyReLU/PReLU and of type
            `nn.Module`, then initialization will be adapted by value of slope.
        **kwargs: Additional params to pass in init function.

    """
    # get name of activation function and extract slope param if possible
    activation_name: Optional[str] = None
    init_kwargs = copy.deepcopy(kwargs)
    if isinstance(nonlinearity, str):
        activation_name = nonlinearity.lower()
    elif isinstance(nonlinearity, nn.Module):
        activation_name = nonlinearity.__class__.__name__.lower()
        assert isinstance(activation_name, str)

        if activation_name == "leakyrelu":  # leakyrelu == LeakyReLU.lower
            activation_name = "leaky_relu"
            init_kwargs["a"] = kwargs.get("a", nonlinearity.negative_slope)
        elif activation_name == "prelu":
            init_kwargs["a"] = kwargs.get("a", nonlinearity.weight.data)

    # select initialization
    if activation_name in {"sigmoid", "tanh"}:
        weignt_init_fn: Callable = nn.init.xavier_uniform_
        init_kwargs = kwargs
    elif activation_name in {"relu", "elu", "leaky_relu", "prelu"}:
        weignt_init_fn = kaiming_normal_
        init_kwargs["nonlinearity"] = activation_name
    else:
        weignt_init_fn = nn.init.normal_
        init_kwargs["std"] = kwargs.get("std", 0.01)

    # init weights of the module
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        weignt_init_fn(module.weight, **init_kwargs)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def net_init_(net: nn.Module) -> None:
    """Inplace initialization of weights of neural network.

    Args:
        net: Network to initialize.

    """
    # create set of all activation functions (in PyTorch)
    activations = tuple(
        m[1]
        for m in inspect.getmembers(nn.modules.activation, inspect.isclass)
        if m[1].__module__ == "torch.nn.modules.activation"
    )

    # init of the layer depends on activation used after it,
    #  so iterate from the last layer to the first
    activation: Optional[nn.Module] = None
    for m in reversed(list(net.modules())):
        if isinstance(m, activations):
            activation = m

        module_init_(m, nonlinearity=activation)

def create_layer(
    layer: Callable[..., nn.Module],
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    layer_name: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Helper function to generalize layer creation.

    Args:
        layer: Layer constructor.
        in_channels: Size of the input sample.
        out_channels: Size of the output e.g. number of channels
            produced by the convolution.
        layer_name: Name of the layer e.g. ``'activation'``.
        **kwargs: Additional params to pass into `layer` function.

    Returns:
        Layer.

    Examples:
        >>> in_channels, out_channels = 10, 5
        >>> create_layer(nn.Linear, in_channels, out_channels)
        Linear(in_features=10, out_features=5, bias=True)
        >>> create_layer(nn.ReLU, in_channels, out_channels, layer_name='act')
        ReLU()

    """
    module: nn.Module
    if layer_name in {"activation", "act", "dropout", "pool", "pooling"}:
        module = layer(**kwargs)
    elif layer_name in {"normalization", "norm", "bn"}:
        module = layer(out_channels, **kwargs)
    else:
        module = layer(in_channels, out_channels, **kwargs)

    return module

def set_seed(seed: int):
    """Configura a semente para reprodutibilidade."""
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # para usar o pad 'reflect', tem comentar aqui
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    
def plot_random_crops_lr_sr_hr(lr_imgs: torch.Tensor, sr_imgs: torch.Tensor,
                               hr_imgs: torch.Tensor, epoch: int, batch_idx:int,
                               save_img_path:str, cmap:str = 'terrain')-> None:
    """
    Plota os 4 primeiros crops aleatórios (LR, SR, HR em colunas) e salva no diretório indicado.

    Args:
        lr_imgs (torch.Tensor): Tensor [B, 1 or 3, H, W]
        sr_imgs (torch.Tensor): Tensor [B, 1 or 3, H*2, W*2]
        hr_imgs (torch.Tensor): Tensor [B, 1 or 3, H*2, W*2]
        epoch (int): Época atual
    """
    C = lr_imgs.shape[1]
    fig, axs = plt.subplots(3, 3, figsize=(9, 12))
    if epoch==None:
        fig.suptitle(f"Teste set- {batch_idx}", fontsize=14)
    else:
        fig.suptitle(f"Época - {epoch}/{batch_idx}", fontsize=14)
    axs[0, 0].set_title("LR"); axs[0, 1].set_title("SR"); axs[0, 2].set_title("HR")
    indices = random.sample(range(len(lr_imgs)), 3)
    for i, idx in enumerate(indices):
        if C==1:
            lr = lr_imgs[idx][0].float().cpu().detach()
            sr = sr_imgs[idx][0].float().cpu().detach()
            hr = hr_imgs[idx][0].float().cpu().detach()
        else:
            lr = lr_imgs[idx].float().permute(1, 2, 0).cpu().detach()
            sr = sr_imgs[idx].float().permute(1, 2, 0).cpu().detach()
            hr = hr_imgs[idx].float().permute(1, 2, 0).cpu().detach()

        if lr.min() < 0:
            lr = (lr + 1) / 2
            sr = (sr + 1) / 2
            hr = (hr + 1) / 2

        if C==1:
            axs[i, 0].imshow(lr, cmap=cmap) 
            axs[i, 1].imshow(sr, cmap=cmap)
            axs[i, 2].imshow(hr, cmap=cmap)
            side_text = f"idx={idx}\
\nHR min={hr.min():0.1f}/max={hr.max():0.1f}\nSR min={sr.min():0.1f}/max={sr.max():0.1f}"
        else:
            axs[i, 0].imshow(lr) 
            axs[i, 1].imshow(sr)
            axs[i, 2].imshow(hr)
            side_text = f"idx={idx}"
        
        axs[i, 0].text(
        -0.05, 0.5,             # x < 0 para "sair" ligeiramente à esquerda do eixo
        side_text,           # texto a exibir
        transform=axs[i, 0].transAxes,  # coordenadas em eixo [0..1]
        va='center',            # centraliza verticalmente
        ha='right',             # alinha texto à direita (de modo a ficar próximo da imagem)
        fontsize= 9,            # tamanho de fonte (ajuste conforme preferência)
        color='black',          # cor do texto
        weight='bold'
        ) 

        for j in range(3):
            axs[i, j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])                            #type: ignore
    plt.pause(1)
    if epoch==None:
        plt.savefig(save_img_path+f"test_batch_{batch_idx}.png")
    else:    
        plt.savefig(save_img_path+f"epoch_{epoch}_batch_{batch_idx}.png")
    plt.close(fig)
    
def val_plot_dem_hr_sr_dif(sr_imgs: torch.Tensor, hr_imgs: torch.Tensor, epoch: int, batch_idx:int,
                               save_img_path:str, cmap:str = 'terrain')-> None:
    """
    Plota os 2 primeiros crops do batch (HR, SR e Dif em colunas) e salva no diretório indicado.

    Args:
        sr_imgs (torch.Tensor): Tensor [B, 1, H, W]
        hr_imgs (torch.Tensor): Tensor [B, 1, H, W]
        epoch (int): Época atual
    """
    fig, axs = plt.subplots(2, 3, figsize=(9, 12), constrained_layout=True)
    fig.suptitle(f"Época - {epoch}/{batch_idx}", fontsize=14)
    axs[0, 0].set_title("HR"); axs[0, 1].set_title("SR"); axs[0, 2].set_title("Dif")
    for i, idx in enumerate(range(2)):
           
        sr = sr_imgs[idx][0].float().cpu().detach()
        hr = hr_imgs[idx][0].float().cpu().detach()
        dif = sr - hr

        fig_hr = axs[i, 0].imshow(hr, cmap=cmap, vmin=hr.min(), vmax=hr.max()) 
        axs[i, 1].imshow(sr, cmap=cmap, vmin=hr.min(), vmax=hr.max())
        fig_dif = axs[i, 2].imshow(dif, cmap='coolwarm', vmin=-10, vmax=10)

        cbar_left  = fig.colorbar(fig_hr, ax=axs[i, 0:2], location="left", shrink=0.6, pad=0.05, anchor=(1.0, 0.5))
        cbar_left.set_label("Elevação (m)")
        cbar_right = fig.colorbar(fig_dif, ax=axs[i, 2], location="right", shrink=0.6, pad=0.05)
        cbar_right.set_label("Diferença (m)")


        for j in range(3):
            axs[i, j].axis("off")
        

    # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.pause(1)
    plt.savefig(save_img_path+f"epoch_{epoch}_batch_{batch_idx}.png")
    plt.close(fig)
        
    
def plot_prediction(predictions:list,
                    save_img_path:str,
                    cmap:str = None                                      #type: ignore
                    ):
    len_preds = len(predictions)
    C = predictions[0][0].shape[1]
    for idx in range(len_preds):
        if C ==3:            
            lr = predictions[idx][0].float().squeeze(0).permute(1, 2, 0).detach()
            sr = predictions[idx][1].float().squeeze(0).permute(1, 2, 0).detach()
        else:
            lr = predictions[idx][0].float().cpu().detach().squeeze(0).squeeze(0)
            sr = predictions[idx][1].float().cpu().detach().squeeze(0).squeeze(0)
        
        name = predictions[idx][2]
        fig, axs = plt.subplots(1,2, dpi = 300)
        fig.suptitle(f"Predições {name}", fontsize=14)
        axs[0].set_title("LR"); axs[1].set_title("SR")
        if C==3:    
            axs[0].imshow(lr)
            axs[1].imshow(sr)
        else:
            axs[0].imshow(lr, cmap=cmap)
            axs[1].imshow(sr, cmap=cmap)

        plt.tight_layout()
        plt.pause(1)
        plt.savefig(save_img_path+f"pred_{name}.png")
        plt.close(fig)

def plot_dem_predictions(predictions, save_img_path, batch_size):
    for idx in range(0, batch_size*2, 2):            
        lr_cop = predictions['lr'][idx].float().permute(1, 2, 0).detach()
        sr_cop = predictions['sr'][idx].float().permute(1, 2, 0).detach()
        hr_ram = predictions['hr'][idx].float().permute(1, 2, 0).detach()
        lr_srtm = predictions['lr'][idx+1].float().permute(1, 2, 0).detach()
        sr_srtm = predictions['sr'][idx+1].float().permute(1, 2, 0).detach()
        dif_cop = sr_cop -hr_ram
        dif_srtm = sr_srtm-hr_ram
        fig, axs = plt.subplots(2,4, figsize=(20,10), dpi = 300)
        fig.suptitle(f"Predições do batch {idx//2+1}", fontsize=14)
        # Títulos das colunas
        fontsize = 8
        axs[0, 0].set_title("LR", fontsize=fontsize)
        axs[0, 1].set_title("RAM", fontsize=fontsize)
        axs[0, 2].set_title("SR", fontsize=fontsize)
        axs[0, 3].set_title("Dif", fontsize=fontsize)

        # axs[1, 0].set_title("SRTM LR", fontsize=fontsize)
        # axs[1, 1].set_title("RAM", fontsize=fontsize)
        # axs[1, 2].set_title("SRTM SR", fontsize=fontsize)
        # axs[1, 3].set_title("Dif SRTM", fontsize=fontsize)

        # Imagens
        min_dif = -10
        max_dif = 10
        axs[0, 0].imshow(lr_cop, cmap='terrain')
        axs[0, 1].imshow(hr_ram, cmap='terrain')
        axs[0, 2].imshow(sr_cop, cmap='terrain')
        axs[0, 3].imshow(dif_cop, cmap='coolwarm', vmin =min_dif, vmax=max_dif )

        axs[1, 0].imshow(lr_srtm, cmap='terrain')
        axs[1, 1].imshow(hr_ram, cmap='terrain')
        axs[1, 2].imshow(sr_srtm, cmap='terrain')
        axs[1, 3].imshow(dif_srtm, cmap='coolwarm', vmin =min_dif, vmax=max_dif)
        
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        # Rótulos à esquerda
        fig.text(0.06, 0.73, "COP", va='center', ha='center', rotation=90, fontsize=10, weight='bold')
        fig.text(0.06, 0.28, "SRTM", va='center', ha='center', rotation=90, fontsize=10, weight='bold')
        
        plt.tight_layout(rect=[0.08, 0.03, 1, 0.95])                #type: ignore
        plt.savefig(save_img_path+f"dem_prediction_batch_{idx//2+1}.png")
        plt.close(fig)
    
def plot_dem_predictions_mopa(lr:torch.Tensor, sr:torch.Tensor, hr:torch.Tensor, 
                              batch_idx:int, save_img_path:str):
    scale = int(hr.shape[3]/lr.shape[3])
    bic = F.interpolate(lr, scale_factor= scale, mode='bicubic',
                        align_corners=False, antialias=False)
    
    bic = bic.squeeze(0).squeeze(0).cpu()
    sr = sr.squeeze(0).squeeze(0).cpu()
    hr = hr.squeeze(0).squeeze(0).cpu()
    
    dif = sr - hr
    dif_bic = bic - hr
    # fig, axs = plt.subplots(2,3, figsize=(20,10), dpi = 300, constrained_layout=True)
    fig = plt.figure(figsize=(25, 15), dpi=300, constrained_layout=True)
    # [cbar_esq | COL0 | COL1 | COL2 | cbar_dir]
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[0.08, 1.6, 2, 2, 0.08],   # mesmas larguras para as 3 colunas centrais
        wspace=0.01,                            # espaçamento horizontal uniforme
        hspace=0.01
    )
    fontsize = 14
    fig.suptitle(f"Predições do batch {batch_idx}", fontsize=2*fontsize)
    
    ax_hr      = fig.add_subplot(gs[:, 1])    # COL0 ocupa 2 linhas
    ax_sr      = fig.add_subplot(gs[0, 2])    # COL1 topo
    ax_bic     = fig.add_subplot(gs[1, 2])    # COL1 baixo
    ax_diff_sr = fig.add_subplot(gs[0, 3])    # COL2 topo
    ax_diff_bc = fig.add_subplot(gs[1, 3])    # COL2 baixo
    
    cax_left  = fig.add_subplot(gs[:, 0])
    cax_right = fig.add_subplot(gs[:, 4])
    
    # Títulos das colunas
    ax_hr.set_title("HR (RAM)", fontsize=fontsize)
    ax_sr.set_title("SR", fontsize=fontsize)
    ax_bic.set_title("Bicubic", fontsize=fontsize)
    ax_diff_sr.set_title("Δ (SR − HR)", fontsize=fontsize)
    ax_diff_bc.set_title("Δ (Bic − HR)", fontsize=fontsize)

    # Imagens
    vmin_terrain, vmax_terrain = float(hr.min()), float(hr.max())
    vabs = 10.0   # ou calcule algo data-driven (ex.: torch.quantile(|dif|, .995))
    vmin_diff, vmax_diff = -vabs, vabs
    
    im_hr  = ax_hr.imshow(hr,  cmap='terrain',  vmin=vmin_terrain, vmax=vmax_terrain)
    im_sr  = ax_sr.imshow(sr,  cmap='terrain',  vmin=vmin_terrain, vmax=vmax_terrain)
    im_bic = ax_bic.imshow(bic, cmap='terrain', vmin=vmin_terrain, vmax=vmax_terrain)

    im_d1 = ax_diff_sr.imshow(dif,     cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff)
    im_d2 = ax_diff_bc.imshow(dif_bic, cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff)

    # Eixos sem ticks
    for ax in (ax_hr, ax_sr, ax_bic, ax_diff_sr, ax_diff_bc):
        ax.set_xticks([]); ax.set_yticks([])
        # ax.set_box_aspect(None)
        # ax.set_aspect('auto')
    
    # terrain_axes = [ax_hr, ax_sr, ax_bic]
    cbar_left  = fig.colorbar(im_hr, cax=cax_left)
    cbar_left.set_label("Elevação (m)")

    # Colorbar 2 (diferenças) compartilhada: coluna 2
    # diff_axes = [ax_diff_sr, ax_diff_bc]
    cbar_right = fig.colorbar(im_d1, cax=cax_right)
    cbar_right.set_label("Diferença (m)")



    plt.savefig(os.path.join(save_img_path, f"predict_{batch_idx}.png"), bbox_inches='tight')
    plt.close(fig)    
    
def log_metrics(module, loss_dict: dict, on_step: bool=True, on_epoch: bool=True, prog_bar: bool=True ):
    """
    Loga métricas de perda com média por época e valores por iteração.
    
    Parâmetros:
        module: instância do LightningModule (normalmente 'self')
        loss_dict: dicionário com nomes e valores de perdas
    """
    for name, value in loss_dict.items():
        module.log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
        
# def calculate_metrics_batch(img_hr: torch.Tensor ,img_sr: torch.Tensor)-> dict: 
#     """calculate the PSNR, SSIM and MSE in a batch of tensors
#     Args:
#         img_hr (torch.tensor): _description_
#         gen_hr (torch.tensor): _description_

#     Returns:
#         list[float]: _description_
#     """
    
#     assert not torch.isnan(img_hr).any(), "HR contém NaN!"
#     assert not torch.isinf(img_sr).any(), "SR contém infinito!"
    
    
#     psnr_batch = piq.psnr(img_hr.float(), img_sr.float())
#     # psnr_batch = psnr_batch.cpu().detach().numpy()
#     psnr_item = psnr_batch.item()
#     ssim_batch = piq.ssim(img_hr.float(), img_sr.float())
#     # ssim_batch = ssim_batch.cpu().detach().numpy()
#     ssim_item = ssim_batch.item()
#     # mse_batch = mean_squared_error(imgs_hr_flaten, gen_hr_flaten)
#     mse_batch = nn.functional.mse_loss(img_sr, img_hr)
#     mse_item = mse_batch.item()

#     return {'psnr':psnr_item, 'ssim': ssim_item,'mse': mse_item}

def nanstd(tensor, dim=None, keepdim=False):
    # Calcula a média ignorando NaNs
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    # Diferença ao quadrado
    squared_diff = (tensor - mean)**2
    # Média das diferenças ao quadrado (ignorando NaNs)
    var = torch.nanmean(squared_diff, dim=dim, keepdim=keepdim)
    # Desvio padrão é a raiz quadrada da variância
    return torch.sqrt(var)


def ordena_filtra_profiler(in_file: str) -> None:
    """
    Lê um arquivo texto contendo múltiplos blocos de saída do cProfile/PyTorch Lightning Profiler,
    onde cada bloco começa com "Profile stats for: <rótulo>" e em algum ponto contém uma linha
    "in XXX.YYY seconds". Retorna uma lista de tuplas (tempo_total, bloco_texto_completo),
    ordenada pelo tempo_total em ordem decrescente. Blocos com tempo_total == 0 são descartados.
    
    Args:
        in_file (str): caminho para o arquivo .txt com todas as saídas concatenadas.
        
    """
    # Regex para localizar início de bloco e capturar rótulo (não usamos rótulo aqui, mas serve de marcador)
    padrao_label = re.compile(r"^Profile stats for:\s*(.+)$")
    # Regex para extrair o valor de tempo na linha “in XXX.YYY seconds”
    padrao_tempo = re.compile(r"in\s+(\d+\.\d+)\s+seconds")
    
    # Vamos carregar todas as linhas do arquivo em memória
    with open(os.path.join(os.getcwd(), in_file), "r", encoding="utf-8") as f:
        linhas = f.readlines()
    
    blocos: list[tuple[float, str]] = []
    current_block_lines: list[str] = []
    current_label: str | None = None
    
    i = 0
    while i < len(linhas):
        linha = linhas[i].rstrip("\n")
        m_label = padrao_label.match(linha)
        
        # Se encontrar um novo “Profile stats for: …”, fechamos o bloco anterior (se existir)
        if m_label:
            # Primeiro, se já estávamos coletando um bloco anterior, processamos ele
            if current_block_lines:
                bloco_texto = "".join(current_block_lines)
                
                # Procurar, dentro desse bloco, a primeira ocorrência de “in XXX.YYY seconds”
                m_tempo = padrao_tempo.search(bloco_texto)
                if m_tempo:
                    tempo_total = float(m_tempo.group(1))
                    if tempo_total > 0.0:
                        blocos.append((tempo_total, bloco_texto))
                # Resetamos para começar a coletar o novo bloco
                current_block_lines = []
            
            # Agora iniciamos um novo bloco com esta linha de rótulo
            current_label = m_label.group(1).strip()
            current_block_lines.append(linha + "\n")
        
        else:
            # Se já iniciamos um bloco (current_label != None), continuamos a coletar
            if current_label is not None:
                current_block_lines.append(linha + "\n")
            # Caso contrário, estamos em linhas que não pertencem a nenhum bloco (pode ser cabeçalho/trailer), então ignoramos
        
        i += 1
    
    # Depois de sair do loop, precisamos processar o último bloco pendente (se existir)
    if current_block_lines:
        bloco_texto = "".join(current_block_lines)
        m_tempo = padrao_tempo.search(bloco_texto)
        if m_tempo:
            tempo_total = float(m_tempo.group(1))
            if tempo_total > 0.0:
                blocos.append((tempo_total, bloco_texto))
    
    # Ordenar a lista de blocos pelo tempo_total, em ordem decrescente
    blocos.sort(key=lambda x: x[0], reverse=True)
    nome_saida = in_file.split('.')
    nome_saida[0]+='_filtrado'
    out_path = '.'.join(nome_saida)
    
    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, (tempo, texto_bloco) in enumerate(blocos, start=1):
            # fout.write(f"\n=== Bloco #{idx}: tempo total = {tempo:.3f} segundos ===\n\n")
            fout.write(texto_bloco)
            # fout.write("\n")  # linha em branco entre blocos (opcional)
    os.remove(in_file)
    
def check_hr_on_dict(batch: dict)-> bool:
    return 'hr' in batch.keys()

def stich_by_average(img: torch.Tensor, scale:int, 
                     patch_size:int, stride:int, generator: nn.Module):
    b, c, h, w = img.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    lr_img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, h_pad, w_pad = lr_img_padded.shape
    
    sr_img_accum = torch.zeros((b, c, h_pad * scale, w_pad * scale), device=img.device)
    weight_mask = torch.zeros_like(sr_img_accum)
    
    for i in range(0, h_pad - patch_size + 1, stride):
        for j in range(0, w_pad - patch_size + 1, stride):
            patch = lr_img_padded[:, :, i:i+patch_size, j:j+patch_size]
            with torch.no_grad():
                sr_patch = generator(patch)  # [B, C, H', W']
            
            i_sr, j_sr = i * scale, j * scale
            sr_img_accum[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += sr_patch
            weight_mask[:, :, i_sr:i_sr + patch_size*scale, j_sr:j_sr + patch_size*scale] += 1

    sr_img = sr_img_accum / weight_mask
    sr_img = sr_img[:, :, :h * scale, :w * scale]
    
    return sr_img 

def stich_by_clip(img: torch.Tensor, 
    n: int, generator: nn.Module, 
    pad: int = 40, scale: int=2 
) -> torch.Tensor:
    
    b, c, H, W = img.shape

    # aplica padding reflexivo uma única vez
    img_pad = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')

    # calcula tamanho aproximado de cada patch na região original
    ph = math.ceil(H / n)
    pw = math.ceil(W / n)

    # prepara tensor de saída
    H_sr, W_sr = H * scale, W * scale
    out = torch.zeros((b, c, H_sr, W_sr), device=img.device)

    cp = pad * scale  # quantidade a recortar em SR

    with torch.no_grad():
        for i in range(n):
            y0 = i * ph
            y1 = min((i + 1) * ph, H)
            ys, ye = y0 + pad, y1 + pad

            for j in range(n):
                x0 = j * pw
                x1 = min((j + 1) * pw, W)
                xs, xe = x0 + pad, x1 + pad

                # extrai patch LR + contexto de 'pad' px em volta
                lr_patch = img_pad[:, :, ys - pad : ye + pad,
                                       xs - pad : xe + pad]

                # gera SR para o patch completo
                sr_patch = generator(lr_patch)

                # recorta o contexto (pad*scale) de cada borda
                sr_center = sr_patch[
                    :,
                    :,
                    cp : sr_patch.size(2) - cp,
                    cp : sr_patch.size(3) - cp
                ]

                # cola o patch central no lugar correto da saída
                out[
                    :,
                    :,
                    y0 * scale : y1 * scale,
                    x0 * scale : x1 * scale
                ] = sr_center
    return out

def single_band_model(conv:nn.Module, weigths:bool = False, input:bool = True):
    if input:
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None))
        if weigths:
            w3 = conv.weight.data
            w1 = w3.mean(dim=1, keepdim=True)
            new_conv.weight.data.copy_(w1)
    else:
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=1,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None))
        if weigths:
            w3 = conv.weight.data
            w1 = w3.mean(dim=0, keepdim=True)
            new_conv.weight.data.copy_(w1)
    return new_conv

def compile_module(module:nn.Module):
    module.to(memory_format=torch.channels_last) #type: ignore[arg-type]
    return torch.compile( module, mode="default", backend="aot_eager", dynamic=True)


##### CLASSES ####
class Normalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, stdev: torch.Tensor):
        super().__init__()

        self.mean = mean[:, None, None].to('cuda')
        self.std = stdev[:, None, None].to('cuda')
        
    def forward(self, inputs: dict,label: str):
        x = inputs[label][..., : len(self.mean), :, :]
        
        if inputs[label].ndim == 4:
            x = (x - self.mean[None, ...]) / self.std[None, ...]
            # x = torch.clamp(x, 0,1)
        else:
            x = (x - self.mean[:x.shape[0]]) / self.std[:x.shape[0]]
            # x = torch.clamp(x, 0,1)
        inputs[label][..., : len(self.mean), :, :] = x
    
        return inputs

    def revert(self, inputs: dict, label: str):
        """
        De-normalize the batch.
        Args:
            inputs (dict): Dictionary with the 'label' key
        """
        x = inputs[label][..., : len(self.mean), :, :]

        if x.ndim == 4:
            x = x * self.std[None, ...] + self.mean[None, ...]
        else:
            x = x * self.std[:x.shape[0]] + self.mean[:x.shape[0]]

        inputs[label][..., : len(self.mean), :, :] = x

        return inputs

class Normalizer_minmax(nn.Module):
    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        super().__init__()

        self.min = min[:, None, None].to('cuda')
        self.max = max[:, None, None].to('cuda')
        self.range = (self.max - self.min).clamp_min(1e-6)
        
    def forward(self, inputs: dict,label: str):
        x = inputs[label][..., : len(self.min), :, :]
        
        if inputs[label].ndim == 4:
            x = (x - self.min[None, ...]) / self.range[None, ...]
            # x = torch.log(x-self.min[None, ...])/torch.log(self.range[None, ...]) 
        else:
            x = (x - self.min[:x.shape[0]]) / self.range[:x.shape[0]]
            # x = torch.log(x-self.min[:x.shape[0]])/torch.log(self.range[:x.shape[0]])

        inputs[label][..., : len(self.min), :, :] = x
    
        return inputs

    def revert(self, inputs: dict, label: str):
        """
        De-normalize the batch.
        Args:
            inputs (dict): Dictionary with the 'label' key
        """
        x = inputs[label][..., : len(self.min), :, :]

        if x.ndim == 4:
            x = x * self.range[None, ...] + self.min[None, ...]
            # x = self.range[None, ...]**x + self.min[None, ...]
        else:
            x = x * self.range[:x.shape[0]] + self.min[:x.shape[0]]
            # x = self.range[:x.shape[0]]**x + self.min[:x.shape[0]]

        inputs[label] = x

        return inputs
    