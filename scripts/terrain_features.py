import numpy as np
import math
import torch.nn as nn
import torch
from torch.nn import functional as F

class Slope(nn.Module):
    def __init__(self):
        super(Slope, self).__init__()
        weight1 = np.array([
            [ -1,  0,  1],
            [ -2,  0,  2],
            [ -1,  0,  1]
        ], dtype=np.float32)
        
        weight2 = np.array([
            [ -1, -2, -1],
            [  0,  0,  0],
            [  1,  2,  1]
        ], dtype=np.float32)

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 30)
        weight2 = weight2 / (8 * 30)
        self.weight1 = nn.Parameter(torch.tensor(weight1))  
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  

    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        ij_slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        ij_slope = torch.arctan(ij_slope) * 180 / math.pi
        return ij_slope

class Aspect(nn.Module):
    def __init__(self):
        super(Aspect, self).__init__()
        weight1 = np.array([
            [ -1,  0,  1],
            [ -2,  0,  2],
            [ -1,  0,  1]
        ], dtype=np.float32)
        
        weight2 = np.array([
            [ -1, -2, -1],
            [  0,  0,  0],
            [  1,  2,  1]
        ], dtype=np.float32)

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8)
        weight2 = weight2 / (8)
        self.weight1 = nn.Parameter(torch.tensor(weight1)) 
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  

    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        aspect_rad = torch.atan2(dy, -dx) 
        aspect = (450.0 - torch.rad2deg(aspect_rad)) % 360.0
        return aspect
    
class Curvature(nn.Module):
    def __init__(self, resolution):
        super(Curvature, self).__init__()
        
        # Kernel para d²z/dx² (central difference)
        kxx = np.array([
            [ 0,  0,  0],
            [ 1, -2,  1],
            [ 0,  0,  0]
        ], dtype=np.float32) / (resolution**2)
        
        # Kernel para d²z/dy²
        kyy = np.array([
            [ 0,  1,  0],
            [ 0, -2,  0],
            [ 0,  1,  0]
        ], dtype=np.float32) / (resolution**2)
        
        # reshape para conv2d: (out_channels, in_channels, H, W)
        kxx = kxx.reshape(1, 1, 3, 3)
        kyy = kyy.reshape(1, 1, 3, 3)
        
        # registramos como parâmetros (podem ficar fixos ou treináveis)
        self.weight_xx = nn.Parameter(torch.from_numpy(kxx), requires_grad=False)
        self.weight_yy = nn.Parameter(torch.from_numpy(kyy), requires_grad=False)
        
        # bias zero (não altera nada)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):

        # segunda derivada em x e em y
        d2x = F.conv2d(x, self.weight_xx, self.bias, padding=1)
        d2y = F.conv2d(x, self.weight_yy, self.bias, padding=1)
        
        # curvatura laplaciana e sinal invertido (convexo >0, côncavo <0)
        curvature = -100.0 * (d2x + d2y)
        return curvature
    
    
class CurvatureTotal(nn.Module):
    def __init__(self, resolution: float, eps: float = 1e-12):
        """
        resolution: tamanho do pixel (mesmas unidades do DEM)
        equal_hessian: se True, impõe t = r (fyy = fxx) como na simplificação do texto
        eps: estabilização numérica
        """
        super().__init__()
        h = float(resolution)
        self.eps = eps

        # Derivadas de 1ª ordem (central difference simples)
        kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32) / h
        ky = kx.t()

        # Derivadas de 2ª ordem (centrais)
        kxx = torch.tensor([[0.,  0.,  0.],
                            [1., -2.,  1.],
                            [0.,  0.,  0.]], dtype=torch.float32) / (h*h)
        kyy = torch.tensor([[0., 1., 0.],
                            [0.,-2., 0.],
                            [0., 1., 0.]], dtype=torch.float32) / (h*h)

        # Mista fxy (central nas diagonais)
        kxy = torch.tensor([[ 1.,  0., -1.],
                            [ 0.,  0.,  0.],
                            [-1.,  0.,  1.]], dtype=torch.float32) / (4.0*h*h)

        # Dar shape (out_c, in_c, H, W)
        def as_weight(k):
            if k.ndim == 2:
                k = k.unsqueeze(0).unsqueeze(0)
            elif k.ndim == 1:
                # kx 1x3 ou ky 3x1 → tornar 2D primeiro
                k = k.unsqueeze(0) if k.shape[0] == 3 else k.unsqueeze(1)
                k = k.unsqueeze(0).unsqueeze(0)
            return nn.Parameter(k, requires_grad=False)

        self.kx  = as_weight(kx)
        self.ky  = as_weight(ky)
        self.kxx = as_weight(kxx)
        self.kyy = as_weight(kyy)
        self.kxy = as_weight(kxy)

    def _conv(self, x, w, padding):
        return F.conv2d(x, w, bias=None, padding=padding)

    def forward(self, z):
        """
        z: tensor (N,1,H,W) com o DEM (SR ou HR)
        retorna: C_total (N,1,H,W)
        """
        # 1ª ordem
        z_x = F.pad(z, (1,1,0,0), mode='replicate')
        p = self._conv(z_x, self.kx, padding=0)   # fx (padding assimétrico horiz. +1)
        # p = F.pad(p, (0,0,1,1), mode='replicate')   # alinhar tamanho (alternativa simples)

        z_y = F.pad(z, (0,0,1,1), mode='replicate')
        q = self._conv(z_y, self.ky, padding=0)   # fy
        # q = F.pad(q, (1,1,0,0), mode='replicate')

        # 2ª ordem
        r = self._conv(z, self.kxx, padding=1)      # fxx
        t = self._conv(z, self.kyy, padding=1)      # fyy
        s = self._conv(z, self.kxy, padding=1)      # fxy

        p2 = p*p
        q2 = q*q
        pq = p*q
        denom_g = p2 + q2 + self.eps
        one_plus_g = 1.0 + p2 + q2

        # Curvaturas (sinais como no artigo)
        Cp = - (p2*r + 2.0*pq*s + q2*t) / (denom_g * (one_plus_g + self.eps).pow(1.5))
        Ct = - (q2*r - 2.0*pq*s + p2*t) / (denom_g * (one_plus_g + self.eps).pow(0.5))

        Ctotal = Cp * Ct
        return Ctotal