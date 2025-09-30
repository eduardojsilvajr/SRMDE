import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import tqdm

root_path = r'C:\Users\Eduardo JR\Fast'
# Caminhos
hr_dir = 'Flickr2k/HR'
lr_dir = 'Flickr2k/LR_bicubic/X2'
train_hr = 'Flickr2k/train/HR'
train_lr = 'Flickr2k/train/LR'
valid_hr = 'Flickr2k/valid/HR'
valid_lr = 'Flickr2k/valid/LR'

# Cria pastas se não existirem
for path in [train_hr, train_lr, valid_hr, valid_lr]:
    os.makedirs(os.path.join(root_path,path), exist_ok=True)

# Lista de imagens
all_images = sorted([f for f in os.listdir(os.path.join(root_path,hr_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
train_imgs, valid_imgs = train_test_split(all_images, test_size=0.2)

# Função de downscale
def generate_lr_bicubic(img: Image.Image, scale: int = 2) -> Image.Image:
    w, h = img.size
    return img.resize((w // scale, h // scale), resample=Image.BICUBIC)

# Processa os conjuntos
def process_set(img_list, hr_out, lr_out):
    for img_name in img_list:
        img_path = os.path.join(os.path.join(root_path,hr_dir), img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Gera e salva HR
        img.save(os.path.join(hr_out, img_name))

        # Gera e salva LR
        lr_img = generate_lr_bicubic(img, scale=2)
        lr_img.save(os.path.join(lr_out, img_name))

# Executa
process_set(train_imgs, os.path.join(root_path,train_hr), os.path.join(root_path,train_lr))
process_set(valid_imgs, os.path.join(root_path,valid_hr), os.path.join(root_path,valid_lr))
