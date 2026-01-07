#!/usr/bin/python
import argparse
import math
import model as Model
from torchvision.transforms.functional import hflip
from torchvision.utils import make_grid
from data.util import totensor
from my_ldm_concat_cond.my_vae import AutoEncoderKL
from my_ldm_concat_cond.taming.modules.losses.vqperceptual import *
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import random
import time
import core.logger as Logger


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, reconstructions):
        # 重构损失
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        loss = rec_loss
        log = {}
        return loss, log


# 每个worker随机数种子定义
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


# 训练数据集定义
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    # assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


class MyDataset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.img_path = get_paths_from_images(self.dataroot[0])  # 训练得到所有图片的地址
        self.img_path_c = get_paths_from_images(self.dataroot[1])  # 训练得到所有图片的地址
        self.data_len = len(self.img_path)
        print(self.data_len)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
        ])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index: int):
        img = Image.open(self.img_path[index]).convert("RGB")
        img_c = Image.open(self.img_path_c[index]).convert("RGB")
        # img = self.transform(img)
        # img_c = self.transform(img_c)
        [img, img_c] = transform_augment(
            [img, img_c], split='val', min_max=(-1, 1))
        return img, img_c


def train_dataloader(dataroot):
    init_fn = worker_init_fn
    train_data = MyDataset(dataroot)
    return DataLoader(train_data, batch_size=1,
                      num_workers=12, shuffle=False,
                      worker_init_fn=init_fn, pin_memory=True)


def val_dataloader(dataroot):
    init_fn = worker_init_fn
    train_data = MyDataset(dataroot)
    return DataLoader(train_data, batch_size=1,
                      num_workers=12, shuffle=False,
                      worker_init_fn=init_fn, pin_memory=True)


# 随机数种子
def seed_everything(seed: int):
    """
    Set the seed for random number generation in Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)  # Python random number generator
    np.random.seed(seed)  # NumPy random number generator
    torch.manual_seed(seed)  # PyTorch random number generator for CPU
    torch.cuda.manual_seed(seed)  # PyTorch random number generator for GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # To ensure determinism in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def vae_train(dataroot, lr, epochs, opt, resume=None):
    # 训练步骤方法，定义了训练过程中的具体操作
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # GPU选择
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32
    # 模型初始化
    loss = LPIPSWithDiscriminator()
    vae = AutoEncoderKL()
    diffusion = Model.create_model(opt)
    vae.to(default_device, dtype=default_type)
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    if resume is not None:
        vae.load_state_dict(torch.load(resume[0]))  # 加载预训练权重

    # 优化器
    opt_ae = torch.optim.Adam(
        list(vae.decoder.parameters()) +
        list(vae.post_quant_conv.parameters()),
        lr=lr, betas=(0.5, 0.9))
    # 初始化数据
    DataLoader = train_dataloader(dataroot)
    # 训练模式
    vae.train()
    # 开始训练
    vae_save_path = 'fine_pt/vae/vae_epoch_{}.pth'
    seed_everything(2025)
    print("start training")
    start_t = time.time()
    t = 0
    for epoch in range(epochs):
        for i, (inputs, cond) in enumerate(DataLoader):
            inputs = inputs.to(default_device, dtype=default_type)
            cond = cond.to(default_device, dtype=default_type)  # 受损图像
            posterior = vae.encode(cond)
            diff_cond = posterior.sample()
            z0 = diffusion.fine_test(cond=diff_cond, x_start=cond)
            reconstructions = vae.decode(z0, cond)
            opt_ae.zero_grad()
            aeloss, log_dict_ae = loss(inputs, reconstructions)
            aeloss.backward()
            opt_ae.step()
            if (i + 1) % 50 == 0:
                t = time.time() - start_t
                start_t = time.time()
                print("Epoch[{}/{}], Step [{}/{}], aeloss: {:.4f}, cost time:{:.2f}"
                      .format(epoch + 1, epochs, i + 1, len(DataLoader), aeloss.item(), t))
        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), vae_save_path.format(epoch + 1))
            print(f'Model saved at epoch {epoch + 1}')
        print(f"epoch:{epoch + 1}/{epochs}", end=' ')
        print("aeloss:", aeloss.item(), end=' ')

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def val(dataroot, pt_file):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 加载数据
    val_data = val_dataloader(dataroot=dataroot)
    # 加载模型
    vae = AutoEncoderKL()
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32
    vae.to(default_device)
    # 验证模式
    vae.eval()
    vae.load_state_dict(torch.load(pt_file))
    idx = 0
    result_path = "results"
    with torch.no_grad():
        for i, (inputs, cond) in enumerate(val_data):
            idx += 1
            inputs = inputs.to(default_device)
            cond = cond.to(default_device)
            xrec, posterior = vae(inputs, cond)
            xrec = xrec.cpu()
            image = tensor2img(tensor=xrec, min_max=(-1, 1))
            # image = xrec.squeeze(0).permute(1, 2, 0)
            # image = (image.clamp(0, 1).detach().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(f"{result_path}/{idx:06d}.jpg")
            print(idx)


if __name__ == '__main__':
    dataroot = [r"C:\Users\liuxd\Desktop\DiFF_UIE\dataset\UIEB_256\hr_256",
                r"C:\Users\liuxd\Desktop\DiFF_UIE\dataset\UIEB_256\sr_16_256"]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    lr = 4.5e-6
    epochs = 100
    # val(dataroot, 'my_pt/best/vae_epoch_20-1-1.pth')
    vae_train(dataroot=dataroot, lr=lr, epochs=epochs, opt=opt,
              resume=[r"my_ldm_concat_cond/my_pt/vae/vae_epoch_60_UIEB.pth",
                      r"my_ldm_concat_cond/my_pt/disc/disc_epoch_60_UIEB.pth"])

