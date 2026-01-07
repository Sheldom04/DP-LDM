import math
from torchvision.transforms.functional import hflip
from torchvision.utils import make_grid
from data.util import totensor
import sys

sys.path.append("..")
from my_vae import AutoEncoderKL
from taming.modules.losses.vqperceptual import *
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import random
import time
import core.metrics as Metrics


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start=0, logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.1,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.if_g = False
        self.kl_weight = kl_weight
        # self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().to(torch.device('cuda')).eval()
        self.ssim_loss = SSIMLoss().to(torch.device('cuda')).eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # 判别器
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                nll_weights=None):
        # 重构损失
        L1 = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = L1
        ssim_loss = self.ssim_loss(inputs, reconstructions)
        rec_loss = rec_loss + ssim_loss
        if self.perceptual_weight > 0:  # 是否计算感知损失
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        # 负对数似然损失
        nll_loss = rec_loss
        weighted_nll_loss = nll_loss
        if nll_weights is not None:
            weighted_nll_loss = nll_weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log = None
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            else:
                d_weight = torch.tensor(0.0)
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            return loss, {"L1": torch.sum(L1) / L1.shape[0],
                          "L_perc": torch.sum(p_loss) * L1.shape[1] * L1.shape[2] * L1.shape[3],
                          "L_ssim": torch.sum(ssim_loss) * L1.shape[1] * L1.shape[2] * L1.shape[3],
                          "L_g": d_weight * disc_factor * g_loss}

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            # disc_factor 的权重系数 若没到阈值则为0  避免判别器过于强大导致模式崩溃
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if self.if_g:
                disc_factor = disc_factor
            else:
                disc_factor = 0.
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


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
    def __init__(self, dataroot, phase):
        self.dataroot = dataroot
        self.img_path = get_paths_from_images(self.dataroot[0])
        self.img_path_c = get_paths_from_images(self.dataroot[1])
        self.data_len = len(self.img_path)
        print(self.data_len)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
        ])
        self.phase = phase

    def __len__(self):
        if self.phase == 'train':
            return self.data_len
        elif self.phase == 'val':
            return 3

    def __getitem__(self, index: int):
        img = Image.open(self.img_path[index]).convert("RGB")
        img_c = Image.open(self.img_path_c[index]).convert("RGB")
        [img, img_c] = transform_augment(
            [img, img_c], split='val', min_max=(-1, 1))
        return img, img_c


def train_dataloader(dataroot):
    init_fn = worker_init_fn
    train_data = MyDataset(dataroot, 'train')
    return DataLoader(train_data, batch_size=2,
                      num_workers=12, shuffle=False,
                      worker_init_fn=init_fn, pin_memory=True)


def val_dataloader(dataroot):
    init_fn = worker_init_fn
    train_data = MyDataset(dataroot, 'val')
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


def vae_train(dataroot_train, dataroot_val, lr, epochs, resume=None):
    # 训练步骤方法，定义了训练过程中的具体操作
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # GPU选择
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32
    # 模型初始化
    loss = LPIPSWithDiscriminator()
    vae = AutoEncoderKL()
    vae.to(default_device, dtype=default_type)
    loss.discriminator.to(default_device, dtype=default_type)
    if resume is not None:
        vae.load_state_dict(torch.load(resume[0]))  # 加载预训练权重
        loss.discriminator.load_state_dict(torch.load(resume[1]))
    global_step = 0
    # 优化器
    opt_ae = torch.optim.Adam(list(vae.encoder.parameters()) +
                              list(vae.decoder.parameters()) +
                              list(vae.quant_conv.parameters()) +
                              list(vae.post_quant_conv.parameters()),
                              lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(loss.discriminator.parameters(),
                                lr=lr, betas=(0.5, 0.9))
    # 初始化数据
    Train_dataLoader = train_dataloader(dataroot_train)
    val_dataLoader = val_dataloader(dataroot_val)
    # 训练模式
    vae.train()
    loss.discriminator.train()
    # 创建保存权重文件夹
    os.makedirs("my_pt/vae/", exist_ok=True)
    os.makedirs("my_pt/disc/", exist_ok=True)
    # 权重文件保存路径
    vae_save_path = 'my_pt/vae/vae_epoch_{}.pth'
    disc_save_path = 'my_pt/disc/disc_epoch_{}.pth'
    seed_everything(2025)
    print("开始训练")
    start_t = time.time()
    t = 0
    for epoch in range(epochs):
        for i, (inputs, cond) in enumerate(Train_dataLoader):
            inputs = inputs.to(default_device, dtype=default_type)
            cond = cond.to(default_device, dtype=default_type)
            reconstructions, posterior = vae(inputs, cond)
            opt_ae.zero_grad()
            aeloss, loss_dict = loss(inputs, reconstructions, posterior, optimizer_idx=0, global_step=global_step,
                                     last_layer=vae.decoder.conv_out.weight, split="train")
            aeloss.backward()
            opt_ae.step()
            opt_disc.zero_grad()
            discloss, _ = loss(inputs, reconstructions, posterior, optimizer_idx=1,
                               global_step=global_step,
                               last_layer=vae.decoder.conv_out.weight, split="train")
            discloss.backward()
            opt_disc.step()
            global_step += 1
            if (i + 1) % 10 == 0:
                t = time.time() - start_t
                start_t = time.time()

                print("Epoch[{}/{}], Step [{}/{}], aeloss: {:.4f}, "
                      "L1_loss:{:.4f}, "
                      "L_perc:{:.4f}, "
                      "L_ssim:{:.4f}, "
                      "L_g:{:.4f}, "
                      "cost time:{:.2f}"
                      .format(epoch + 1, epochs, i + 1, len(Train_dataLoader), aeloss.item(),
                              loss_dict["L1"].item(),
                              loss_dict["L_perc"].item(),
                              loss_dict["L_ssim"].item(),
                              loss_dict["L_g"].item(),
                              t))

        if (epoch + 1) % 1 == 0:
            idx = 0
            avg_psnr = 0
            val_results_path = "val_results"
            os.makedirs(val_results_path, exist_ok=True)  # 创建文件夹
            for _, (inputs, cond) in enumerate(val_dataLoader):
                idx += 1
                inputs = inputs.to(default_device)
                cond = cond.to(default_device)
                xrec, posterior = vae(inputs, cond)
                xrec = tensor2img(xrec.detach().float().cpu(), min_max=(-1, 1))
                inputs = tensor2img(inputs.detach().float().cpu(), min_max=(-1, 1))
                image = Image.fromarray(xrec)
                image.save(f"{val_results_path}/{epoch + 1}_{idx}_xrec.jpg")
                avg_psnr += Metrics.calculate_psnr(
                    xrec, inputs)
            avg_psnr = avg_psnr / idx
            print("PSNR:", avg_psnr)
            if (epoch + 1) % 10 == 0:
                torch.save(vae.state_dict(), vae_save_path.format(epoch + 1))
                torch.save(loss.discriminator.state_dict(), disc_save_path.format(epoch + 1))
                print(f'Model saved at epoch {epoch + 1}:psnr={avg_psnr}')


def val(dataroot, pt_file):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 加载数据
    val_data = val_dataloader(dataroot=dataroot)
    # 加载模型
    vae = AutoEncoderKL()
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae.to(default_device)
    # 验证模式
    vae.eval()
    vae.load_state_dict(torch.load(pt_file))
    idx = 0
    result_path = "results"
    os.makedirs(result_path, exist_ok=True)
    with torch.no_grad():
        for i, (inputs, cond) in enumerate(val_data):
            idx += 1
            inputs = inputs.to(default_device)
            cond = cond.to(default_device)
            posterior = vae.encode(inputs)
            xrec = vae.decode(posterior.sample(), cond)
            xrec = xrec.cpu()
            image = tensor2img(tensor=xrec, min_max=(-1, 1))
            image = Image.fromarray(image)
            image.save(f"{result_path}/{idx:06d}.jpg")
            print(idx)


if __name__ == '__main__':
    dataroot_train = [r"C:\Users\liuxd\Desktop\DiFF_UIE\dataset\LSUI_256\hr_256",  # train_GT
                      r"C:\Users\liuxd\Desktop\DiFF_UIE\dataset\LSUI_256\sr_16_256"]  # train_Degraded
    dataroot_val = [r"C:\Users\liuxd\Desktop\DiFF_UIE\dataset\LSUI_256_test\hr_256",  # val_GT
                    r"C:\Users\liuxd\Desktop\DiFF_UIE\dataset\LSUI_256_test\sr_16_256"]  # val_Degraded
    lr = 4.5e-6
    epochs = 100
    # val(dataroot_val, 'my_pt/best/vae_epoch_49_original.pth')
    vae_train(dataroot_train=dataroot_train, dataroot_val=dataroot_val, lr=lr, epochs=epochs, resume=None)
