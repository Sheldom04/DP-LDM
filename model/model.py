import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from PIL import Image
import torchvision.transforms as transforms
# from .ddpm_trans_modules.style_loss import LossNetwork
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        # self.logvar = torch.full(fill_value=0., size=(opt['model']['beta_schedule']['train']["n_timestep"],),
        #                          device=self.device)
        self.netG = self.set_device(networks.define_G(opt))

        # self.netG_air = self.set_device(networks.define_G(opt))
        # self.dis_water = self.set_device(D2())
        # self.dis_air = self.set_device(D2())
        self.schedule_phase = None
        # set loss and load resume state
        self.set_loss()
        self.loss_func = nn.MSELoss(reduction='sum').to(self.device)
        # self.loss_style = LossNetwork().to(self.device)
        # self.style_loss = VGGPerceptualLoss().to(self.device)
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            print('train')
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                # optim_params = list(filter(lambda p: p.requires_grad, self.netG.denoise_fn.parameters()))
                optim_params = list(self.netG.denoise_fn.parameters())
                # for name, param in self.netG.named_parameters():
                #     print(name)
                # optim_params.append(self.logvar)
                # optim_params = list(self.netG.parameters()) + list(self.netG_air.parameters())
                # optim_params_dis = list(self.dis_air.parameters()) + list(self.dis_water.parameters())
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()

        # self.print_network()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def feed_data(self, data):
        self.data = self.set_device(data)
        with torch.no_grad():
            # 训练图片 一个SR(退化) 一个HR(清晰)
            # self.data['style'] = self.data['HR']
            self.data['SR'] = self.data['SR']
            self.data['HR'] = self.data['HR']
            # self.data['HR'] = self.vae.encode(self.data['HR']).latent_dist.sample().mul_(0.18215)

    def feed_data2(self, data):
        self.data = self.set_device(data)
        with torch.no_grad():
            self.data['Origin'] = self.data['Origin']
            self.data['SR'] = self.data['SR']
            self.data['HR'] = self.data['HR']

    def optimize_parameters(self, flag=None):
        # need to average in multi-gpu
        if flag is None:
            self.optG.zero_grad()  # 先梯度清零
            l_pix = self.netG(self.data, flag=None)  # 再计算loss

            l_pix.backward()   # 计算梯度
            self.optG.step()   # 反向传播开始优化
            # print('single mse:', l_pix.item())
            # set log
            self.log_dict['l_pix'] = l_pix.item()

    def optimize_parameters2(self):  # 多GPU使用这个
        # need to average in multi-gpu
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, cand=None, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data, continous, cand=cand)
        return self.SR
                # self.SR = self.vae.decode(self.SR / 0.18215).sample

        # self.netG.train()

    def fine_test(self, cond, x_start, cand=None, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous)
            else:
                self.SR = self.netG.fine_p_sample_loop(
                    x=cond, x_start=x_start, continous=continous, cand=cand)
            return self.SR
                # self.SR = self.vae.decode(self.SR / 0.18215).sample


    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            # out_dict['INF'] = self.data['SR'].detach().float().cpu()
            # self.data['HR'] = self.vae.decode(self.data['HR'] / 0.18215).sample
            out_dict['HR'] = self.data['Origin'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG   # netG中加载权重的时候包含vae的权重导致权重被覆盖了
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            check_point = torch.load(gen_path)
            specific_module_state_dict = {k: v for k, v in check_point.items() if
                                          'denoise_fn' in k}
            network.load_state_dict(specific_module_state_dict, strict=False)

            # load_part_of_model(network, gen_path, s=(not self.opt['model']['finetune_norm']))
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
