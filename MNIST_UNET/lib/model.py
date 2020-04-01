"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
np.set_printoptions(threshold=1e6)

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
#from lib.visualizer import Visualizer
#from lib.loss import l2_loss
#from lib.evaluate import evaluate


class MNIST_UNET(nn.Module):
    def __init__(self, opt, dataloader):
        super(MNIST_UNET, self).__init__()

        self.opt = opt
        #self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.total_steps = len(dataloader)
        self.device = torch.device('cuda:0' if self.opt.device != 'cpu' else 'cpu')

        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        weights_init(self.netg)
        weights_init(self.netd)

        self.l_adv = self.l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = self.l2_loss
        self.l_bce = nn.BCELoss()

        # Initialize input tensors.
        self.input_imgs = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        #self.label = torch.empty(size=(self.opt.batchsize, ), dtype=torch.float32, device=self.device)
        #self.gt = torch.empty(size=(self.opt.batchsize, ), dtype=torch.long, device=self.device)

        self.real_label = torch.ones(size=(self.opt.batchsize, ), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize, ), dtype=torch.float32, device=self.device)

    def train(self):
        """

       Train the model.
        """
        ##
        # TRAIN
        self.netd.train()
        self.netg.train()
        optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        # Train for niter epochs.
        print(">> Train model %s steps." %self.total_steps)
        #if self.opt.resume != '':
        #    netG_weights_path = os.path.join(self.opt.resume, 'netG.pth')
        #    netD_weights_path = os.path.join(self.opt.resume, 'netD.pth')
        #    if os.path.exists(netG_weights_path):



        self.step_reward = []
        for step in tqdm(range(self.total_steps)):
            # Train for one step
            step_iter = 0
            loss_d_step = 0
            loss_g_step = 0
            loss_g_adv_step = 0
            loss_g_con_step = 0
            loss_g_enc_step = 0


            current_dataloader = []
            next_batch = None

            for count, (input_imgs, gt) in enumerate(self.dataloader):
                if count < step:
                    current_dataloader.append(input_imgs)
                elif count == step:
                    next_batch = input_imgs
            self.set_input(next_batch)
            intrinsic_loss = self.calculate_intrinsic_loss()
            self.save_images(self.input_imgs, self.fake_imgs, step)
            print('step: %s, reward: %s.'%(step, intrinsic_loss))
            self.step_reward.append(intrinsic_loss)

            current_dataloader.append(next_batch)

            netG_weights_path = os.path.join(self.opt.resume, 'netG.pth')
            netD_weights_path = os.path.join(self.opt.resume, 'netD.pth')
            if os.path.exists(netG_weights_path) and os.path.exists(netD_weights_path):
                self.netg.load_state_dict(torch.load(netG_weights_path)['state_dict'])
                self.netd.load_state_dict(torch.load(netD_weights_path)['state_dict'])


            for epoch in range(self.opt.niter):
                epoch_iter = 0
                loss_d_epoch = 0
                loss_g_epoch = 0
                loss_g_adv_epoch = 0
                loss_g_con_epoch = 0
                loss_g_enc_epoch = 0

                for input_imgs in current_dataloader:

                    self.set_input(input_imgs)
                    self.fake_imgs, self.latent_i, self.latent_o = self.netg(self.input_imgs)
                    self.pred_real, self.feat_real = self.netd(self.input_imgs)
                    self.pred_fake, self.feat_fake = self.netd(self.fake_imgs.detach())

                    # Update generator
                    optimizer_g.zero_grad()
                    self.err_g_adv = self.l_adv(self.netd(self.input_imgs)[0], self.netd(self.fake_imgs)[0])
                    self.err_g_con = self.l_con(self.input_imgs, self.fake_imgs)
                    self.err_g_enc = self.l_enc(self.latent_i, self.latent_o)
                    self.err_g = self.err_g_adv * self.opt.w_adv +\
                                self.err_g_con * self.opt.w_con +\
                                self.err_g_enc *self.opt.w_enc
                    self.err_g.backward()
                    optimizer_g.step()

                    # Update discriminator
                    optimizer_d.zero_grad()
                    self.err_d_real = self.l_bce(self.pred_real, self.real_label)
                    self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)
                    self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
                    self.err_d.backward()
                    optimizer_d.step()

                    if self.err_d.item() < 1e-5:
                        weights_init(self.netd)
                        print('Reloading netd')

                self.save_weights()
        self.draw_reward()




    def calculate_intrinsic_loss(self):
        """

        :param input_imgs: Current seen batch images
        :return: The calculated intrinsic rewards
        """
        with torch.no_grad():
            print(">>Geting current intrinsic reward.")
            netG_weights_path = os.path.join(self.opt.resume, 'netG.pth')
            if os.path.exists(netG_weights_path) :
                pretrained_dict = torch.load(netG_weights_path)['state_dict']
                self.netg.load_state_dict(pretrained_dict)
            else:
                weights_init(self.netg)

            # Creat big error tensor for the current seen batch images.
            self.fake_imgs, self.latent_i, self.latent_o = self.netg(self.input_imgs)
            if self.opt.use_con_reward:
                con_reward = self.l_con(self.fake_imgs, self.input_imgs)
            else:
                con_reward = 0
            enc_reward = self.l_enc(self.latent_i, self.latent_o)
            total_reward = enc_reward + con_reward
            return total_reward.to('cpu').numpy().item()

    def l2_loss(self, input, target, size_average=True):
        if size_average:
            return torch.mean(torch.pow((input - target), 2))
        else:
            return torch.pow((input - target), 2)

    def set_input(self, input_imgs):
        # Set input and ground truth
        with torch.no_grad():
            self.input_imgs.resize_(input_imgs.size()).copy_(input_imgs)
            #self.gt.resize(gt.size()).copy_(gt)
            #self.label.resize_(gt.size())

    def save_weights(self):
        weight_dir = os.path.join(self.opt.resume, 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'state_dict': self.netg.state_dict()}, os.path.join(weight_dir, 'netG.pth'))
        torch.save({'state_dict': self.netd.state_dict()}, os.path.join(weight_dir, 'netD.pth'))

    def save_images(self, real, fake, step):
        N, C, W, H = real.shape
        stitch_images = np.zeros((C, W*N, 3*H))
        image_dir = os.path.join(self.opt.resume, 'images')
        if not os.path.exists(image_dir): os.makedirs(image_dir)
        for i in range(N):
            real_img = (real[i, :, :, :] * 255).to('cpu').numpy().astype(np.int)
            fake_img = (fake[i, :, :, :] * 255).to('cpu').numpy().astype(np.int)
            mask_img = np.abs(real_img - fake_img).astype(np.uint8)
            print(np.min(real_img))
            print(np.max(real_img))
            stitch_images[:, W*i: W*i+W, :H] = real_img.astype(np.uint8)
            stitch_images[:, W*i: W*i+W, H:2*H] = fake_img.astype(np.uint8)
            print(np.min(fake_img))
            print(np.max(fake_img))
            stitch_images[:, W*i: W*i+W, 2*H:] = mask_img
        stitch_images = stitch_images.squeeze(0)
        #stitch_images = stitch_images.numpy()
        plt.imsave(os.path.join(image_dir, '%s.png'%(step+1)), stitch_images, cmap='gray')

    def draw_reward(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.total_steps + 1), self.step_reward)
        plt.xlabel("steps")
        plt.ylabel("reward")
        plt.savefig(os.path.join(self.opt.resume, 'images', 'rewards.png'))
        plt.show()
