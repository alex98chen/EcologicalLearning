# -*- coding: utf-8 -*-
"""GAN_agent.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OVL2m9qHWiTM5vvTtV6A7JByTjlqCgPA
"""

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, NetD, NetG, weights_init
from utils import global_grad_norm_


class GANAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            history_size=4,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            hidden_dim=512):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net, history_size)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.netG = NetG(z_dim=hidden_dim)  # (input_size, z_dim=hidden_dim)
        self.netD = NetD(z_dim=1)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.optimizer_G_policy = optim.Adam(list(self.model.parameters()) + list(self.netG.parameters()), lr=learning_rate)
        #self.optimizer_G = optim.Adam(list(self.netG.parameters()), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(list(self.netD.parameters()), lr=learning_rate, betas=(0.5, 0.999))

        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)

        self.model = self.model.to(self.device)

    def reconstruct(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        reconstructed = self.netG(state*255)[0]
        return reconstructed.detach().cpu().numpy()

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        reconstructed_img, embedding, reconstructed_embedding = self.netG(obs * 255)
        ############# use reconstruction as reward ##################################
        intrinsic_reward = (obs * 255 - reconstructed_img).abs().sum((3, 2, 1))

        ############ use Encode as reward ##################################
        #embedding = embedding.squeeze()
        #reconstructed_embedding = reconstructed_embedding.squeeze()
        #intrinsic_reward = (embedding - reconstructed_embedding).pow(2).sum(1) / 2

        return intrinsic_reward.detach().cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        # reconstruction_loss = nn.MSELoss(reduction='none')]
        l_adv = nn.MSELoss(reduction='none')
        l_con = nn.L1Loss(reduction='none')
        l_enc = nn.MSELoss(reduction='none')
        l_bce = nn.BCELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        # recon_losses = np.array([])
        # kld_losses = np.array([])
        mean_err_g_adv_per_batch = np.array([])
        mean_err_g_con_per_batch = np.array([])
        mean_err_g_enc_per_batch = np.array([])
        mean_err_d_per_batch = np.array([])

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for generative curiosity (GAN loss)
                ############### netG forward ##############################################
                with torch.no_grad():
                    input_next_obs_batch = next_obs_batch[sample_idx] * 255

                gen_next_state, latent_i, latent_o = self.netG(input_next_obs_batch)

                ############## netG backward ##############################################
                self.optimizer_G_policy.zero_grad()

                err_g_adv_per_img = l_adv(self.netD(input_next_obs_batch)[1], self.netD(gen_next_state)[1]).mean(
                    axis=list(range(1, 4)))
                err_g_con_per_img = l_con(input_next_obs_batch, gen_next_state).mean(
                    axis=list(range(1, len(gen_next_state.shape))))
                err_g_enc_per_img = l_enc(latent_i, latent_o).mean(axis=list(range(1, len(latent_i.shape))))

                # TODO: keep this proportion of experience used for VAE update?
                # Proportion of experience used for VAE update
                img_num = len(err_g_con_per_img)
                mask = torch.rand(img_num).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                mean_err_g_adv = (err_g_adv_per_img * mask).sum() / torch.max(mask.sum(),
                                                                              torch.Tensor([1]).to(self.device))
                mean_err_g_con = (err_g_con_per_img * mask).sum() / torch.max(mask.sum(),
                                                                              torch.Tensor([1]).to(self.device))
                mean_err_g_enc = (err_g_enc_per_img * mask).sum() / torch.max(mask.sum(),
                                                                              torch.Tensor([1]).to(self.device))

                # hyperparameter weights:
                w_adv = 1
                w_con = 50
                w_enc = 1

                mean_err_g = mean_err_g_adv * w_adv + \
                             mean_err_g_con * w_con + \
                             mean_err_g_enc * w_enc
                #mean_err_g.backward()
                # global_grad_norm_(list(self.netG.parameters()))
                #self.optimizer_G.step()


                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()


                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + mean_err_g
                loss.backward()
                # global_grad_norm_(list(self.model.parameters())+list(self.vae.parameters())) do we need this step
                global_grad_norm_(list(self.model.parameters()) + list(self.netG.parameters()))  # or just norm policy
                self.optimizer_G_policy.step()

                mean_err_g_adv_per_batch = np.append(mean_err_g_adv_per_batch, mean_err_g_adv.detach().cpu().numpy())
                mean_err_g_con_per_batch = np.append(mean_err_g_con_per_batch, mean_err_g_con.detach().cpu().numpy())
                mean_err_g_enc_per_batch = np.append(mean_err_g_enc_per_batch, mean_err_g_enc.detach().cpu().numpy())

                ############### netD forward ##############################################
                pred_real, feature_real = self.netD(input_next_obs_batch)
                pred_fake, feature_fake = self.netD(gen_next_state.detach())

                ############## netD backward ##############################################
                self.optimizer_D.zero_grad()
                with torch.no_grad():
                    real_label = torch.ones_like(pred_real).to(self.device)
                    fake_label = torch.zeros_like(pred_fake).to(self.device)

                err_d_real_per_img = l_bce(pred_real, real_label)
                err_d_fake_per_img = l_bce(pred_fake, fake_label)

                mean_err_d_real = (err_d_real_per_img * mask).sum() / torch.max(mask.sum(),
                                                                                torch.Tensor([1]).to(self.device))
                mean_err_d_fake = (err_d_fake_per_img * mask).sum() / torch.max(mask.sum(),
                                                                                torch.Tensor([1]).to(self.device))

                mean_err_d = (mean_err_d_real + mean_err_d_fake) / 2
                mean_err_d.backward()
                global_grad_norm_(list(self.netD.parameters()))
                self.optimizer_D.step()

                mean_err_d_per_batch = np.append(mean_err_d_per_batch, mean_err_d.detach().cpu().numpy())

                ############# policy update ###############################################

        return mean_err_g_con_per_batch, mean_err_g_enc_per_batch

    def train_just_gan(self, s_batch, next_obs_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))

        l_adv = nn.MSELoss(reduction='none')
        l_con = nn.L1Loss(reduction='none')
        l_enc = nn.MSELoss(reduction='none')
        l_bce = nn.BCELoss(reduction='none')

        mean_err_g_adv_per_batch = np.array([])
        mean_err_g_con_per_batch = np.array([])
        mean_err_g_enc_per_batch = np.array([])
        mean_err_d_per_batch = np.array([])

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                ############### netG forward ##############################################
                self.netG.train()

                with torch.no_grad():
                    input_next_obs_batch = next_obs_batch[sample_idx] * 255

                gen_next_state, latent_i, latent_o = self.netG(input_next_obs_batch)

                self.optimizer_G_policy.zero_grad()

                err_g_adv_per_img = l_adv(self.netD(input_next_obs_batch)[1], self.netD(gen_next_state)[1]).mean(
                    axis=list(range(1, 4)))
                err_g_con_per_img = l_con(input_next_obs_batch, gen_next_state).mean(
                    axis=list(range(1, len(gen_next_state.shape))))
                err_g_enc_per_img = l_enc(latent_i, latent_o).mean(axis=list(range(1, len(latent_i.shape))))

                # kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1)

                # TODO: keep this proportion of experience used for VAE update?
                # Proportion of experience used for VAE update
                img_num = len(err_g_con_per_img)
                mask = torch.rand(img_num).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                mean_err_g_adv = (err_g_adv_per_img * mask).sum() / torch.max(mask.sum(),
                                                                              torch.Tensor([1]).to(self.device))
                mean_err_g_con = (err_g_con_per_img * mask).sum() / torch.max(mask.sum(),
                                                                              torch.Tensor([1]).to(self.device))
                mean_err_g_enc = (err_g_enc_per_img * mask).sum() / torch.max(mask.sum(),
                                                                              torch.Tensor([1]).to(self.device))

                # hyperparameter weights:
                w_adv = 1
                w_con = 50
                w_enc = 1

                mean_err_g = mean_err_g_adv * w_adv + \
                             mean_err_g_con * w_con + \
                             mean_err_g_enc * w_enc
                mean_err_g.backward()
                global_grad_norm_(list(self.netG.parameters())) # Do we need to global grad norm netG and NetD?
                self.optimizer_G_policy.step()

                mean_err_g_adv_per_batch = np.append(mean_err_g_adv_per_batch, mean_err_g_adv.detach().cpu().numpy())
                mean_err_g_con_per_batch = np.append(mean_err_g_con_per_batch, mean_err_g_con.detach().cpu().numpy())
                mean_err_g_enc_per_batch = np.append(mean_err_g_enc_per_batch, mean_err_g_enc.detach().cpu().numpy())

                ############### netD forward ##############################################
                pred_real, feature_real = self.netD(input_next_obs_batch)
                pred_fake, feature_fake = self.netD(gen_next_state.detach())

                ############## netD backward ##############################################
                self.optimizer_D.zero_grad()
                with torch.no_grad():
                    real_label = torch.ones_like(pred_real).to(self.device)
                    fake_label = torch.zeros_like(pred_fake).to(self.device)

                err_d_real_per_img = l_bce(pred_real, real_label)
                err_d_fake_per_img = l_bce(pred_fake, fake_label)
                mean_err_d_real = (err_d_real_per_img * mask).sum() / torch.max(mask.sum(),
                                                                                torch.Tensor([1]).to(self.device))
                mean_err_d_fake = (err_d_fake_per_img * mask).sum() / torch.max(mask.sum(),
                                                                                torch.Tensor([1]).to(self.device))

                mean_err_d = (mean_err_d_real + mean_err_d_fake) / 2
                mean_err_d.backward()
                global_grad_norm_(list(self.netD.parameters()))
                self.optimizer_D.step()

                mean_err_d_per_batch = np.append(mean_err_d_per_batch, mean_err_d.detach().cpu().numpy())


        return mean_err_g_con_per_batch, mean_err_g_enc_per_batch