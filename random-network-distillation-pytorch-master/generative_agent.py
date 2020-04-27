import numpy as np
import pytorch_ssim

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, VAEDecoder
from utils import global_grad_norm_


class GenerativeAgent(object):
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
            hidden_dim=512,
        ):
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

        self.decoder = VAEDecoder(z_dim=hidden_dim).to(self.device)
        self.vae_optimizer = optim.Adam(
            self.decoder.parameters(), lr=1e-3,
        )

        self.model = self.model.to(self.device)
        self.policy_optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def train(self):
        self.model.train()
        self.decoder.train()

    def eval(self):
        self.model.eval()
        self.decoder.eval()

    def reconstruct(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        reconstructed = self.decoder(state.unsqueeze(0)).squeeze(0)
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
        embedding = self.decoder.representation(obs)
        reconstructed_embedding = self.decoder.representation(self.decoder(obs))
        intrinsic_reward = (embedding - reconstructed_embedding).pow(2).sum(1) / 2
        # fake_x = self.decoder(obs)
        # intrinsic_reward = nn.MSELoss(reduction='none')(fake_x, obs).sum(
        #     axis=list(range(1, len(fake_x.shape))))

        return intrinsic_reward.detach().cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch,
                    adv_batch, next_obs_batch, old_policy, no_policy=False):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        reconstruction_loss = nn.MSELoss(reduction='mean')
        adv_loss = nn.BCELoss(reduction='mean')

        recon_losses = np.array([])
        kld_losses = np.array([])

        if not no_policy:
            target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
            target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

            with torch.no_grad():
                policy_old_list = torch.stack(old_policy).permute(1, 0, 2
                    ).contiguous().view(-1, self.output_size).to(self.device)

                m_old = Categorical(F.softmax(policy_old_list, dim=-1))
                log_prob_old = m_old.log_prob(y_batch)
                # ------------------------------------------------------------
        else:
            dis_losses = np.array([])
            gen_losses = np.array([])

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                input_x = next_obs_batch[sample_idx]

                # --------------------------------------------------------------------------------
                # for generative curiosity (VAE loss)
                fake_x = self.decoder(input_x)

                d = len(fake_x.shape)
                recon_loss = reconstruction_loss(fake_x, input_x)
                # recon_loss = reconstruction_loss(
                #         fake_x, input_x
                #     ).mean(axis=list(range(1, d)))
                # recon_loss = -1 * pytorch_ssim.ssim(fake_x, input_x, size_average=False)

                # kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).mean()

                # TODO: keep this proportion of experience used for VAE update?
                # Proportion of experience used for VAE update
                # mask = torch.rand(len(recon_loss)).to(self.device)
                # mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                # recon_loss = (recon_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # kld_loss = (kld_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

                recon_losses = np.append(recon_losses, recon_loss.detach().cpu().numpy())
                # kld_losses = np.append(kld_losses, kld_loss.detach().cpu().numpy())

                self.vae_optimizer.zero_grad()
                loss = recon_loss
                loss.backward()
                global_grad_norm_(list(self.decoder.parameters()))
                self.vae_optimizer.step()

                if no_policy:
                    continue
                # ---------------------------------------------------------------------------------

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

                self.policy_optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy
                loss.backward()
                global_grad_norm_(self.model.parameters())
                self.policy_optimizer.step()

        outputs = {
            'data/reconstruction_loss_per_rollout': recon_losses,
        }
        if len(kld_losses) != 0:
            outputs['data/kld_loss_per_rollout'] = kld_losses
        return outputs
