# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12Yy4VPmeWyWuglJmsgtPEekq4920triU
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from collections import OrderedDict 

class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False, history_size=4):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=history_size,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                256),
            nn.ReLU(),
            linear(
                256,
                448),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(448, 448),
            nn.ReLU()
        )

        self.critic_ext = linear(448, 1)
        self.critic_int = linear(448, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return policy, value_ext, value_int


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class UnFlatten(nn.Module):
    def forward(self, input, shape=(64, 7, 7)):
        return input.view(input.size(0), *shape)

        
class VAE(nn.Module):
    def __init__(self, input_size, z_dim=128):
        super(VAE, self).__init__()

        self.input_size = input_size

        feature_output = 7 * 7 * 64
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(feature_output, z_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(feature_output, z_dim),
        )

        self.fc3 = nn.Linear(z_dim, feature_output)

        # TODO: write a different decoder???
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=8,
                stride=4),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x, return_hidden=False):
        h = self.encoder(x)
        h_z, mu, logvar = self.bottleneck(h)
        z = self.fc3(h_z)
        if return_hidden:
            return self.decoder(z), mu, logvar, h_z
        else:
            return self.decoder(z), mu, logvar


class Predictor(nn.Module):
    def __init__(self, input_size, z_dim=128):
        super(Predictor, self).__init__()

        self.input_size = input_size

        feature_output = 7 * 7 * 64

        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, x):
        return self.predictor(x)


class Encoder(nn.Module):
  def __init__(self, z_dim=128):
    super(Encoder, self).__init__()
    #self.input_size = input_size

    # input is 1 * 84 * 84
    main = nn.Sequential(OrderedDict([
        ('paramid-conv0-1->64', nn.Conv2d(1, 64, 4, 2, 1, bias=False)),
        ('paramid-relu0-64', nn.LeakyReLU(0.2, inplace=True)), # output is 64 * 42 * 42

        ('paramid-conv1-64->128', nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
        ('paramid-batchnorm1-128', nn.BatchNorm2d(128)),
        ('paramid-relu1-128', nn.LeakyReLU(0.2, inplace=True)), # output is 128 * 21 * 21

        ('paramid-conv2-128->256', nn.Conv2d(128, 256, 9, 2, 1, bias=False)),
        ('paramid-batchnorm2-256', nn.BatchNorm2d(256)),
        ('paramid-relu2-256', nn.LeakyReLU(0.2, inplace=True)), # output is 256 * 8 * 8

        ('paramid-conv3-256->512', nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
        ('paramid-batchnorm3-512', nn.BatchNorm2d(512)),
        ('paramid-relu3-512', nn.LeakyReLU(0.2, inplace=True)), # output is 512 * 4 * 4

        ('paramid-conv4-512->{0}'.format(z_dim), nn.Conv2d(512, z_dim, 4, 1, 0, bias=False)),
        ('paramid-batchnorm4-{0}'.format(z_dim), nn.BatchNorm2d(z_dim)),
        ('paramid_relu4-{0}'.format(z_dim), nn.LeakyReLU(0.2, inplace=True)), # output is 128 * 1 * 1

    ]))

    self.main = main

  def forward(self, input):
    output = self.main(input) 


class Decoder(nn.Module):
  def __init__(self, z_dim):
    super(Decoder, self).__init__()

    # input is z_dim * 1 * 1
    main = nn.Sequential(OrderedDict([
        ('paramid-conv4-{0}->512'.format(z_dim), nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False)),
        ('paramid-batchnorm4-512', nn.BatchNorm2d(512)),
        ('paramid-relu4', nn.ReLU(True)),

        ('paramid-conv3-512->256', nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
        ('paramid-batchnorm3-256', nn.BatchNorm2d(256)),
        ('paramid-relu3', nn.ReLU(True)),

        ('paramid-conv2-256->128', nn.ConvTranspose2d(256, 128, 9, 2, 1, bias=False)),
        ('paramid-batchnorm2-128', nn.BatchNorm2d(128)),
        ('paramid-relu2-128', nn.ReLU(True)),

        ('paramid-conv1-128->64', nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)),
        ('paramid-batchnorm1-64', nn.BatchNorm2d(64)),
        ('paramid-relu1-64', nn.ReLU(True)),

        ('paramid-conv0-64-1', nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False)),
        ('paramid-relu0-64', nn.ReLU(True)),

        ('final-sigmoid-1', nn.Sigmoid())
    ]))
    self.main = main

  def forward(self, input):
    output = self.main(input)
    return output 


class NetG(nn.Module):
  def __init__(self, z_dim=128):
    super(NetG, self).__init__()
    self.encoder1 = Encoder(z_dim)
    self.decoder = Decoder(z_dim)
    self.encoder2 = Encoder(z_dim)

  def forward(self, real):
    latent_i = self.encoder1(real)
    fake = self.decoder(latent_i)
    latent_o = self.encoder2(fake)
    return fake, latent_i, latent_o


class NetD(nn.Module):
  def __init__(self,z_dim=1):
    super(NetD, self).__init__()
    model = Encoder(z_dim)
    layers = list(model.main.children())

    self.features = nn.Sequential(*layers[:-1])
    self.classifier = nn.Sequential(layers[-1])
    self.classifier.add_module('Sigmoid', nn.Sigmoid())

  def forward(self,img):
    features = self.features(img)
    classifier = self.classifier(features)
    classifier = classifier.view(-1, 1).squeeze(1)

    return classifier, features

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)