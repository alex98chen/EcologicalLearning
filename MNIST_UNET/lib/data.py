"""
LOAD DATA from file.
"""


##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

##

def select_from_dataset(opt, datasets):
    totoal_num_imgs = len(datasets)
    select_num_imgs = opt.select_num
    select_idx = np.random.choice(totoal_num_imgs, select_num_imgs)
    datasets.data, datasets.targets = datasets.data[select_idx], datasets.targets[select_idx]

    return datasets
def load_data(opt):
    """ Load Data

    Returns:
     dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['mnist']:

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                #transforms.Normalize((0.1307, ), (0.3801, ))
            ]
        )

        # Just use a part of test set since the mnist dataset is highly repeatitive
        datasets = MNIST(root='./data', train=False, download=True, transform=transform)
        datasets = select_from_dataset(opt, datasets)
        dataloader = torch.utils.data.DataLoader(
            dataset=datasets,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=opt.workers,
            drop_last=opt.droplast,
            worker_init_fn=None
        )
        return dataloader

