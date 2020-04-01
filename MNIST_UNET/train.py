"""
TRAIN MNIST_Unet

. Example: Run the following command from the terminal.
    run train.py                             \
        --model mnist_unet                        \
        --dataset mnist \
        --batchsize 32                          \
        --isize 32                        \
        --nz 100                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import MNIST_UNET

##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = MNIST_UNET(opt, dataloader)
    ##
    # TRAIN MODEL
    model.train()


if __name__ == '__main__':
    train()
