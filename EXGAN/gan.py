import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models import *


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")

parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

parser.add_argument("--shots", type=int, default=10, help=" few shot numbers for XAI")
parser.add_argument('--XAI_method', nargs='?', default = IntegratedGradients, help = "IntegratedGradients, DeepLift, ...")
parser.add_argument("--alpha", type=float, default=0.5, help="balancing reals and samples")
parser.add_argument("--MNISTlabel", type=int, default=8, help="balancing reals and samples")
args = parser.parse_args()

print(args)


xaigan = XAIGAN(args)
D, samples, reals = xaigan.trainGAN()
common_mask= xaigan.common_masking(D, samples, reals)


#앞으로 구현해야 할 것
train_dualGAN() # D가 두개로 재학습 시키는 모듈, few-shot 데이터들 한정.