import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from captum.attr import visualization as viz
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    LRP,
)

def get_indices(dataset,class_name):
            indices =  []
            for i in range(len(dataset.targets)):
                if dataset.targets[i] == class_name:
                    indices.append(i)
            return indices
        
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def getmasking(tensor, version = 'thresh', thresh = 0.001):
    '''
    top-k version should be implemented later.

    '''
    if version == 'thresh':
        return (tensor > thresh).float()
    else:
        return (tensor>tensor.mean()).float() 


    
class XAIGAN:
    
    def __init__(self, args):
        
        self.args = args
        self.img_shape = (args.channels, args.img_size, args.img_size)
        self.cuda = True if torch.cuda.is_available() else False
        # Configure data loader

        os.makedirs("../../data/mnist", exist_ok=True)        
        dataset = datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            )
        idx = get_indices(dataset, self.args.MNISTlabel) 
        self.dataloader = DataLoader(dataset,batch_size=self.args.batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
  
        self.alpha = args.alpha
        self.beta = args.beta
        
        
    def train_GAN(self):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        # Initialize generator and discriminator
        generator = XAIGAN.Generator(self.args, self.img_shape)
        discriminator = XAIGAN.Discriminator(self.args, self.img_shape)
        
        if self.cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        for epoch in range(self.args.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
        
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
        
                # -----------------
                #  Train Generator
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim))))
        
                # Generate a batch of images
                gen_imgs = generator(z)
        
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
                g_loss.backward()
                optimizer_G.step()
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                optimizer_D.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
        
                d_loss.backward()
                optimizer_D.step()
        
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.args.n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                )
        
                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.args.sample_interval == 0:
                    save_image(gen_imgs.data[:16], "images/%d.png" % batches_done, nrow=4, normalize=True)
                
        return discriminator, gen_imgs.data[:self.args.shots], real_imgs.data[:self.args.shots]
    
    
    
    def train_dualGAN(self, common_mask):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        # Initialize generator and discriminator
        generator = XAIGAN.Generator(self.args, self.img_shape)
        discriminator = XAIGAN.Discriminator(self.args, self.img_shape)
        discriminator_mask = XAIGAN.Discriminator(self.args, self.img_shape)
        
        if self.cuda:
            generator.cuda()
            discriminator.cuda()
            discriminator_mask.cuda()
            adversarial_loss.cuda()
            common_mask = common_mask.cuda()
            
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_Dm = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        for epoch in range(self.args.n_epochs_dual):
            for i, (imgs, _) in enumerate(self.dataloader):
                
                if i > self.args.shots: # for controlling few-shit condition
                    break
        
                # Adversarial ground truths for D
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                real_imgs_mask = real_imgs * common_mask
        
                # -----------------
                #  Train Generator
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim))))
        
                # Generate a batch of images
                gen_imgs = generator(z)
                gen_imgs_mask = gen_imgs * common_mask
        
                # Loss measures generator's ability to fool the discriminator
                g_loss = (1-self.beta) * adversarial_loss(discriminator(gen_imgs), valid) \
                    + self.beta * adversarial_loss(discriminator_mask(gen_imgs_mask), valid)
        
                g_loss.backward()
                optimizer_G.step()
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                optimizer_D.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (1-self.beta) * (real_loss + fake_loss) / 2
                
        
                d_loss.backward()
                optimizer_D.step()
                
                # ---------------------
                #  Train Mask_Discriminator
                # ---------------------
        
                optimizer_Dm.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss_mask = adversarial_loss(discriminator_mask(real_imgs_mask), valid)
                fake_loss_mask = adversarial_loss(discriminator_mask(gen_imgs_mask.detach()), fake)
                dm_loss = self.beta * (real_loss_mask + fake_loss_mask) / 2
                
        
                dm_loss.backward()
                optimizer_Dm.step()
        
                print(
                    "[Epoch %d/%d] [D loss: %f] [Dm loss: %f] [G loss: %f]"
                    % (epoch, self.args.n_epochs_dual, d_loss.item(), dm_loss.item(), g_loss.item())
                )
        
                batches_done = epoch * len(self.dataloader) + i
                if epoch % 20 == 0:
                    save_image(gen_imgs.data[:16], "images/EXGAN_%d.png" % epoch, nrow=4, normalize=True)
                
    
    def common_masking(self, D, samples, reals):
        
        '''
        In current prototype version, few-shot setting is not applied yet.

        '''
        # define xai_model
        xai_model = self.args.XAI_method(D)
        
        # extract heatmap from samples       
        heatmap_samples = attributions, delta = xai_model.attribute(samples, target = 0, return_convergence_delta=True)
        
        # extract heatmap from reals
        heatmap_reals = attributions, delta = xai_model.attribute(reals, target = 0, return_convergence_delta=True)
        
        
        # extract common heatmap
        heatmap_com = torch.zeros(self.img_shape)
        for hs, hr in zip(heatmap_samples[0], heatmap_reals[0]):
            hs = hs.cpu()
            hr = hr.cpu()
            heatmap_com += self.alpha * hs + (1-self.alpha) * hr
        # takes average
        heatmap_com /= self.args.shots  
        
        ### visualize imgs ####
        # # show original image
        # reals_np = reals.cpu()
        # imshow(torchvision.utils.make_grid(reals_np[0][0]))
        # # visualize heatmap
        # plt.imshow(heatmap[0][0][0].cpu(), cmap="seismic", clim=(-0.25, 0.25))
        common_mask = getmasking(heatmap_com, version = 'average')
        
        return common_mask
    

    class Generator(nn.Module):
        def __init__(self, args, img_shape):
            super(XAIGAN.Generator, self).__init__()
            self.img_shape = img_shape
    
            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
    
            self.model = nn.Sequential(
                *block(args.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )
    
        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *self.img_shape)
            return img
    
    
    class Discriminator(nn.Module):
        def __init__(self, args, img_shape):
            super(XAIGAN.Discriminator, self).__init__()
            self.img_shape = img_shape
    
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
    
        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
    
            return validity
