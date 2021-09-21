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
    
def getmasking(tensor, version = 'thresh', thresh = 0.001, portion = 0.3):
    '''
    

    Parameters
    ----------
    tensor : TYPE
        target tensor input
    version : TYPE, optional
        'thresh': theresholding based on attribution value
        'topk': extract top k portion high attribution value
        'mean' : upper than mean attribution values would be activated
    thresh : TYPE, optional
        theresholding value. The default is 0.001.
    portion : TYPE, optional
        the portion of highest attribution value when the version is 'topk'. The default is 0.3.

    Returns
    -------
    masking matrix that activate(1) and deactivate(0) components

    '''


    if version == 'thresh':
        return (tensor > thresh).float()
    
    if version == 'topk':
        
        # flatten tensor
        #fl = tensor[0].flatten()
        fl = tensor.flatten()
        # number to activate
        n = int(len(fl) * portion)
        # extract topk value indice 
        val, ind = torch.topk(fl, n)
        # assign ind values
        masked = torch.zeros(len(fl))
        masked[ind] = 1
        # reshape it and return tensor
        return masked.reshape(tensor.shape)
    
    else:
        return (tensor>tensor.mean()).float() 


    
class XAIGAN:
    
    def __init__(self, args):
        
        self.args = args
        self.img_shape = (args.channels, args.img_size, args.img_size)
        self.cuda = True if torch.cuda.is_available() else False
        # Configure data loader

        base_dir = './resized_basecelebA'
        transform = transforms.Compose(
                    [transforms.Scale(self.args.img_size),
                      transforms.Resize(self.args.img_size),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std =[0.5, 0.5, 0.5])])
        dset = torchvision.datasets.ImageFolder(base_dir, transform)
        dset.imgs.sort()
        self.basedata_loader = DataLoader(dset, batch_size=self.args.batch_size, shuffle=True)

        self.alpha = args.alpha
        self.beta = args.beta
        
        self.interval = 100  #interval epoch that extract masking
        self.number_sampling = 10
   
    
    def train_dualGAN(self):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        # Initialize generator and discriminator
        generator = XAIGAN.Generator()
        discriminator = XAIGAN.Discriminator()
        discriminator_mask = XAIGAN.Discriminator()
        
        if self.cuda:
            generator.cuda()
            discriminator.cuda()
            discriminator_mask.cuda()
            adversarial_loss.cuda()   
            
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_Dm = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        loss_hist = []
        
        for epoch in range(self.args.n_epochs_dual):
            
        
            for i, (imgs, _) in enumerate(self.basedata_loader):
        
                # Adversarial ground truths for D
                valid = Variable(Tensor(len(imgs), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(len(imgs), 1).fill_(0.0), requires_grad=False)          
                

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                
                if epoch % self.interval == 0:
                    # sample generated images and real ones randomly
                    print(f"heatmap updated at {epoch} epoch step.")
                    z_samp =torch.normal(torch.zeros(len(imgs), 100), torch.ones(len(imgs), 100))
                    z_samp = z_samp.cuda() if self.cuda else z_samp
                    fakes = generator(z_samp)
                    fake_samps = fakes.data[:self.number_sampling]
                    real_samps = real_imgs.data[:self.number_sampling]
                    common_mask = self.common_masking(discriminator, fake_samps, real_samps)
                    if self.cuda: common_mask = common_mask.cuda()
                    self.cm_test = common_mask
                    self.img_test = real_imgs
                
                real_imgs_mask = real_imgs * common_mask
        
                # -----------------
                #  Train Generator
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z = torch.normal(torch.zeros(len(imgs), 100), torch.ones(len(imgs), 100))
                z = z.cuda() if self.cuda else z
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
                tot_loss = d_loss.item() + dm_loss.item() + g_loss.item()
                batches_done = epoch * len(self.basedata_loader) + i
                if epoch % 30 == 0:
                    save_image(gen_imgs.data[:16], "images/EXGAN_%d.png" % epoch, nrow=4, normalize=True)
                    
        if self.args.eval_mode == True:
            number_to_gen = 100 #should be lower than batch size
            for i in range(number_to_gen):
                save_image(gen_imgs.data[i],"samples/sample_%d.png"%i, normalize=True)
                save_image(real_imgs.data[i],"reals/real_%d.png"%i, normalize=True)
                    
                    
        #     loss_hist.append(tot_loss)
        # plt.plot(loss_hist)
    
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
        # plt.imshow(heatmap_com[0].cpu(), cmap="seismic")
        common_mask = getmasking(heatmap_com, version = 'topk', portion = 0.3)
       # print("shape", common_mask.shape)
       
        return common_mask
    

    

    # class Generator(nn.Module):
    #     def __init__(self, args, img_shape):
    #         super(XAIGAN.Generator, self).__init__()
    #         self.img_shape = img_shape
    #         self.args = args
    
    #         def block(in_feat, out_feat, normalize=True):
    #             layers = [nn.Linear(in_feat, out_feat)]
    #             if normalize:
    #                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
    #             layers.append(nn.LeakyReLU(0.2, inplace=True))
    #             return layers
    
    #         self.model = nn.Sequential(
    #             *block(args.latent_dim, 128, normalize=False),
    #             *block(128, 256),
    #             *block(256, 512),
    #             *block(512, 1024),
    #             nn.Linear(1024, int(np.prod(img_shape))),
    #             nn.Tanh()
    #         )
            
    
    #     def forward(self, z):
    #         img = self.model(z)
    #         img = img.view(img.size(0), *self.img_shape)
    #         return img

    
    
    # class Discriminator(nn.Module):
    #     def __init__(self, args, img_shape):
    #         super(XAIGAN.Discriminator, self).__init__()
    #         self.args = args
    #         self.img_shape = img_shape
    
    #         self.model = nn.Sequential(
    #             nn.Linear(int(np.prod(img_shape)), 512),
    #             nn.LeakyReLU(0.2, inplace=True),
    #             nn.Linear(512, 256),
    #             nn.LeakyReLU(0.2, inplace=True),
    #             nn.Linear(256, 1),
    #             nn.Sigmoid(),
    #         )

    
    #     def forward(self, img):
    #         img_flat = img.view(img.size(0), -1)
    #         validity = self.model(img_flat)

    #         return validity
        
    class Generator(nn.Module):
        def __init__(self):
            super(XAIGAN.Generator, self).__init__()
            self.projection = nn.Linear(100, 1024*4*4)
            self.layers = nn.Sequential(
                # First block
                nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
    
                # Second block
                nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
    
                # Third block
                nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
    
                # Fourth block
                nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.Tanh(),
            )
            
        def forward(self, random_noise):
            x = self.projection(random_noise)
            x = x.view(-1, 1024, 4, 4)
            return self.layers(x)
    
        @staticmethod
        def weights_init(layer):
            layer_class_name = layer.__class__.__name__
            if 'Conv' in layer_class_name:
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            elif 'BatchNorm' in layer_class_name:
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)
    
    
    class Discriminator(nn.Module):
        def __init__(self):
            super(XAIGAN.Discriminator, self).__init__()
            self.layers = nn.Sequential(
                # First block
                nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.LeakyReLU(0.2),
    
                # Second block
                nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
    
                # Third block
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
    
                # Fourth block
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
    
                # Fifth block
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
            )
            self.output = nn.Linear(256*2*2, 1) #256*2*2
            self.output_function = nn.Sigmoid()
    
        def forward(self, x):
            x = self.layers(x)
            x = x.view(-1, 256*2*2)
            #x = x.view(-1,2560)
            x = self.output(x)
            return self.output_function(x)
    
        @staticmethod
        def weights_init(layer):
            layer_class_name = layer.__class__.__name__
            if 'Conv' in layer_class_name:
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            elif 'BatchNorm' in layer_class_name:
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)

        