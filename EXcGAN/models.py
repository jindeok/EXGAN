import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image

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
        
        tensorlist = [tensor[0],tensor[1],tensor[2]]
        for i, ts in enumerate(tensorlist):
            
            # flatten tensor
            fl = ts.flatten()
            # number to activate
            n = int(len(fl) * portion)
            # extract topk value indice 
            val, ind = torch.topk(fl, n)
            # assign ind values
            masked = torch.zeros(len(fl))
            masked[ind] = 1
            # reshape it and return tensor
            masked = masked.reshape(ts.shape)
            tensor[i] = masked
            
        return tensor
    
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

        few_dir = './resized_novelcelebA'
        transform = transforms.Compose(
                    [transforms.Scale(self.args.img_size),
                      transforms.Resize(self.args.img_size),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std =[0.5, 0.5, 0.5])])
        dset_f = torchvision.datasets.ImageFolder(few_dir , transform)
        dset_f.imgs.sort()
        print(dset_f.class_to_idx)
        self.fewdata_loader = DataLoader(dset_f, batch_size=self.args.batch_size, shuffle=True)

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
            for i, (imgs, _) in enumerate(self.basedata_loader):
        
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
                

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
        
                ### train G ###
        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim))))
        
                # Generate a batch of images
                gen_imgs = generator(z)
        
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
                g_loss.backward()
                optimizer_G.step()
        
                ### train D ###
        
                optimizer_D.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
        
                d_loss.backward()
                optimizer_D.step()
        
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.args.n_epochs, i, len(self.basedata_loader), d_loss.item(), g_loss.item())
                )
        
                batches_done = epoch * len(self.basedata_loader) + i
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
        loss_hist = []
        for epoch in range(self.args.n_epochs_dual):
            for i, (imgs, label) in enumerate(self.fewdata_loader):

        
                # Adversarial ground truths for D
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
                
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                
                real_imgs_mask = real_imgs * common_mask
        
        
                #label processing
                label= torch.nn.functional.one_hot(label, num_classes= self.args.shots)
                label = label.to(device='cuda')
                
                ### train G ###        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim))))
        
                # Generate a batch of images
                gen_imgs = generator.forward_few(z, label)
                gen_imgs_mask = gen_imgs * common_mask
        
                # Loss measures generator's ability to fool the discriminator
                g_loss = (1-self.beta) * adversarial_loss(discriminator(gen_imgs), valid) \
                    + self.beta * adversarial_loss(discriminator_mask(gen_imgs_mask), valid)
        
                g_loss.backward()
                optimizer_G.step()
        
                ### train D ###
        
                optimizer_D.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator.forward_few(real_imgs, label), valid)
                fake_loss = adversarial_loss(discriminator.forward_few(gen_imgs.detach(), label), fake)
                d_loss = (1-self.beta) * (real_loss + fake_loss) / 2
                
        
                d_loss.backward()
                optimizer_D.step()
                
                ### train mask-D ###
        
                optimizer_Dm.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss_mask = adversarial_loss(discriminator_mask.forward_few(real_imgs_mask, label), valid)
                fake_loss_mask = adversarial_loss(discriminator_mask.forward_few(gen_imgs_mask.detach(), label), fake)
                dm_loss = self.beta * (real_loss_mask + fake_loss_mask) / 2
                
        
                dm_loss.backward()
                optimizer_Dm.step()
        
                print(
                    "[Epoch %d/%d] [D loss: %f] [Dm loss: %f] [G loss: %f]"
                    % (epoch, self.args.n_epochs_dual, d_loss.item(), dm_loss.item(), g_loss.item())
                )
                tot_loss = d_loss.item() + dm_loss.item() + g_loss.item()
                batches_done = epoch * len(self.fewdata_loader) + i
                if epoch % 20 == 0:
                    save_image(gen_imgs.data[:16], "images/EXGAN_%d.png" % epoch, nrow=4, normalize=True)
                    
        if self.args.eval_mode == True:
            # SHould also be changed by putting labels
            number_to_gen = 100 #should be lower than batch size
            for i in range(number_to_gen):
                save_image(gen_imgs.data[i],"samples/sample_%d.png"%i, normalize=True)
                save_image(real_imgs.data[i],"reals/real_%d.png"%i, normalize=True)
                    
                    
        # loss_hist.append(tot_loss)
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
        imshow(common_mask.cpu())
       # print("shape", common_mask.shape)
       
        return common_mask, heatmap_com
    

    

    class Generator(nn.Module):
        def __init__(self, args, img_shape):
            super(XAIGAN.Generator, self).__init__()
            self.img_shape = img_shape
            self.args = args
    
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
            
            self.model_few = nn.Sequential(
                *block(args.latent_dim + args.shots, 128, normalize=False),
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
        
        def forward_few(self, z, label):
            
            input = torch.cat((z, label), 1) 
            img = self.model_few(input)
            img = img.view(img.size(0), *self.img_shape)
            
            return img
    
    
    class Discriminator(nn.Module):
        def __init__(self, args, img_shape):
            super(XAIGAN.Discriminator, self).__init__()
            self.args = args
            self.img_shape = img_shape
    
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            
            self.model_few = nn.Sequential(
                nn.Linear(int(np.prod(img_shape) + args.shots), 512),
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
        
        def forward_few(self, img, label):
            
            img_flat = img.view(img.size(0), -1)
            input = torch.cat((img_flat, label), 1)           
            validity = self.model_few(input)
    
            return validity


class Fewshot_dataset(Dataset):

    def __init__(self, file_list, transform, num_few):
        self.file_list = file_list
        self.transform = transform
        self.num_few = num_few

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        
        img_path = self.file_list[index] # 데이터셋에서 파일 하나를 특정
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path[0:1] # 파일명으로부터 라벨명 추출        
        label_onehot = torch.nn.functional.one_hot(label, num_classes= self.num_few) # 현재 퓨 샷 유저는 10명

        return img_transformed, label_onehot