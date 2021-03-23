import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os
from torchvision import transforms
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray




def get_args(parser):
    parser.add_argument("--input_image", type=str, default="./images/i_img.jpeg")


class DocEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.LeakyReLU(0.1)            
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)      
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(64,64,3,stride=2,padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.out = torch.nn.Sigmoid()     
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.up(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.out(x)
        
        return x
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        op = self.forward(x)
        loss= F.mse_loss(op, y)
        self.log('train_loss', loss,on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        op = self.forward(x)
        loss= F.mse_loss(op, y)
        self.log('val_loss', loss,on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer       
        
 

def op_visualize(x,op):

    o_img = op.detach().cpu().numpy().squeeze(0)
    o_img = o_img.transpose(1,2,0)
    o_img = np.clip(o_img, 0, 1)
    x = x.detach().cpu().numpy().squeeze(0)
    x = x.transpose(1,2,0)
    io.imsave("test.jpg", o_img)

    f, axarr = plt.subplots(1,2,figsize=(10,20))
    axarr[0].imshow(x,cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(o_img,cmap='gray')
    axarr[1].axis('off')

    plt.show()

def model_inf(args,model):
    i_img = io.imread(args.input_image)
    i_img = rgb2gray(i_img)
    x = transforms.ToTensor()(i_img).unsqueeze(0)
    x = x.to(dtype=torch.float32) 
    op=model(x)
    op_visualize(x,op)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    print(args.input_image)

    model = DocEncoder.load_from_checkpoint("best_model.ckpt")
    model.eval()

    model_inf(args,model)






