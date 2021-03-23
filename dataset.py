import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import cv2
import os

class DocDataset(Dataset):

  def __init__(self, dir="",mode="train",transforms=None,x_transforms=None):
    self.files = os.listdir(mode)
    self.mode = mode
    self.transforms = transforms
    self.x_transforms = x_transforms

  def __len__(self):
      return len(self.files)

      
  def __getitem__(self, index):
    if self.mode=="train":
      x_img = io.imread("data/train/"+self.files[index])
      y_img = io.imread("data/train_cleaned/"+self.files[index])

      if self.transforms:
        t = self.transforms(image=x_img,image0=y_img)
        x_img = t['image']
        y_img = t['image0']

        if self.x_transforms:
          x_t = self.transforms(image=x_img)
          x_img = x_t['image']


        x_img = transforms.ToTensor()(x_img)
        y_img = transforms.ToTensor()(y_img)

        # x_img = self.transforms(x_img)
        # y_img = self.transforms(y_img)
      return x_img, y_img

    else:
      x_img = io.imread("data/test/"+self.files[index])
      
      img_name = "test/"+self.files[index]



      if self.transforms:
        t = self.transforms(image=x_img)
        x_img = t['image']

      x_img = transforms.ToTensor()(x_img)

      return x_img,img_name