import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.data_root = 'vimeo_septuplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist))
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def crop(self, img0, img1, gt, img2, img3, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = [img0, img1, gt, img2, img3]
        imgs = [img[x:x + h, y:y + w, :] for img in imgs]
        img0, img1, gt, img2, img3 = imgs
        return img0, img1, gt, img2, img3

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png',
                    imgpath + '/im7.png']

        # Load images
        imgs = [cv2.imread(f) for f in imgpaths]
        img0, img1, gt, img2, img3 = [cv2.resize(img, (960, 540)) for img in imgs]
        timestep = 0.5
        return img0, img1, gt, img2, img3, timestep

    def __getitem__(self, index):
        img0, img1, gt, img2, img3, timestep = self.getimg(index)
        imgs = [img0, img1, gt, img2, img3]
        if self.dataset_name == 'train':
            img0, img1, gt, img2, img3 = self.crop(img0, img1, gt, img2, img3, 448, 448)
            imgs = [img0, img1, gt, img2, img3]
            imgs = [img[:, :, ::-1] for img in imgs]
            if random.uniform(0, 1) < 0.5:
                imgs = [img[::-1] for img in imgs]
            if random.uniform(0, 1) < 0.5:
                imgs = [img[:, ::-1] for img in imgs]
            if random.uniform(0, 1) < 0.5:
                imgs = reversed(imgs) # gt位置不会变
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in imgs]
            elif p < 0.5:
                imgs = [cv2.rotate(img, cv2.ROTATE_180) for img in imgs]
            elif p < 0.75:
                imgs = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in imgs]
        imgs = [torch.from_numpy(img.copy()).permute(2, 0, 1) for img in imgs]
        img0, img1, gt, img2, img3 = imgs
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, img2, img3, gt), 0), timestep
