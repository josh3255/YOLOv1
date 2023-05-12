import os
import cv2

import math

import torch
import torchvision

from utils.utils import get_all_files_in_folder

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
video_extensions = ['mp4', 'avi', 'wmv', 'mov', 'flv', 'mkv', 'mpeg', 'webm']

class COCODataset(Dataset):
    def __init__(self, args, json_path):
        self.args = args
        self.coco = COCO(json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_size = args.img_size
        self.cell_size = self.img_size // self.args.S

        self.transforms = A.Compose([
            A.Resize(self.img_size, self.img_size),
            # A.RandomCrop(self.img_size, self.img_size, p=0.2),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.2),
            ToTensorV2(),
        ], bbox_params={'format' : 'coco'})

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            cls = ann['category_id'] - 1

            if w <= 0 or h <= 0:
                continue
            bboxes.append([x1, y1, w, h, cls])
        
        augmented = self.transforms(image=img, bboxes=bboxes)
        img = augmented['image'].float() / 255.0
        bboxes = augmented['bboxes']
        target = self.encoder(bboxes)

        if torch.cuda.is_available():
            return img.cuda(), target.cuda()
        else:
            return img, target

    def __len__(self):
        return len(self.ids)

    def encoder(self, bboxes): 
        target = torch.zeros((self.args.S, self.args.S, self.args.B * 5 + self.args.C))
        
        for bbox in bboxes:
            x1, y1, w, h, cls = bbox

            cx = (x1 + (w / 2))
            cy = (y1 + (h / 2))
            w = math.sqrt(w)
            h = math.sqrt(h)

            cell_x = cx / self.cell_size
            tx = cell_x - int(cell_x)

            cell_y = cy / self.cell_size
            ty = cell_y - int(cell_y)

            if target[int(cell_y)][int(cell_x)][4] != 0:
                continue

            target[int(cell_y)][int(cell_x)][0:5] = torch.tensor([tx, ty, w, h, 1.0])
            target[int(cell_y)][int(cell_x)][5:10] = torch.tensor([tx, ty, w, h, 1.0])
            target[int(cell_y)][int(cell_x)][5 * self.args.B + int(cls)] = 1.0

        return target

class ValDataset(Dataset):
    def __init__(self, args, json_path):
        self.args = args
        self.coco = COCO(json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_size = args.img_size

        self.transforms = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2(),
        ])

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmented = self.transforms(image=img)
        img = augmented['image'].float() / 255.0
        
        if torch.cuda.is_available():
            return torch.tensor(img_id).cuda(), img.cuda()
        else:
            return torch.tensor(img_id), img

    def __len__(self):
        return len(self.ids)

class TestDataset(Dataset):
    def __init__(self, args, path):
        self.args = args
        self.path = path

        self.img_size = args.img_size
        self.transforms = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2(),
        ])

        if os.path.isdir(path):
            file_list = os.listdir(path)
            self.files = sorted([os.path.join(path, f) for f in file_list if f.split('.')[-1].lower() in image_extensions])
        elif os.path.isfile(path):
            ext = path.split('.')[-1].lower()
            if ext in image_extensions:
                self.files = [path]
            elif ext in video_extensions:
                self.files = None
                self.cap = cv2.VideoCapture(path)
                self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, index):
        if self.files is not None:
            img = cv2.imread(self.files[index])
            ori_img = cv2.resize(img.copy(), (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, img = self.cap.read()
            if not ret:
                raise ValueError('Failed to read frame')
            ori_img = cv2.resize(img.copy(), (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=img)
        img = augmented['image'].float() / 255.0

        return img, ori_img, self.files[index]

    def __len__(self):
        if self.files is not None:
            return len(self.files)
        else:
            return self.n_frames

def collate_fn(batch):
    images = []
    ori_images = []
    file_paths = []

    for image, ori_img, path in batch:
        images.append(image)
        ori_images.append(ori_img)
        file_paths.append(path)
    
    return torch.stack(images), ori_images, file_paths