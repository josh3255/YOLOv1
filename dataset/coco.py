import os
import cv2

import torch
import torchvision

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

class COCODataset(Dataset):
    def __init__(self, args, json_path):
        self.args = args
        self.coco = COCO(json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_size = 448

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        annotations = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']

            x1 = x1 / self.coco.imgs[img_id]['width'] * self.img_size
            y1 = y1 / self.coco.imgs[img_id]['height'] * self.img_size
            w = w / self.coco.imgs[img_id]['width'] * self.img_size
            h = h / self.coco.imgs[img_id]['height'] * self.img_size
            cls = ann['category_id'] - 1
            annotations.append([x1, y1, w, h, cls])

        target = self.encoder(annotations)
        
        if torch.cuda.is_available():
            return img.cuda(), target.cuda()
        else:
            return img, target

    def __len__(self):
        return len(self.ids)

    def encoder(self, annotations):
        
        divisor = self.img_size // self.args.S

        target = torch.zeros((self.args.S, self.args.S, self.args.B * 5 + self.args.C))
        
        for annotation in annotations:
            x1, y1, w, h, cls = annotation
            cx = (x1 + (w / 2)) / divisor
            cy = (y1 + (h / 2)) / divisor
            w = w / self.img_size
            h = h / self.img_size

            cell_pos_x = int(cx)
            # offset_x = cx - cell_pos_x
            
            cell_pos_y = int(cy)
            # offset_y = cy - cell_pos_y
            
            if target[cell_pos_x][cell_pos_y][4] != 0:
                continue

            target[cell_pos_x][cell_pos_y][0:5] = torch.tensor([cx, cy, w, h, 1])
            target[cell_pos_x][cell_pos_y][5:10] = torch.tensor([cx, cy, w, h, 1])
            target[cell_pos_x][cell_pos_y][10+int(cls)] = 1

        return target

def collate_fn(batch):
    max_objs = max([len(b[1]) for b in batch])
    images = []
    targets = []

    for image, target in batch:
        images.append(image)

        num_objs = len(target)
        padding = torch.zeros((max_objs - num_objs, 5))
        target = torch.tensor(target)
        target = torch.cat((target, padding), dim=0)
        targets.append(target)

    return torch.stack(images), torch.stack(targets)
