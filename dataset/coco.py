import os
import cv2

import torch
import torchvision

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

class COCODataset(Dataset):
    def __init__(self, json_path):
        self.coco = COCO(json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cls = ann['category_id'] - 1
            target.append([x, y, w, h, cls])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)

        target = torch.tensor(target)
        
        return img ,target

    def __len__(self):
        return len(self.ids)

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
