import cv2
import torch

import logging

import numpy as np

from collections import OrderedDict

from models.yolo import YOLO
from config import get_args
from dataset.coco import TestDataset, collate_fn
from utils.utils import post_processing, draw_bbox
from utils.utils import non_max_suppression
from torch.utils.data import DataLoader

def demo(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('YOLOv1')

    # dataloader
    test_dataset = TestDataset(args, args.source)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1, shuffle=False)
    
    # Model Set-up
    model = YOLO(args)
    model.eval()

    device_count = torch.cuda.device_count()
    devices = [torch.device("cuda:"+str(i) if torch.cuda.is_available() else "cpu") for i in range(device_count)]
    
    model = torch.nn.DataParallel(model, device_ids=range(device_count))
    for i in range(device_count):
        model.to(devices[i])

    if args.weights != '':
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for batch_idx, (inp, img, path) in enumerate(test_dataloader):
            output = model(inp)
            output = output.view(7, 7, 5 * args.B + args.C)
            
            bboxes, scores, classes = post_processing(args, output)
            bboxes, scores = non_max_suppression(bboxes, scores, args.iou_threshold)
            
            # Only batch size 1 is supported.
            img = img[0]
            path = path[0]
            for bbox, score, cls in zip(bboxes, scores, classes):
                img = draw_bbox(img, bbox, cls.item(), score.item())
            cv2.imwrite('./demo/{}'.format(path.split('/')[-1]), img) 

def main():
    args = get_args()
    demo(args)

if __name__ == "__main__":
	main()