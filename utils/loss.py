import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class YOLOLoss(nn.Module):
    def __init__(self, args):
        super(YOLOLoss, self).__init__()

        self.img_size = args.img_size

        self.S = args.S
        self.B = args.B
        self.C = args.C

        self.l_coord = args.l_coord
        self.l_noobj = args.l_noobj

        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def compute_iou(self, pred, target):
        # pred: [x1, y1, x2, y2]
        # target: [x1, y1, x2, y2]
        
        # Get coordinates of intersection rectangle
        x1 = max(pred[0], target[0])
        y1 = max(pred[1], target[1])
        x2 = min(pred[2], target[2])
        y2 = min(pred[3], target[3])

        # Calculate area of intersection rectangle
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate area of union rectangle
        pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
        target_area = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
        union = pred_area + target_area - intersection

        # Calculate IoU
        iou = intersection / union

        return iou

    def forward(self, pred, target):
        batch_size = pred.shape[0]

        obj_mask = target[:, :, :, 4] > 0
        noobj_mask = target[:, :, :, 4] == 0

        obj_pred = pred[obj_mask]
        noobj_pred = pred[noobj_mask]
        obj_target = target[obj_mask]
        noobj_target = target[noobj_mask]

        pred_bbox1 = obj_pred[:, :4]
        pred_bbox2 = obj_pred[:, 5:9]

        pred_obj1 = obj_pred[:, 4]
        pred_obj2 = obj_pred[:, 9]

        pred_cls = obj_pred[:, 10:]

        target_bbox1 = obj_target[:, :4]
        target_bbox2 = obj_target[:, 5:9]

        target_obj1 = obj_target[:, 4]
        target_obj2 = obj_target[:, 9]

        target_cls = obj_target[:, 10:]
        
        # If you want to backward by selecting a box with a high iou instead of random backward
        # you can remove the below and implement it using compute_iou function.

        # Due to the bias that occurs during learning, do not use reponsible.
        if random.random() > 0.5:
            loc_loss = self.mse_loss(pred_bbox1, target_bbox1)
            obj_loss = self.mse_loss(pred_obj1, target_obj1)
        else:
            loc_loss = self.mse_loss(pred_bbox2, target_bbox2)
            obj_loss = self.mse_loss(pred_obj2, target_obj2)
        
        cls_loss = self.mse_loss(pred_cls, target_cls)
        noobj_loss = self.mse_loss(noobj_pred[:, 4], noobj_target[:, 4]) + \
                        self.mse_loss(noobj_pred[:, 9], noobj_target[:, 9])

        return self.l_coord * loc_loss / batch_size, cls_loss / batch_size, obj_loss / batch_size, self.l_noobj * noobj_loss / batch_size