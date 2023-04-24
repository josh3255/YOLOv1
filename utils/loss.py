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
        
    def forward(self, pred, target):
        batch_size = pred.shape[0]

        mse_loss = torch.nn.MSELoss()
        ce_loss = torch.nn.CrossEntropyLoss()

        loc_loss = 0

        # localization loss
        obj_mask = target[:, :, :, 4] > 0
        obj_pred = pred[obj_mask]
        obj_target = target[obj_mask]

        pred_bbox1 = obj_pred[:, 0:4]
        pred_bbox2 = obj_pred[:, 5:9]

        target_bbox1 = obj_target[:, 0:4]
        target_bbox2 = obj_target[:, 5:9]
        
        # Use randomly selected boxes for regression 
        # ious1 = compute_iou(pred_bbox1, target_bbox1)
        # ious2 = compute_iou(pred_bbox2, target_bbox2)
        
        # cx, cy to tx, ty after compute iou
        # int_tb1 = torch.floor(target_bbox1[:, :2])
        # target_bbox1[:, :2] = target_bbox1[:, :2] - int_tb1
        # int_tb2 = torch.floor(target_bbox2[:, :2])
        # target_bbox2[:, :2] = target_bbox2[:, :2] - int_tb2
        
        # object loss
        pred_obj1 = obj_pred[:, 4]
        pred_obj2 = obj_pred[:, 9]
        pred_class = obj_pred[:, 5 * self.B : 5 * self.B + self.C]

        target_obj1 = obj_target[:, 4]
        target_obj2 = obj_target[:, 9]
        target_class = obj_target[:, 5 * self.B : 5 * self.B + self.C]
        
        loc_loss = 0
        obj_loss = 0
        noobj_loss = 0
        
        for i in range(len(pred_bbox1)):
            if random.random() > 0.5:
                loc_loss += mse_loss(pred_bbox1[i, :2], target_bbox1[i, :2]) \
                        + mse_loss(pred_bbox1[i, 2:4], torch.sqrt(target_bbox1[i, 2:4]))
                obj_loss += mse_loss(pred_obj1[i], target_obj1[i])
            else:
                loc_loss += mse_loss(pred_bbox2[i, :2], target_bbox2[i, :2]) \
                        + mse_loss(pred_bbox2[i, 2:4], torch.sqrt(target_bbox2[i, 2:4]))
                obj_loss += mse_loss(pred_obj2[i], target_obj2[i]) 
        
        # noobject loc loss
        noobj_mask = target[:, :, :, 4] == 0
        noobj_pred = pred[noobj_mask]
        noobj_target = target[noobj_mask]

        noobj_loss = self.l_noobj * (mse_loss(noobj_pred[:, 4], noobj_target[:, 4]) \
                                    + mse_loss(noobj_pred[:, 9], noobj_target[:, 9]))

        # classification loss
        pred_class = F.softmax(pred_class, dim=1)
        target_class = torch.argmax(target_class, dim=1)
        
        cls_loss = ce_loss(pred_class, target_class)

        return self.l_coord * loc_loss, cls_loss, obj_loss, noobj_loss
        
def compute_iou(boxes1, boxes2):
    ious = []

    img_size = 448
    divisor = img_size // 7

    for box1, box2 in zip(boxes1, boxes2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1 = x1 * divisor
        x2 = x2 * divisor
        y1 = y1 * divisor
        y2 = y2 * divisor

        w1 = w1 ** 2
        h1 = h1 ** 2
        w2 = w2 ** 2
        h2 = h2 ** 2

        # Calculate the coordinates of the intersection rectangle
        x_left = max(x1 - w1 / 2, x2 - w2 / 2)
        y_top = max(y1 - h1 / 2, y2 - h2 / 2)
        x_right = min(x1 + w1 / 2, x2 + w2 / 2)
        y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)
        
        # If the intersection is empty, return 0
        if x_right < x_left or y_bottom < y_top:
            iou = torch.tensor(0.0)
            if torch.cuda.is_available():
                iou = iou.cuda()
            ious.append(iou)
            continue
        
        # Calculate the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate the area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate the IOU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        if torch.cuda.is_available():
            iou = iou.cuda()
        
        ious.append(iou)
    return ious