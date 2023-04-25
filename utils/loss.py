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
        
    def forward(self, pred, target):
        # Due to the bias that occurs during the learning process, do not use reponsible.

        batch_size = pred.shape[0]

        loc_loss = 0
        obj_loss = 0
        cls_loss = 0
        noobj_loss = 0

        obj_mask = target[:, :, :, 4] > 0
        obj_pred = pred[obj_mask]
        obj_target = target[obj_mask]
        
        pred_bbox1 = obj_pred[:, 0:4]
        pred_bbox2 = obj_pred[:, 5:9]

        pred_obj1 = obj_pred[:, 4]
        pred_obj2 = obj_pred[:, 9]

        pred_class = obj_pred[:, 5 * self.B : 5 * self.B + self.C]

        target_bbox = obj_target[:, 0:4]
        target_obj = obj_target[:, 4]
        target_class = obj_target[:, 5 * self.B : 5 * self.B + self.C]
        
        # (obj) localization loss & objectness loss
        # Use random selected boxes for regression
        if random.random() > 0.5:
            loc_loss += self.mse_loss(pred_bbox1, target_bbox)
            obj_loss += self.mse_loss(pred_obj1, target_obj)
        else:
            loc_loss += self.mse_loss(pred_bbox2, target_bbox)
            obj_loss += self.mse_loss(pred_obj2, target_obj)
        
        # (obj) classification loss
        cls_loss += self.mse_loss(pred_class, target_class)
        
        
        noobj_mask = target[:, :, :, 4] == 0
        noobj_pred = pred[noobj_mask]
        noobj_target = target[noobj_mask]

        noobj_pred_bbox1 = noobj_pred[:, 0:4]
        noobj_pred_bbox2 = noobj_pred[:, 5:9]

        noobj_pred_obj1 = noobj_pred[:, 4]
        noobj_pred_obj2 = noobj_pred[:, 9]

        noobj_pred_class = noobj_pred[:, 5 * self.B : 5 * self.B + self.C]

        noobj_target_bbox = noobj_target[:, :4]
        noobj_target_obj = noobj_target[:, 4]
        noobj_target_class = noobj_target[:, 5 * self.B : 5 * self.B + self.C]

        # (noobj) localization loss
        noobj_loss += self.mse_loss(noobj_pred_bbox1, noobj_target_bbox)
        noobj_loss += self.mse_loss(noobj_pred_bbox2, noobj_target_bbox)

        # (noobj) objectness loss
        noobj_loss += self.mse_loss(noobj_pred_obj1, noobj_target_obj)
        noobj_loss += self.mse_loss(noobj_pred_obj2, noobj_target_obj)

        # (noobj) classification loss
        noobj_loss += self.mse_loss(noobj_pred_class, noobj_target_class)

        return self.l_coord * loc_loss / batch_size, cls_loss / batch_size, obj_loss / batch_size, self.l_noobj * noobj_loss / batch_size