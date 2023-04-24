import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import logging

from utils.loss import YOLOLoss
from models.yolo import YOLO
from config import get_args
from dataset.coco import COCODataset

def train(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('YOLOv1')

    # dataloader
    train_dataset = COCODataset(args, args.train_ann)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model Set-up
    model = YOLO(args)

    device_count = torch.cuda.device_count()
    devices = [torch.device("cuda:"+str(i) if torch.cuda.is_available() else "cpu") for i in range(device_count)]
    
    model = torch.nn.DataParallel(model, device_ids=range(device_count))
    for i in range(device_count):
        model.to(devices[i])

    # Loss, Optim, and Scheduler Set-up
    criterion = YOLOLoss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,\
                            betas=(args.momentum, 0.999), weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])

    torch.autograd.set_detect_anomaly(True)
    # Model Training
    for epoch in range(args.max_epoch):
        epoch_loss = 0.0

        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            outputs = model(inputs)

            loc_loss, cls_loss, obj_loss, noobj_loss = criterion(outputs, targets)
            total_loss = loc_loss + cls_loss + obj_loss + noobj_loss
            epoch_loss = epoch_loss + total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info('step : {}/{} || total loss : {:.3f} || loc loss : {:.3f} || cls loss : {:.3f} || obj loss : {:.3f} || noobj loss : {:.3f}'\
                                .format(batch_idx + 1, len(train_dataloader), total_loss.item() / args.batch_size, loc_loss.item() / args.batch_size, cls_loss.item() / args.batch_size, obj_loss.item() / args.batch_size, noobj_loss.item() / args.batch_size))

        scheduler.step()

        logger.info('epoch : {}/{} || epoch loss : {:.3f}'.format(epoch + 1, args.max_epoch, epoch_loss / len(train_dataloader) / args.batch_size))

        torch.save({
            'model_state_dict' : model.module.state_dict(),
            # 'optim_state_dict' : optimizer.state_dict()
        }, 'weights/last.pt')
                

def main():
    args = get_args()
    train(args)

if __name__ == "__main__":
	main()