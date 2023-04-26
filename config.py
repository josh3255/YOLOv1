import argparse

def get_args():
    parser = argparse.ArgumentParser(description='YOLO Arguments')

    # dataset args
    parser.add_argument('--train-ann', type=str, default='/data/detection/detection/annotations/coco_train.json', help='COCO format train json')
    parser.add_argument('--val-ann', type=str, default='/data/detection/detection/annotations/val.json', help='COCO format val json')

    # training args
    parser.add_argument('--img-size', type=int, default=448, help='')
    parser.add_argument('--batch-size', type=int, default=8, help='')
    parser.add_argument('--max-epoch', type=int, default=135, help='' )
    parser.add_argument('--eval-interval', type=int, default=5, help='')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--resume', type=str, default='', help='')

    # demo args
    parser.add_argument('--source', type=str, default='', help='')
    parser.add_argument('--output', type=str, default='./demo', help='')
    parser.add_argument('--weights', type=str, default='', help='')
    parser.add_argument('--threshold', type=float, default='0.65', help='')
    parser.add_argument('--iou-threshold', type=float, default='0.3', help='')

    # loss args
    parser.add_argument('--l-coord', type=float, default=5.0, help='')
    parser.add_argument('--l-noobj', type=float, default=0.5, help='')

    # model args
    parser.add_argument('--S', type=int, default=7, help='num of grid cell')
    parser.add_argument('--B', type=int, default=2, help='num of bounding box per cell')
    parser.add_argument('--C', type=int, default=3, help='num of classes')

    # augmentation args

    args = parser.parse_args()
    return args