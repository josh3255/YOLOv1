# YOLOv1

YOLOv1 Paper : [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

This code follows the contents of the paper as much as possible, but has the following differences.

1. Use only one prediction within each cell.

2. To prevent data from being heavily modulated, color space variations are within the 20% range.

3. Confidence and class probability pass through the sigmoid layer.


## TODO

- [x] data augmentation
- [x] add demo.py
- [x] non-maximum suppression
- [ ] validation term in train.py
- [ ] video demo
- [ ] upload pretrained weights


## Installation

Currently, only support docker development environments.

1. Clone the repo

``` shell
git clone https://github.com/josh3255/YOLOv1.git
cd YOLOv1
```

2. Build the docker image

``` shell
docker build -t YOLOv1 docker/
```

3. Docker run

``` shell
nvidia-docker run --name yolov1 -it -v /your/data/path:/data/ -v /your/projects/path:/YOLOv1/ YOLOv1
```

4. Install Requirements

``` shell
pip install -r requirements.txt
```


## Before Training

1. Currently, only support training to COCO format. Please prepare images and json file before training.

2. Modify the config.py file to match your env (batch size, lr, num of class, etc..).

***

## Training

``` shell
python train.py --resume weights/path
```


## Demo

1. make directory to save results

``` shell
mkdir demo
```

2. Run

```shell
python demo.py --weights weights/path --source img/or/folder/path
```

