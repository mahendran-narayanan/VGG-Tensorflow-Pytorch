# VGG-Tensorflow-Pytorch
Code contains VGG model in Tensorflow and pytorch

```
usage: main.py [-h] [--model {tf,torch}]

Create VGG model in Tensorflow or Pytorch package

optional arguments:
  -h, --help             show this help message and exit
  --model {tf,torch}     Model will be created on Tensorflow, Pytorch (default: Tensorflow)
  --depth {11,13,16,19}  VGG model depth (default: 11)
```

To create model in Tensorflow:

```
python3 main.py --model tf --depth 11
```

To create model in Pytorch:

```
python3 main.py --model torch --depth 11
```


## VGG Diagram

![alt text](Pictures/VGG.png)

VGG : [Method](https://paperswithcode.com/method/vgg)

Paper link: [VGG paper](https://arxiv.org/pdf/1409.1556v6.pdf)