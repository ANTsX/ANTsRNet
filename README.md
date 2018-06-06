[![Build Status](https://travis-ci.org/ANTsX/ANTsRNet.png?branch=master)](https://travis-ci.org/ANTsX/ANTsRNet)

# ANTsRNet

A collection of well-known deep learning architectures ported to the R language and tools for basic medical image processing.

Examples available at [ANTsRNetExamples](https://github.com/ntustison/ANTsRNetExamples).

## Image segmentation

* U-Net (2-D) or V-Net (3-D) with a [multi-label Dice loss function](https://github.com/ntustison/ANTsRNet/blob/master/Models/createUnetModel.R#L1-L91)
    * [O. Ronneberger, P. Fischer, and T. Brox.  U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597)
    * [Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.](https://arxiv.org/pdf/1606.04797.pdf)

## Image classification

* AlexNet (2-D, 3-D)
    * [A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification with Deep Convolutional Neural Networks.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* Vgg16/Vgg19 (2-D, 3-D)
    * [K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition.](https://arxiv.org/abs/1409.1556)
* ResNet/ResNeXt (2-D, 3-D)
    * [Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.  Deep Residual Learning for Image Recognition.](https://arxiv.org/abs/1512.03385)
    * [Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He.  Aggregated Residual Transformations for Deep Neural Networks.](https://arxiv.org/abs/1611.05431)
* GoogLeNet (Inception v3) (2-D)
    * [C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going Deeper with Convolutions.](https://arxiv.org/abs/1512.00567)
* DenseNet (2-D, 3-D)
    * [G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected Convolutional Networks Networks.](https://arxiv.org/abs/1608.06993)

## Object detection

* Single Shot MultiBox Detector (SSD) (2-D, 3-D)
    * [W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, and A. Berg.  SSD: Single Shot MultiBox Detector.](https://arxiv.org/abs/1512.02325)
    * SSD7: small 7-layer architecture
    * SSD300/SSD512: porting of original architectures

## Image super-resolution

* Super-resolution convolutional neural network (SRCNN) (2-D, 3-D)
    * [Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang.  Image Super-Resolution Using Deep Convolutional Networks.](https://arxiv.org/abs/1501.00092)
* Expanded super-resolution (ESRCNN) (2-D, 3-D)
    * [Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang.  Image Super-Resolution Using Deep Convolutional Networks.](https://arxiv.org/abs/1501.00092)
* Denoising auto encoder super-resolution (DSRCNN) (2-D, 3-D)
* Deep denoise super-resolution (DDSRCNN) (2-D, 3-D)
* ResNet super-resolution (SRResNet) (2-D, 3-D)
    * [Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi.  Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.](https://arxiv.org/abs/1609.04802)
* [Python implementations](https://github.com/titu1994/Image-Super-Resolution/)    
