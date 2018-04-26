# ANTsRNet

A collection of well-known deep learning architectures ported to the R language.

And tools for basic medical image processing.

## R package dependencies

* [ANTsR](https://github.com/stnava/ANTsR)
* [keras](https://github.com/rstudio/keras) (install from github: ``devtools::install_github( "rstudio/keras")``)
* abind

## Image segmentation

* U-Net (2-D) or V-Net (3-D) with a [multi-label Dice loss function](https://github.com/ntustison/ANTsRNet/blob/master/Models/createUnetModel.R#L1-L91)
    * [O. Ronneberger, P. Fischer, and T. Brox.  U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    * [Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/pdf/1606.04797.pdf)

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

* Single Shot MultiBox Detector (2-D, 3-D) 
    * [W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, and A. Berg.  SSD: Single Shot MultiBox Detector.](https://arxiv.org/abs/1512.02325)
    * SSD7: small 7-layer architecture 
    * SSD300/SSD512: porting of original architectures
 
# Misc topics

* Optimizers
* [Blog:  Important papers](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
* [Blog:  Intuitive explanation of convnets](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [Deep learning book](http://www.deeplearningbook.org)
* Important Keras [FAQ](https://keras.rstudio.com/articles/faq.html)
* Custom keras layers in R
    * [Link 1](https://keras.rstudio.com/articles/custom_layers.html)
    * [Link 2](https://cran.rstudio.com/web/packages/keras/vignettes/about_keras_layers.html)
    * [Link 3](https://cran.rstudio.com/web/packages/keras/vignettes/custom_layers.html)

# To do:

* __WIP:__ YOLO9000
    * [Joseph Redmon, Ali Farhadi.  YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
    * [Implementation](https://github.com/ykamikawa/yolov2)
* deconvnet.R
* ResNet and AlexNet use lambda layers so those models aren't writeable to file (h5 format).  So we need to redo to rewrite to json or something else.  At least I think that's the problem. 
* Need to go through and make sure that the 'tf' vs. 'th' ordering is accounted for.  Currently, tensorflow is assumed.  Should work with theano but need to check this.  Actually, given that Theano is [no longer in active development](https://groups.google.com/forum/#!topic/theano-users/7Poq8BZutbY), perhaps we should just stick with a tensorflow backend.

****************
****************

# My GPU set-up

## Hardware

* Computer 
    * iMac (27-inch, Mid 2011)
    * Processor 3.4 GHz Intel Core i7
    * Memory 16 GB 1333 MHz DDR3 
    * macOS High Sierra (Version 10.13.2)
* GPU
    * [NVIDIA Titan Xp](https://www.nvidia.com/en-us/titan/titan-xp/)
    * [Akitio Node - Thunderbolt3 eGPU](https://www.akitio.com/expansion/node)
    * [Thunderbolt 3 <--> Thunderbolt 2 adapter](https://www.apple.com/shop/product/MMEL2AM/A/thunderbolt-3-usb-c-to-thunderbolt-2-adapter)
    * [Thunderbolt 2 cable](https://www.apple.com/shop/product/MD862LL/A/apple-thunderbolt-cable-2-m)

## Software

* Tensorflow-gpu
* Keras in R
* [NVIDIA CUDA toolkit 9.1](https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1012)
* [NVIDIA CUDA Deep Neural Network library (cuDNN) 7.0](https://www.developer.nvidia.com/cudnn)
* Python 3.6

## Set-up

(see note in Misc. about when to plug in/turn on eGPU)

1. [Put together Titan XP and Aikito node](https://becominghuman.ai/deep-learning-gaming-build-with-nvidia-titan-xp-and-macbook-pro-with-thunderbolt2-5ceee7167f8b)
2. [Install web drivers and GPU support](https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/)
3. Install NVIDIA toolkit and cuDNN
4. Re-install web drivers and GPU support
5. [Install tensorflow-gpu](https://medium.com/@fabmilo/how-to-compile-tensorflow-with-cuda-support-on-osx-fd27108e27e1)    
6. [Install keras with tensorflow-gpu](https://keras.rstudio.com)

__Update (April 11, 2018):__ The recent MacOSx update (10.13.4) broke the eGPU compatiblity as explained [here](https://egpu.io/forums/mac-setup/script-enable-egpu-on-tb1-2-macs-on-macos-10-13-4/).  

## Misc. notes

* I originally set-up the hardware followed by the drivers (steps 1 and 2) but the tensorflow installation caused some problems.  I believe they were from ``csrutil enable --without kext`` instead of ``csrutil disable`` in step 2 so I ended up using the latter.
* As described in the [comments](https://gist.github.com/smitshilu/53cf9ff0fd6cdb64cca69a7e2827ed0f), I had to change the following files:
    * tensorflow/third_party/gpus/cuda/BUILD.tpl (comment out line 113 ``linkopts = ["-lgomp"],``)
    * tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc (remove all instances of ``align(sizeof(T))``)
    * tensorflow/core/kernels/split_lib_gpu.cu.cc (remove all instances of ``align(sizeof(T))``)
    * tensorflow/core/kernels/concat_lib_gpu.impl.cu.cc (remove all instances of ``align(sizeof(T))``)
* Since I ended up re-installing the NVIDIA drivers, I think I should have performed Step 3 before Step 2 in the Set-up above.  
* I had to revert back to older Xcode and command line tools (8.3.2) and then switch back.  


* Time differences on [MNIST example](https://github.com/ntustison/ANTsRNet/blob/master/Examples/AlexNetExample/mnist.R)
    * tensorflow-cpu on Mac Pro (Late 2013):  ~2100 seconds / epoch
    * tensorflow-gpu (the described set-up):  ~97 seconds / epoch
* Time differences on [U-net example](https://github.com/ntustison/ANTsRNet/tree/master/Examples/UnetExample)
    * tensorflow-cpu on Mac Pro (Late 2013):  ~56 seconds / epoch
    * tensorflow-gpu (the described set-up):  ~2 seconds / epoch

* During a run a kernel panic resulted in the computer shutting down.  When it came back on, the GPU was no longer recognized but was listed as a "NVIDIA chip" in the "About this mac" --> "System Report" --> "Graphics/Displays".  Reinstalling the web driver and eGPU support didn't bring it back but then I read where I needed to 
    1. unplug the eGPU
    2. Boot into OSX
    3. Login
    4. Plug in the eGPU
    5. Logout
    6. Log back in
