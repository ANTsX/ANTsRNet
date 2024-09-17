<!--
[![Build Status](https://travis-ci.org/ANTsX/ANTsRNet.png?branch=master)](https://travis-ci.org/ANTsX/ANTsRNet)
[![Codecov test coverage](https://codecov.io/gh/muschellij2/ANTsRNet/branch/master/graph/badge.svg)](https://codecov.io/gh/muschellij2/ANTsRNet?branch=master)
-->

 <!-- badges: start -->
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)
[![PubMed](https://img.shields.io/badge/ANTsX_paper-Open_Access-8DABFF?logo=pubmed)](https://pubmed.ncbi.nlm.nih.gov/33907199/)

<!-- badges: end -->

# ANTsRNet

A collection of deep learning architectures and applications ported to the R language and tools for basic medical image processing. Based on `keras` and `tensorflow` with cross-compatibility with our python analog [ANTsPyNet](https://github.com/ntustison/ANTsPyNet/).

[Documentation page](https://antsx.github.io/ANTsRNet/)

[ANTsXNet tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#antsxnet)

<p align="middle">
  <img src="docs/figures/coreANTsXNetTools.png" width="600" />
</p>

<details>
<summary>Installation</summary>
 
### Prerequisites

You will need R (>=3.2) and C/C++ development tools including [CMake](https://cmake.org/download/) (>= 3.16.3).

### Installation steps

First, install keras in R

```R
> install.packages(keras)
> keras::install_keras()
```

Then from the command line:

```bash
git clone https://github.com/stnava/ITKR.git
git clone https://github.com/ANTsX/ANTsRCore.git
git clone https://github.com/ANTsX/ANTsR.git
R CMD INSTALL ITKR 
R CMD INSTALL ANTsRCore
R CMD INSTALL ANTsR
R CMD INSTALL ANTsRNet

```
</details>

<details>
<summary>Architectures</summary>

### Image voxelwise segmentation/regression

* [U-Net (2-D, 3-D)](https://arxiv.org/abs/1505.04597)
* [U-Net + ResNet (2-D, 3-D)](https://arxiv.org/abs/1608.04117)
* [Dense U-Net (2-D, 3-D)](https://arxiv.org/pdf/1709.07330.pdf)

### Image classification/regression

* [AlexNet (2-D, 3-D)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* [VGG (2-D, 3-D)](https://arxiv.org/abs/1409.1556)
* [ResNet (2-D, 3-D)](https://arxiv.org/abs/1512.03385)
* [ResNeXt (2-D, 3-D)](https://arxiv.org/abs/1611.05431)
* [WideResNet (2-D, 3-D)](http://arxiv.org/abs/1605.07146)
* [DenseNet (2-D, 3-D)](https://arxiv.org/abs/1608.06993)

### Object detection

### Image super-resolution

* [Super-resolution convolutional neural network (SRCNN) (2-D, 3-D)](https://arxiv.org/abs/1501.00092)
* [Expanded super-resolution (ESRCNN) (2-D, 3-D)](https://arxiv.org/abs/1501.00092)
* [Denoising auto encoder super-resolution (DSRCNN) (2-D, 3-D)]()
* [Deep denoise super-resolution (DDSRCNN) (2-D, 3-D)](https://arxiv.org/abs/1606.08921)
* [ResNet super-resolution (SRResNet) (2-D, 3-D)](https://arxiv.org/abs/1609.04802)
* [Deep back-projection network (DBPN) (2-D, 3-D)](https://arxiv.org/abs/1803.02735)
* [Super resolution GAN](https://arxiv.org/abs/1609.04802)

### Registration and transforms

* [Spatial transformer network (STN) (2-D, 3-D)](https://arxiv.org/abs/1506.02025)

### Generative adverserial networks

* [Generative adverserial network (GAN)](https://arxiv.org/abs/1406.2661)
* [Deep Convolutional GAN](https://arxiv.org/abs/1511.06434)
* [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* [Improved Wasserstein GAN](https://arxiv.org/abs/1704.00028)
* [Cycle GAN](https://arxiv.org/abs/1703.10593)
* [Super resolution GAN](https://arxiv.org/abs/1609.04802)

### Clustering

* [Deep embedded clustering (DEC)](https://arxiv.org/abs/1511.06335)
* [Deep convolutional embedded clustering (DCEC)](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

</details>

<details>
<summary>Applications</summary>

* [Brain applications](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#brain-applications)

    * [Multi-modal brain extraction](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#brain-extraction)
    * [Deep Atropos (Six-tissue brain segmentation)](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#deep-atropos)
    * [Cortical thickness](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#cortical-thickness)
    * [Desikan-Killiany-Tourville parcellation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#desikan-killiany-tourville-parcellation)
    * [DeepFLASH (medial temporal lobe parcellation)](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#deepflash-medial-temporal-lobe-parcellation)
    * [Hippmapp3r (hippocampal segmentation)](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#hippmapp3r)
    * [Brain AGE](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#brain-age)
    * [Claustrum segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#claustrum-segmentation)
    * [Hypothalamus segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#hypothalamus-segmentation)
    * [Cerebellum morphology](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#cerebellum-morphology)
    * White matter hyperintensities segmentation 
        * [SYSU](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#white-matter-hyperintensities-segmentation-sysu)
        * [Hypermapp3r](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#white-matter-hyperintensities-segmentation-hypermapp3r)
        * [SHIVA](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#white-matter-hyperintensities-segmentation-shiva)
        * [ANTsXNet](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#white-matter-hyperintensities-segmentation-antsxnet)
    * [Perivascular spaces segmentation (SHIVA)](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#perivascular-spaces-segmentation-shiva)
    * [Brain tumor segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#brain-tumor-segmentation)
    * [MRA-TOF vessel segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#mra-tof-vessel-segmentation)
    * [Lesion segmentation (WIP)](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#lesion-segmentation-wip)
    * [Whole head inpainting](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#whole-head-inpainting)

* [Lung applications](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#lung-applications)

    * [Lung extraction](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#lung-extraction) 
    * [Functional lung segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#functional-lung-segmentation)
    * [Pulmonary artery segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#pulmonary-artery-segmentation)
    * [Pulmonary airway segmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#pulmonary-airway-segmentation)
    * [CheXNet](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#chexnet)

* [Mouse applications](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#mouse-applications)
    * [Mouse brain extraction](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#mouse-brain-extraction)
    * [Mouse brain parcellation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#mouse-brain-parcellation)
    * [Mouse cortical thickness](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#mouse-cortical-thickness)

* [General applications](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#general-applications)

    * [MRI super resolution](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#mri-super-resolution)
    * [No reference image quality assesment using TID](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#no-reference-image-quality-assesment-using-tid)
    * [Full reference image quality assessment](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#full-reference-image-quality-assessment)

* [Data augmentation](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#data-augmentation)

    * [Noise](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#noise)
    * [Histogram intensity warping](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#histogram-intensity-warping)
    * [Simulate bias field](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#simulate-bias-field)
    * [Random spatial transformations](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#random-spatial-transformations)
    * [Combined](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#combined)

</details>

<details>
<summary>Publications</summary>

* Nicholas J. Tustison, Min Chen, Fae N. Kronman, Jeffrey T. Duda, Clare Gamlin, Mia G. Tustison, Michael Kunst, Rachel Dalley, Staci Sorenson, Quanxi Wang, Lydia Ng, Yongsoo Kim, and James C. Gee.  The ANTsX Ecosystem for Mapping the Mouse Brain. [(biorxiv)](https://www.biorxiv.org/content/10.1101/2024.05.01.592056v1)

* Nicholas J. Tustison, Michael A. Yassa, Batool Rizvi, Philip A. Cook, Andrew J. Holbrook, Mithra Sathishkumar, Mia G. Tustison, James C. Gee, James R. Stone, and Brian B. Avants. ANTsX neuroimaging-derived structural phenotypes of UK Biobank. _Scientific Reports_, 14(1):8848, Apr 2024. [(pubmed)](https://pubmed.ncbi.nlm.nih.gov/38632390/)

* Nicholas J. Tustison,  Talissa A. Altes, Kun Qing, Mu He, G. Wilson Miller, Brian B. Avants, Yun M. Shim, James C. Gee, John P. Mugler III, and Jaime F. Mata.  Image- versus histogram-based considerations in semantic segmentation of pulmonary hyperpolarized gas images. _Magnetic Resonance in Medicine_, 86(5):2822-2836, Nov 2021. [(pubmed)](https://pubmed.ncbi.nlm.nih.gov/34227163/)

* Andrew T. Grainger, Arun Krishnaraj, Michael H. Quinones, Nicholas J. Tustison, Samantha Epstein, Daniela Fuller, Aakash Jha, Kevin L. Allman, Weibin Shi. Deep Learning-based Quantification of Abdominal Subcutaneous and Visceral Fat Volume on CT Images, _Academic Radiology_, 28(11):1481-1487, Nov 2021.  [(pubmed)](https://pubmed.ncbi.nlm.nih.gov/32771313/) 

* Nicholas J. Tustison, Philip A. Cook, Andrew J. Holbrook, Hans J. Johnson, John Muschelli, Gabriel A. Devenyi, Jeffrey T. Duda, Sandhitsu R. Das, Nicholas C. Cullen, Daniel L. Gillen, Michael A. Yassa, James R. Stone, James C. Gee, and Brian B. Avants for the Alzheimer’s Disease Neuroimaging Initiative.  The ANTsX ecosystem for quantitative biological and medical imaging. _Scientific Reports_.  11(1):9068, Apr 2021. [(pubmed)](https://pubmed.ncbi.nlm.nih.gov/33907199/)

* Nicholas J. Tustison, Brian B. Avants, and James C. Gee. Learning image-based spatial transformations via convolutional neural networks: a review,  _Magnetic Resonance Imaging_, 64:142-153, Dec 2019.  [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/31200026)

* Nicholas J. Tustison, Brian B. Avants, Zixuan Lin, Xue Feng, Nicholas Cullen, Jaime F. Mata, Lucia Flors, James C. Gee, Talissa A. Altes, John P. Mugler III, and Kun Qing.  Convolutional Neural Networks with Template-Based Data Augmentation for Functional Lung Image Quantification, _Academic Radiology_, 26(3):412-423, Mar 2019. [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/30195415)

* Andrew T. Grainger, Nicholas J. Tustison, Kun Qing, Rene Roy, Stuart S. Berr, and Weibin Shi.  Deep learning-based quantification of abdominal fat on magnetic resonance images. _PLoS One_, 13(9):e0204071, Sep 2018.  [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/30235253)

* Cullen N.C., Avants B.B. (2018) Convolutional Neural Networks for Rapid and Simultaneous Brain Extraction and Tissue Segmentation. In: Spalletta G., Piras F., Gili T. (eds) Brain Morphometry. Neuromethods, vol 136. Humana Press, New York, NY [doi](https://doi.org/10.1007/978-1-4939-7647-8_2)

</details>

<details>
<summary>Acknowledgements</summary>

* We gratefully acknowledge the support of the NVIDIA Corporation with the donation of two Titan Xp GPUs used for this research.

* We gratefully acknowledge the grant support of the [Office of Naval Research](https://www.onr.navy.mil) and [Cohen Veterans Bioscience](https://www.cohenveteransbioscience.org).

</details>
