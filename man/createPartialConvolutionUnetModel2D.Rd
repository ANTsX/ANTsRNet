% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/createPartialConvolutionUnetModel.R
\name{createPartialConvolutionUnetModel2D}
\alias{createPartialConvolutionUnetModel2D}
\title{2-D implementation of the U-net architecture for inpainting using partial
convolution.}
\usage{
createPartialConvolutionUnetModel2D(
  inputImageSize,
  numberOfPriors = 0,
  numberOfFilters = c(64, 128, 256, 512, 512, 512, 512, 512),
  kernelSize = c(7, 5, 5, 3, 3, 3, 3, 3),
  usePartialConv = TRUE
)
}
\arguments{
\item{inputImageSize}{Used for specifying the input tensor shape.  The
shape (or dimension) of that tensor is the image dimensions followed by
the number of channels (e.g., red, green, and blue).  The batch size
(i.e., number of training images) is not specified a priori.}

\item{numberOfPriors}{Specify tissue priors for use during the decoding branch.
Default = 0.}

\item{numberOfFilters}{vector explicitly setting the number of filters at
each layer.  Defaults to number used in the paper.}

\item{kernelSize}{single scalar or tuple of same length as the number of filters.
Specifies the kernel size schedule for the encoding path.  Defaults to the
kernel sizes used in the paper.}

\item{usePartialConv}{boolean.  Testing.  Switch between vanilla convolution
layers and partial convolution layers.}
}
\value{
a u-net keras model
}
\description{
\if{html}{\out{<div class="sourceCode">}}\preformatted{    \url{https://arxiv.org/abs/1804.07723}
}\if{html}{\out{</div>}}
}
\author{
Tustison NJ
}
