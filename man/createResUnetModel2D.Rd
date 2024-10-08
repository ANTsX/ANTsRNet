% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/createResUnetModel.R
\name{createResUnetModel2D}
\alias{createResUnetModel2D}
\title{2-D implementation of the Resnet + U-net deep learning architecture.}
\usage{
createResUnetModel2D(
  inputImageSize,
  numberOfOutputs = 1,
  numberOfFiltersAtBaseLayer = 32,
  bottleNeckBlockDepthSchedule = c(3, 4),
  convolutionKernelSize = c(3, 3),
  deconvolutionKernelSize = c(2, 2),
  dropoutRate = 0,
  weightDecay = 1e-04,
  mode = c("classification", "regression")
)
}
\arguments{
\item{inputImageSize}{Used for specifying the input tensor shape.  The
shape (or dimension) of that tensor is the image dimensions followed by
the number of channels (e.g., red, green, and blue).  The batch size
(i.e., number of training images) is not specified a priori.}

\item{numberOfOutputs}{Meaning depends on the \code{mode}.  For
'classification' this is the number of segmentation labels.  For 'regression'
this is the number of outputs.}

\item{numberOfFiltersAtBaseLayer}{number of filters at the beginning and end
of the \verb{'U'}.  Doubles at each descending/ascending layer.}

\item{bottleNeckBlockDepthSchedule}{vector that provides the encoding layer
schedule for the number of bottleneck blocks per long skip connection.}

\item{convolutionKernelSize}{2-d vector defining the kernel size
during the encoding path}

\item{deconvolutionKernelSize}{2-d vector defining the kernel size
during the decoding}

\item{dropoutRate}{float between 0 and 1 to use between dense layers.}

\item{weightDecay}{weighting parameter for L2 regularization of the
kernel weights of the convolution layers.  Default = 0.0.}

\item{mode}{'classification' or 'regression'.}
}
\value{
a res/u-net keras model
}
\description{
Creates a keras model of the U-net + ResNet deep learning architecture for
image segmentation and regression with the paper available here:
}
\details{
\if{html}{\out{<div class="sourceCode">}}\preformatted{    \url{https://arxiv.org/abs/1608.04117}
}\if{html}{\out{</div>}}

This particular implementation was ported from the following python
implementation:

\if{html}{\out{<div class="sourceCode">}}\preformatted{    \url{https://github.com/veugene/fcn_maker/}
}\if{html}{\out{</div>}}
}
\examples{

library( ANTsR )
library( ANTsRNet )
library( keras )

imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
trainingBatchSize <- length( imageIDs )

# Perform simple 3-tissue segmentation.

segmentationLabels <- c( 1, 2, 3 )
numberOfLabels <- length( segmentationLabels )
initialization <- paste0( 'KMeans[', numberOfLabels, ']' )

domainImage <- antsImageRead( getANTsRData( imageIDs[1] ) )

X_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ), 1 ) )
Y_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ) ) )

images <- list()
segmentations <- list()

for( i in seq_len( trainingBatchSize ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  image <- antsImageRead( getANTsRData( imageIDs[i] ) )
  mask <- getMask( image )
  segmentation <- atropos( image, mask, initialization )$segmentation

  X_train[i,,, 1] <- as.array( image )
  Y_train[i,,] <- as.array( segmentation )
  }
Y_train <- encodeUnet( Y_train, segmentationLabels )

# Perform a simple normalization

X_train <- ( X_train - mean( X_train ) ) / sd( X_train )

# Create the model

model <- createResUnetModel2D( c( dim( domainImage ), 1 ),
  numberOfOutputs = numberOfLabels )
rm(domainImage); gc()

metric_multilabel_dice_coefficient <-
  custom_metric( "multilabel_dice_coefficient",
    multilabel_dice_coefficient )

loss_dice <- function( y_true, y_pred ) {
  -multilabel_dice_coefficient(y_true, y_pred)
}
attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"

model \%>\% compile( loss = loss_dice,
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = c( metric_multilabel_dice_coefficient,
    metric_categorical_crossentropy ) )

# Comment out the rest due to travis build constraints

# Fit the model

# track <- model \%>\% fit( X_train, Y_train,
#              epochs = 100, batch_size = 4, verbose = 1, shuffle = TRUE,
#              callbacks = list(
#                callback_model_checkpoint( "resUnetModelInterimWeights.h5",
#                    monitor = 'val_loss', save_best_only = TRUE ),
#                callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#              ),
#              validation_split = 0.2 )
rm(X_train); gc()
rm(Y_train); gc()
# Save the model and/or save the model weights

# save_model_hdf5( model, filepath = 'resUnetModel.h5' )
# save_model_weights_hdf5( unetModel, filepath = 'resUnetModelWeights.h5' ) )
rm(model); gc()

}
\author{
Tustison NJ
}
