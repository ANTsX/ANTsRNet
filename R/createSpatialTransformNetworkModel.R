#' 2-D implementation of the spatial transform network.
#'
#' Creates a keras model of the spatial transform network:
#'
#'         \url{https://arxiv.org/abs/1506.02025}
#'
#' based on the following python Keras model:
#'
#'         \url{https://github.com/oarriaga/STN.keras/blob/master/src/models/STN.py}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param samplingSize a list of 2-D vectors specifying the kernel
#' size at each convolution layer.  Default values are the same as given in
#' the original paper.  The length of kernel size vectors must be 1 greater
#' than the vector length of the number of filters.
#' @param numberOfClassificationLabels Number of classes.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' mnistData <- dataset_mnist()
#' numberOfLabels <- 10
#'
#' # Extract a small subset for something that can run quickly
#'
#' X_trainSmall <- mnistData$train$x[1:100,,]
#' X_trainSmall <- array( data = X_trainSmall, dim = c( dim( X_trainSmall ), 1 ) )
#' Y_trainSmall <- to_categorical( mnistData$train$y[1:100], numberOfLabels )
#'
#' X_testSmall <- mnistData$test$x[1:10,,]
#' X_testSmall <- array( data = X_testSmall, dim = c( dim( X_testSmall ), 1 ) )
#' Y_testSmall <- to_categorical( mnistData$test$y[1:10], numberOfLabels )
#'
#' # We add a dimension of 1 to specify the channel size
#'
#' inputImageSize <- c( dim( X_trainSmall )[2:3], 1 )
#'
#' model <- createSpatialTransformNetworkModel2D( inputImageSize = inputImageSize,
#'   numberOfClassificationLabels = numberOfLabels )
#'
#' @import keras
#' @export
createSpatialTransformNetworkModel2D <- function( inputImageSize,
  resampledSize = c( 10, 10 ), numberOfClassificationLabels = 10 )
{

  getInitialWeights <- function( outputSize )
    {
    np <- reticulate::import( "numpy" )

    b <- np$zeros( c( 2L, 3L ), dtype = "float32" )
    b[1, 1] <- 1
    b[2, 2] <- 1

    W <- np$zeros( c( as.integer( outputSize ), 6L ), dtype = 'float32' )

    # Layer weights in R keras are stored as lists of length 2 (W & b)
    weights <- list()
    weights[[1]] <- W
    weights[[2]] <- as.array( as.vector( t( b ) ) )
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  outputs <- outputs %>% layer_conv_2d( filters = 20, kernel_size = c( 5, 5 ) )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  outputs <- outputs %>% layer_conv_2d( filters = 20, kernel_size = c( 5, 5 ) )

  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = 50 )
  outputs <- outputs %>% layer_activation_relu()

  weights <- getInitialWeights( outputSize = 50L )
  outputs <- outputs %>% layer_dense( units = 6, weights = weights )

  outputs <- layer_bilinear_interpolation_2d( list( inputs, outputs ), resampledSize )
  outputs <- outputs %>% layer_conv_2d( filters = 32, kernel_size = c( 3, 3 ) )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = 256 )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels )

  outputs <- outputs %>% layer_activation_softmax()

  stnModel <- keras_model( inputs = inputs, outputs = outputs )

  return( stnModel )
}

