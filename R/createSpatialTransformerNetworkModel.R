#' 2-D implementation of the spatial transformer network.
#'
#' Creates a keras model of the spatial transformer network:
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
#' @param resampledSize resampled size of the transformed input images.
#' @param numberOfClassificationLabels Number of classes.
#'
#' @return a keras model
#' @author Tustison NJ
#' @examples
#' \dontrun{
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
#' model <- createSpatialTransformerNetworkModel2D( inputImageSize = inputImageSize,
#'   resampledSize = c( 30, 30 ), numberOfClassificationLabels = numberOfLabels )
#'
#' }
#' @import keras
#' @export
createSpatialTransformerNetworkModel2D <- function( inputImageSize,
  resampledSize = c( 30, 30 ), numberOfClassificationLabels = 10 )
{

  getInitialWeights2D <- function( outputSize )
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

    return( weights )
    }

  inputs <- layer_input( shape = inputImageSize )

  localization <- inputs
  localization <- localization %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  localization <- localization %>% layer_conv_2d( filters = 20, kernel_size = c( 5, 5 ) )
  localization <- localization %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  localization <- localization %>% layer_conv_2d( filters = 20, kernel_size = c( 5, 5 ) )

  localization <- localization %>% layer_flatten()
  localization <- localization %>% layer_dense( units = 50L )
  localization <- localization %>% layer_activation( 'relu' )

  weights <- getInitialWeights2D( outputSize = 50L )
  localization <- localization %>% layer_dense( units = 6L, weights = weights )

  outputs <- layer_spatial_transformer_2d( list( inputs, localization ),
    resampledSize, transformType = 'affine', interpolatorType = 'linear',
    name = "layer_spatial_transformer" )
  outputs <- outputs %>%
    layer_conv_2d( filters = 32L, kernel_size = c( 3, 3 ), padding = 'same' )
  outputs <- outputs %>% layer_activation( activation = "relu" )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  outputs <- outputs %>% layer_conv_2d( filters = 32L, kernel_size = c( 3, 3 ) )
  outputs <- outputs %>% layer_activation( activation = "relu" )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = 256L )
  outputs <- outputs %>% layer_activation( activation = "relu" )
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels )

  outputs <- outputs %>% layer_activation('softmax')

  stnModel <- keras_model( inputs = inputs, outputs = outputs )

  return( stnModel )
}

#' 3-D implementation of the spatial transformer network.
#'
#' Creates a keras model of the spatial transformer network:
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
#' @param resampledSize resampled size of the transformed input images.
#' @param numberOfClassificationLabels Number of classes.
#'
#' @return a keras model
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{
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
#' model <- createSpatialTransformerNetworkModel2D( inputImageSize = inputImageSize,
#'   resampledSize = c( 30, 30 ), numberOfClassificationLabels = numberOfLabels )
#'
#'}
#' @import keras
#' @export
createSpatialTransformerNetworkModel3D <- function( inputImageSize,
  resampledSize = c( 30, 30, 30 ), numberOfClassificationLabels = 10 )
{

  getInitialWeights3D <- function( outputSize )
    {
    np <- reticulate::import( "numpy" )

    b <- np$zeros( c( 3L, 4L ), dtype = "float32" )
    b[1, 1] <- 1
    b[2, 2] <- 1
    b[3, 3] <- 1

    W <- np$zeros( c( as.integer( outputSize ), 12L ), dtype = 'float32' )

    # Layer weights in R keras are stored as lists of length 2 (W & b)
    weights <- list()
    weights[[1]] <- W
    weights[[2]] <- as.array( as.vector( t( b ) ) )

    return( weights )
    }

  inputs <- layer_input( shape = inputImageSize )

  localization <- inputs
  localization <- localization %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
  localization <- localization %>% layer_conv_3d( filters = 20, kernel_size = c( 5, 5, 5 ) )
  localization <- localization %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
  localization <- localization %>% layer_conv_3d( filters = 20, kernel_size = c( 5, 5, 5 ) )

  localization <- localization %>% layer_flatten()
  localization <- localization %>% layer_dense( units = 50L )
  localization <- localization %>% layer_activation( 'relu' )

  weights <- getInitialWeights3D( outputSize = 50L )
  localization <- localization %>% layer_dense( units = 12L, weights = weights )

  outputs <- layer_spatial_transformer_3d( list( inputs, localization ),
    resampledSize, transformType = 'affine', interpolatorType = 'linear',
    name = "layer_spatial_transformer" )
  outputs <- outputs %>%
    layer_conv_3d( filters = 32L, kernel_size = c( 3, 3, 3 ), padding = 'same' )
  outputs <- outputs %>% layer_activation( activation = "relu" )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
  outputs <- outputs %>% layer_conv_3d( filters = 32L, kernel_size = c( 3, 3, 3 ) )
  outputs <- outputs %>% layer_activation( activation = "relu" )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = 256L )
  outputs <- outputs %>% layer_activation( activation = "relu" )
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels )

  outputs <- outputs %>% layer_activation_softmax()

  stnModel <- keras_model( inputs = inputs, outputs = outputs )

  return( stnModel )
}
