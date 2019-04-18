#' 2-D implementation of the Vgg deep learning architecture.
#'
#' Creates a keras model of the Vgg deep learning architecture for image
#' recognition based on the paper
#'
#' K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for
#'   Large-Scale Image Recognition
#'
#' available here:
#'
#'         \url{https://arxiv.org/abs/1409.1556}
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         \url{https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param layers a vector determining the number of filters defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning.
#' @param convolutionKernelSize 2-d vector definining the kernel size
#' during the encoding path
#' @param poolSize 2-d vector defining the region for each pooling layer.
#' @param strides 2-d vector describing the stride length in each direction.
#' @param denseUnits integer for the number of units in the last layers.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param style \verb{'16'} or \verb{'19'} for VGG16 or VGG19, respectively.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return a VGG keras model to be used with subsequent fitting
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#' library( ANTsR )
#'
#' mnistData <- dataset_mnist()
#' numberOfLabels <- 10
#'
#' # Extract a small subset for something that can run quickly.
#' # We also need to resample since the native mnist data size does
#' # not fit with GoogLeNet parameters.
#'
#' resampledImageSize <- c( 100, 100 )
#' numberOfTrainingData <- 10
#' numberOfTestingData <- 5
#'
#' X_trainSmall <- as.array(
#'   resampleImage( as.antsImage( mnistData$train$x[1:numberOfTrainingData,,] ),
#'     c( numberOfTrainingData, resampledImageSize ), TRUE ) )
#' X_trainSmall <- array( data = X_trainSmall, dim = c( dim( X_trainSmall ), 1 ) )
#' Y_trainSmall <- to_categorical( mnistData$train$y[1:numberOfTrainingData], numberOfLabels )
#'
#' X_testSmall <- as.array(
#'   resampleImage( as.antsImage( mnistData$test$x[1:numberOfTestingData,,] ),
#'     c( numberOfTestingData, resampledImageSize ), TRUE ) )
#' X_testSmall <- array( data = X_testSmall, dim = c( dim( X_testSmall ), 1 ) )
#' Y_testSmall <- to_categorical( mnistData$test$y[1:numberOfTestingData], numberOfLabels )
#'
#' # We add a dimension of 1 to specify the channel size
#'
#' inputImageSize <- c( dim( X_trainSmall )[2:3], 1 )
#'
#' model <- createVggModel2D( inputImageSize = c( resampledImageSize, 1 ),
#'   numberOfClassificationLabels = numberOfLabels )
#'
#' model %>% compile( loss = 'categorical_crossentropy',
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( 'categorical_crossentropy', 'accuracy' ) )
#'
#' # Comment out the rest due to travis build constraints
#'
#' # track <- model %>% fit( X_trainSmall, Y_trainSmall, verbose = 1,
#' #   epochs = 1, batch_size = 2, shuffle = TRUE, validation_split = 0.5 )
#'
#' # Now test the model
#'
#' # testingMetrics <- model %>% evaluate( X_testSmall, Y_testSmall )
#' # predictedData <- model %>% predict( X_testSmall, verbose = 1 )
#'
#' @import keras
#' @export
createVggModel2D <- function( inputImageSize,
                               numberOfClassificationLabels = 1000,
                               layers = c( 1, 2, 3, 4, 4 ),
                               lowestResolution = 64,
                               convolutionKernelSize = c( 3, 3 ),
                               poolSize = c( 2, 2 ),
                               strides = c( 2, 2 ),
                               denseUnits = 4096,
                               dropoutRate = 0.0,
                               style = 19,
                               mode = 'classification'
                             )
{

  if( style != 19 && style != 16 )
    {
    stop( "Incorrect style.  Must be either '16' or '19'." )
    }

  vggModel <- keras_model_sequential()

  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

    if( i == 1 )
      {
      vggModel %>%
        layer_conv_2d( input_shape = inputImageSize, filters = numberOfFilters,
                      kernel_size = convolutionKernelSize, activation = 'relu',
                      padding = 'same' ) %>%
        layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                      activation = 'relu', padding = 'same' )
      } else if( i == 2 ) {
      vggModel %>%
        layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                      activation = 'relu', padding = 'same' ) %>%
        layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                      activation = 'relu', padding = 'same' )
      }  else {
      if( style == 16 )
        {
        vggModel %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' )
        } else {  # style == 19
        vggModel %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' )
        }
      }

    vggModel %>% layer_max_pooling_2d( pool_size = poolSize, strides = strides )
    }

  vggModel %>% layer_flatten()
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }

  layerActivation <- ''
  if( mode == 'classification' )
    {
    if( numberOfClassificationLabels == 2 )
      {
      layerActivation <- 'sigmoid'
      } else {
      layerActivation <- 'softmax'
      }
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  vggModel %>% layer_dense( units = numberOfClassificationLabels,
    activation = layerActivation )

  return( vggModel )
}

#' 3-D implementation of the Vgg deep learning architecture.
#'
#' Creates a keras model of the Vgg deep learning architecture for image
#' recognition based on the paper
#'
#' K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for
#'   Large-Scale Image Recognition
#'
#' available here:
#'
#'         \url{https://arxiv.org/abs/1409.1556}
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         \url{https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning.
#' @param convolutionKernelSize 3-d vector definining the kernel size
#' during the encoding path
#' @param poolSize 3-d vector defining the region for each pooling layer.
#' @param strides 3-d vector describing the stride length in each direction.
#' @param denseUnits integer for the number of units in the last layers.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param style \verb{'16'} or \verb{'19'} for VGG16 or VGG19, respectively.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return a VGG keras model to be used with subsequent fitting
#' @author Tustison NJ
#' @examples
#' # Simple examples, must run successfully and quickly. These will be tested.
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
#' model <- createVggModel2D( inputImageSize = inputImageSize,
#'   numberOfClassificationLabels = numberOfLabels )
#'
#' model %>% compile( loss = 'categorical_crossentropy',
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( 'categorical_crossentropy', 'accuracy' ) )
#'
#' track <- model %>% fit( X_trainSmall, Y_trainSmall, verbose = 1,
#'   epochs = 2, batch_size = 20, shuffle = TRUE, validation_split = 0.25 )
#'
#' # Now test the model
#'
#' testingMetrics <- model %>% evaluate( X_testSmall, Y_testSmall )
#' predictedData <- model %>% predict( X_testSmall, verbose = 1 )
#'
#' }
#' @import keras
#' @export
createVggModel3D <- function( inputImageSize,
                               numberOfClassificationLabels = 1000,
                               layers = c( 1, 2, 3, 4, 4 ),
                               lowestResolution = 64,
                               convolutionKernelSize = c( 3, 3, 3 ),
                               poolSize = c( 2, 2, 2 ),
                               strides = c( 2, 2, 2 ),
                               denseUnits = 4096,
                               dropoutRate = 0.0,
                               style = 19,
                               mode = 'classification'
                             )
{

  if( style != 19 && style != 16 )
    {
    stop( "Incorrect style.  Must be either '16' or '19'." )
    }

  vggModel <- keras_model_sequential()

  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

    if( i == 1 )
      {
      vggModel %>%
        layer_conv_3d( input_shape = inputImageSize, filters = numberOfFilters,
                      kernel_size = convolutionKernelSize, activation = 'relu',
                      padding = 'same' ) %>%
        layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                      activation = 'relu', padding = 'same' )
      } else if( i == 2 ) {
      vggModel %>%
        layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                      activation = 'relu', padding = 'same' ) %>%
        layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                      activation = 'relu', padding = 'same' )
      }  else {
      if( style == 16 )
        {
        vggModel %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' )
        } else {  # style == 19
        vggModel %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                        activation = 'relu', padding = 'same' )
        }
      }

    vggModel %>% layer_max_pooling_3d( pool_size = poolSize, strides = strides )
    }

  vggModel %>% layer_flatten()
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }

  layerActivation <- ''
  if( mode == 'classification' )
    {
    if( numberOfClassificationLabels == 2 )
      {
      layerActivation <- 'sigmoid'
      } else {
      layerActivation <- 'softmax'
      }
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  vggModel %>% layer_dense( units = numberOfClassificationLabels,
    activation = layerActivation )

  return( vggModel )
}

