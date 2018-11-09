#' 2-D implementation of the GoogLeNet deep learning architecture.
#'
#' Creates a keras model of the GoogLeNet deep learning architecture for image
#' recognition based on the paper
#'
#' C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke,
#'   A. Rabinovich, Going Deeper with Convolutions
#' C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception
#'   Architecture for Computer Vision
#'
#' available here:
#'
#'         https://arxiv.org/abs/1409.4842
#'         https://arxiv.org/abs/1512.00567
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return a GoogLeNet keras model
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
#' model <- createGoogLeNetModel2D( inputImageSize = inputImageSize,
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
#' @import keras
#' @export
createGoogLeNetModel2D <- function( inputImageSize,
                                    numberOfClassificationLabels = 1000,
                                    mode = 'classification'
                                  )
{
  K <- keras::backend()

  convolutionAndBatchNormalization2D <- function( model,
                                                  numberOfFilters,
                                                  kernelSize,
                                                  padding = 'same',
                                                  strides = c( 1, 1 ) )
    {
    K <- keras::backend()

    channelAxis <- 1
    if( K$image_data_format() == 'channels_last' )
      {
      channelAxis <- 3
      }

    model <- model %>% layer_conv_2d( numberOfFilters,
      kernel_size = kernelSize, padding = padding, strides = strides,
      use_bias = TRUE )
    model <- model %>% layer_batch_normalization( axis = channelAxis,
      scale = FALSE )
    model <- model %>% layer_activation( activation = 'relu' )

    return( model )
    }

  channelAxis <- 1
  if( K$image_data_format() == 'channels_last' )
    {
    channelAxis <- 3
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- convolutionAndBatchNormalization2D( inputs, numberOfFilters = 32,
    kernelSize = c( 3, 3 ), strides = c( 2, 2 ), padding = 'valid' )
  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 32,
    kernelSize = c( 3, 3 ), padding = 'valid' )
  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 3, 3 ) )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )

  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 80,
    kernelSize = c( 1, 1 ), padding = 'valid' )
  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 3, 3 ) )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )

  # mixed 0, 1, 2: 35x35x256
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 48,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 64, kernelSize = c( 5, 5 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
    numberOfFilters = 32, kernelSize = c( 1, 1 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 1: 35x35x256
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 48,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 64, kernelSize = c( 5, 5 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
    numberOfFilters = 32, kernelSize = c( 1, 1 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 2: 35x35x256
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 48,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 64, kernelSize = c( 5, 5 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
    numberOfFilters = 32, kernelSize = c( 1, 1 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 3: 17x17x768
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 384,
    kernelSize = c( 3, 3 ), strides = c( 2, 2 ), padding = 'valid' )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 96, kernelSize = c( 3, 3 ), strides = c( 2, 2 ), padding = 'valid' )
  branchLayers[[3]] <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 4: 17x17x768
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 128,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 128, kernelSize = c( 1, 7 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 128,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 128, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 128, kernelSize = c( 1, 7 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 128, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
    numberOfFilters = 192, kernelSize = c( 1, 1 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 4: 17x17x768
  for( i in 1:2 )
    {
    branchLayers <- list()
    branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
      kernelSize = c( 1, 1 ) )
    branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 160,
      kernelSize = c( 1, 1 ) )
    branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
      numberOfFilters = 160, kernelSize = c( 1, 7 ) )
    branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
      numberOfFilters = 192, kernelSize = c( 7, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 160,
      kernelSize = c( 1, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
      numberOfFilters = 160, kernelSize = c( 7, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
      numberOfFilters = 160, kernelSize = c( 1, 7 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
      numberOfFilters = 160, kernelSize = c( 7, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
      numberOfFilters = 192, kernelSize = c( 1, 7 ) )
    branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
      strides = c( 1, 1 ), padding = 'same' )
    branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
      numberOfFilters = 192, kernelSize = c( 1, 1 ) )
    outputs <- layer_concatenate( branchLayers, axis = channelAxis )
    }

  # mixed 7: 17x17x768
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]],
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
    numberOfFilters = 192, kernelSize = c( 1, 1 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 8: 8x8x1280
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( branchLayers[[1]],
    numberOfFilters = 320, kernelSize = c( 3, 3 ), strides = c( 2, 2 ),
    padding = 'valid' )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]],
    numberOfFilters = 192, kernelSize = c( 3, 3 ), strides = c( 2, 2 ),
    padding = 'valid' )
  branchLayers[[3]] <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 9: 8x8x2048
  for( i in 1:2 )
    {
    branchLayers <- list()

    branchLayer <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 320,
      kernelSize = c( 1, 1 ) )
    branchLayers[[1]] <- branchLayer

    branchLayer <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 384,
      kernelSize = c( 1, 1 ) )
    branchLayer1 <- convolutionAndBatchNormalization2D( branchLayer,
      numberOfFilters = 384, kernelSize = c( 1, 3 ) )
    branchLayer2 <- convolutionAndBatchNormalization2D( branchLayer,
      numberOfFilters = 384, kernelSize = c( 3, 1 ) )
    branchLayers[[2]] <- layer_concatenate( list( branchLayer1, branchLayer2 ),
      axis = channelAxis )

    branchLayer <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 448,
      kernelSize = c( 1, 1 ) )
    branchLayer <- convolutionAndBatchNormalization2D( branchLayer,
      numberOfFilters = 384, kernelSize = c( 3, 3 ) )
    branchLayer1 <- convolutionAndBatchNormalization2D( branchLayer,
      numberOfFilters = 384, kernelSize = c( 1, 3 ) )
    branchLayer2 <- convolutionAndBatchNormalization2D( branchLayer,
      numberOfFilters = 384, kernelSize = c( 3, 1 ) )
    branchLayers[[3]] <- layer_concatenate( list( branchLayer1, branchLayer2 ),
      axis = channelAxis )

    branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ),
      strides = c( 1, 1 ), padding = 'same' )
    branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]],
      numberOfFilters = 192, kernelSize = c( 1, 1 ) )

    outputs <- layer_concatenate( branchLayers, axis = channelAxis )
    }
  outputs <- outputs %>% layer_global_average_pooling_2d()


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

  outputs <- outputs %>%
    layer_dense( units = numberOfClassificationLabels, activation = layerActivation )

  googLeNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( googLeNetModel )
}





