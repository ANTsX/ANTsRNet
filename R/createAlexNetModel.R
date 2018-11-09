#' 2-D implementation of the AlexNet deep learning architecture.
#'
#' Creates a keras model of the AlexNet deep learning architecture for image
#' recognition based on the paper
#'
#' A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification
#'   with Deep Convolutional Neural Networks.
#'
#' available here:
#'
#'         http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/duggalrahul/AlexNet-Experiments-Keras/
#'         https://github.com/lunardog/convnets-keras/
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param numberOfDenseUnits number of dense units.
#' @param dropoutRate optional regularization parameter between \verb{[0, 1]}.
#' Default = 0.0.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#' @return an AlexNet keras model
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
#' model <- createAlexNetModel2D( inputImageSize = inputImageSize,
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
createAlexNetModel2D <- function( inputImageSize,
                                  numberOfClassificationLabels = 1000,
                                  numberOfDenseUnits = 4096,
                                  dropoutRate = 0.0,
                                  mode = 'classification'
                                )
{

  splitTensor2D <- function( axis = 4, ratioSplit = 1, idSplit = 1 )
    {
    f <- function( X )
      {
      K <- keras::backend()

      Xdims <- K$int_shape( X )
      div <- as.integer( Xdims[[axis]] / ratioSplit )
      axisSplit <- ( ( idSplit - 1 ) * div + 1 ):( idSplit * div )

      if( axis == 1 )
        {
        output <- X[axisSplit,,,]
        } else if( axis == 2 ) {
        output <- X[, axisSplit,,]
        } else if( axis == 3 ) {
        output <- X[,, axisSplit,]
        } else if( axis == 4 ) {
        output <- X[,,, axisSplit]
        } else {
        stop( "Wrong axis specification." )
        }
      return( output )
      }

    return( layer_lambda( f = f ) )
    }

  crossChannelNormalization2D <- function(
    alpha = 1e-4, k = 2, beta = 0.75, n = 5L )
    {
    normalizeTensor2D <- function( X )
      {
      K <- keras::backend()

      #  Theano:  [batchSize, channelSize, widthSize, heightSize]
      #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]

      if( K$image_data_format() == "channels_last" )
        {
        Xshape <- X$get_shape()
        } else {
        Xshape <- X$shape()
        }
      X2 <- K$square( X )

      half <- as.integer( n / 2 )

      extraChannels <- K$spatial_2d_padding(
        K$permute_dimensions( X2, c( 1L, 2L, 3L, 0L ) ),
        padding = list( c( 0L, 0L ), c( half, half ) ) )
      extraChannels <- K$permute_dimensions(
        extraChannels, c( 3L, 0L, 1L, 2L ) )
      scale <- k

      Xdims <- K$int_shape( X )
      ch <- as.integer( Xdims[[length( Xdims )]] )
      for( i in 1:n )
        {
        scale <- scale + alpha * extraChannels[,,, i:( i + ch - 1 )]
        }
      scale <- K$pow( scale, beta )

      return( X / scale )
      }

    return( layer_lambda( f = normalizeTensor2D ) )
    }

  inputs <- layer_input( shape = inputImageSize )

  # Conv1
  outputs <- inputs %>% layer_conv_2d( filters = 96,
    kernel_size = c( 11, 11 ), strides = c( 4, 4 ), activation = 'relu' )

  # Conv2
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )
  normalizationLayer <- crossChannelNormalization2D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )

  convolutionLayer <- outputs %>% layer_conv_2d( filters = 128,
    kernel_size = c( 5, 5 ), padding = 'same' )
  lambdaLayersConv2 <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayersConv2[[i+1]] <- outputs %>% splitLayer
    }
  outputs <- layer_concatenate( lambdaLayersConv2 )

  # Conv3
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )
  normalizationLayer <- crossChannelNormalization2D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )
  outputs <- outputs %>% layer_conv_2d( filters = 384,
    kernel_size = c( 3, 3 ), padding = 'same' )

  # Conv4
  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )
  convolutionLayer <- outputs %>% layer_conv_2d( filters = 192,
    kernel_size = c( 3, 3 ), padding = 'same' )
  lambdaLayersConv4 <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayersConv4[[i+1]] <- outputs %>% splitLayer
    }
  outputs <- layer_concatenate( lambdaLayersConv4 )

  # Conv5
  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )
  normalizationLayer <- crossChannelNormalization2D()
  outputs <- outputs %>% normalizationLayer

  convolutionLayer <- outputs %>% layer_conv_2d( filters = 128,
    kernel_size = c( 3, 3 ), padding = 'same' )
  lambdaLayersConv5 <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayersConv5[[i+1]] <- outputs %>% splitLayer
    }
  outputs <- layer_concatenate( lambdaLayersConv5 )

  outputs <- outputs %>%
    layer_max_pooling_2d( pool_size = c( 3, 3 ), strides = c( 2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = numberOfDenseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
    }
  outputs <- outputs %>% layer_dense( units = numberOfDenseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
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

  outputs <- outputs %>%
    layer_dense( units = numberOfClassificationLabels, activation = layerActivation )

  alexNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( alexNetModel )
}

#' 3-D implementation of the AlexNet deep learning architecture.
#'
#' Creates a keras model of the AlexNet deep learning architecture for image
#' recognition based on the paper
#'
#' A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification
#'   with Deep Convolutional Neural Networks.
#'
#' available here:
#'
#'         http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/duggalrahul/AlexNet-Experiments-Keras/
#'         https://github.com/lunardog/convnets-keras/
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param numberOfDenseUnits number of dense units.
#' @param dropoutRate optional regularization parameter between \verb{[0, 1]}.
#' Default = 0.0.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#' @return an AlexNet keras model
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
#' model <- createAlexNetModel2D( inputImageSize = inputImageSize,
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
createAlexNetModel3D <- function( inputImageSize,
                                  numberOfClassificationLabels = 1000,
                                  numberOfDenseUnits = 4096,
                                  dropoutRate = 0.0,
                                  mode = 'classification'
                                )
{

  splitTensor3D <- function( axis = 5, ratioSplit = 1, idSplit = 1 )
    {
    f <- function( X )
      {
      K <- keras::backend()

      Xdims <- K$int_shape( X )
      div <- as.integer( Xdims[[axis]] / ratioSplit )
      axisSplit <- ( ( idSplit - 1 ) * div + 1 ):( idSplit * div )

      if( axis == 1 )
        {
        output <- X[axisSplit,,,,]
        } else if( axis == 2 ) {
        output <- X[, axisSplit,,,]
        } else if( axis == 3 ) {
        output <- X[,, axisSplit,,]
        } else if( axis == 4 ) {
        output <- X[,,, axisSplit,]
        } else if( axis == 5 ) {
        output <- X[,,,, axisSplit]
        } else {
        stop( "Wrong axis specification." )
        }
      return( output )
      }

    return( layer_lambda( f = f ) )
    }

  crossChannelNormalization3D <- function(
    alpha = 1e-4, k = 2, beta = 0.75, n = 5L )
    {
    normalizeTensor3D <- function( X )
      {
      K <- keras::backend()
      #  Theano:  [batchSize, channelSize, widthSize, heightSize, depthSize]
      #  tensorflow:  [batchSize, widthSize, heightSize, depthSize, channelSize]

      if( K$image_data_format() == "channels_last" )
        {
        Xshape <- X$get_shape()
        } else {
        Xshape <- X$shape()
        }
      X2 <- K$square( X )

      half <- as.integer( n / 2 )

      extraChannels <- K$spatial_3d_padding(
        K$permute_dimensions( X2, c( 1L, 2L, 3L, 4L, 0L ) ),
        padding = list( c( 0L, 0L ), c( 0L, 0L ), c( half, half ) ) )
      extraChannels <- K$permute_dimensions(
        extraChannels, c( 4L, 0L, 1L, 2L, 3L ) )
      scale <- k

      Xdims <- K$int_shape( X )
      ch <- as.integer( Xdims[[length( Xdims )]] )
      for( i in 1:n )
        {
        scale <- scale + alpha * extraChannels[,,,, i:( i + ch - 1 )]
        }
      scale <- K$pow( scale, beta )

      return( X / scale )
      }

    return( layer_lambda( f = normalizeTensor3D ) )
    }

  inputs <- layer_input( shape = inputImageSize )

  # Conv1
  outputs <- inputs %>% layer_conv_3d( filters = 96,
    kernel_size = c( 11, 11, 11 ), strides = c( 4, 4, 4 ), activation = 'relu' )

  # Conv2
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ) )
  normalizationLayer <- crossChannelNormalization3D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )

  convolutionLayer <- outputs %>% layer_conv_3d( filters = 128,
    kernel_size = c( 5, 5, 5 ), padding = 'same' )
  lambdaLayersConv2 <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor3D( axis = 5, ratioSplit = 2, idSplit = i )
    lambdaLayersConv2[[i+1]] <- outputs %>% splitLayer
    }
  outputs <- layer_concatenate( lambdaLayersConv2 )

  # Conv3
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ) )
  normalizationLayer <- crossChannelNormalization3D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )
  outputs <- outputs %>% layer_conv_3d( filters = 384,
    kernel_size = c( 3, 3, 3 ), padding = 'same' )

  # Conv4
  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )
  convolutionLayer <- outputs %>% layer_conv_3d( filters = 192,
    kernel_size = c( 3, 3, 3 ), padding = 'same' )
  lambdaLayersConv4 <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor3D( axis = 5, ratioSplit = 2, idSplit = i )
    lambdaLayersConv4[[i+1]] <- outputs %>% splitLayer
    }
  outputs <- layer_concatenate( lambdaLayersConv4 )

  # Conv5
  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )
  normalizationLayer <- crossChannelNormalization3D()
  outputs <- outputs %>% normalizationLayer

  convolutionLayer <- outputs %>% layer_conv_3d( filters = 128,
    kernel_size = c( 3, 3, 3 ), padding = 'same' )
  lambdaLayersConv5 <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor3D( axis = 5, ratioSplit = 2, idSplit = i )
    lambdaLayersConv5[[i+1]] <- outputs %>% splitLayer
    }
  outputs <- layer_concatenate( lambdaLayersConv5 )

  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = numberOfDenseUnits,
    activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
    }
  outputs <- outputs %>% layer_dense( units = numberOfDenseUnits,
    activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
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

  outputs <- outputs %>%
    layer_dense( units = numberOfClassificationLabels, activation = layerActivation )

  alexNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( alexNetModel )
}
