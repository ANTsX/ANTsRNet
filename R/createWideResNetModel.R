#' 2-D implementation of the Wide ResNet deep learning architecture.
#'
#' Creates a keras model of the Wide ResNet deep learning architecture for image
#' classification/regression.  The paper is available here:
#'
#'         https://arxiv.org/abs/1512.03385
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/titu1994/Wide-Residual-Networks
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfOutputs Number of outputs in the final layer
#' @param depth integer determining the depth of the network.  Related to the
#' actual number of layers by the \code{numberOfLayers = depth * 6 + 4}.
#' Default = 2 (such that \code{numberOfLayers = 16}.)
#' @param width integer determining the width of the network.  Default = 1.
#' @param residualBlockSchedule vector determining the number of filters
#' per convolutional block. Default = \code{c( 16, 32, 64 )}.
#' @param dropoutRate Dropout percentage.  Default = 0.0.
#' @param weightDecay weight for l2 regularizer in convolution layers.
#' Default = 0.0005.
#' @param poolSize pool size for final average pooling layer.  Default = c( 8, 8 ).
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return a Wide ResNet keras model
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
#' X_trainSmall <- mnistData$train$x[1:10,,]
#' X_trainSmall <- array( data = X_trainSmall, dim = c( dim( X_trainSmall ), 1 ) )
#' Y_trainSmall <- to_categorical( mnistData$train$y[1:10], numberOfLabels )
#'
#' X_testSmall <- mnistData$test$x[1:10,,]
#' X_testSmall <- array( data = X_testSmall, dim = c( dim( X_testSmall ), 1 ) )
#' Y_testSmall <- to_categorical( mnistData$test$y[1:10], numberOfLabels )
#'
#' # We add a dimension of 1 to specify the channel size
#'
#' inputImageSize <- c( dim( X_trainSmall )[2:3], 1 )
#'
#' model <- createWideResNetModel2D( inputImageSize = inputImageSize,
#'   numberOfOutputs = numberOfLabels )
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
#' rm(model); gc()
#' @import keras
#' @export
createWideResNetModel2D <- function( inputImageSize,
                                     numberOfOutputs = 1000,
                                     depth = 2,
                                     width = 1,
                                     residualBlockSchedule = c( 16, 32, 64 ),
                                     poolSize = c( 8, 8 ),
                                     dropoutRate = 0.0,
                                     weightDecay = 0.0005,
                                     mode = 'classification'
                                   )
{
  mode <- match.arg( mode )

  channelAxis <- 1
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1
    }

  initialConvolutionLayer <- function( model, numberOfFilters )
    {
    model <- model %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = c( 3, 3 ), padding = 'same',
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    return( model )
    }

  customConvolutionLayer <- function( initialModel, base, width, strides = c( 1, 1 ),
    dropoutRate = 0.0, expand = TRUE )
    {
    numberOfFilters <- as.integer( base * width )

    if( expand == TRUE )
      {
      model <- initialModel %>% layer_conv_2d( filters = numberOfFilters, kernel_size = c( 3, 3 ),
        padding = 'same', strides = strides, kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )
      } else {
      model <- initialModel
      }

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    model <- model %>% layer_conv_2d( filters = numberOfFilters, kernel_size = c( 3, 3 ),
      padding = 'same', kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

    if( expand == TRUE )
      {
      skipLayer <- initialModel %>% layer_conv_2d( filters = numberOfFilters,
        kernel_size = c( 1, 1 ), padding = 'same', strides = strides,
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( model, skipLayer ) )
      } else {
      if( dropoutRate > 0.0 )
        {
        model <- model %>% layer_dropout( rate = dropoutRate )
        }

      model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
        epsilon = 1.0e-5, gamma_initializer = "uniform" )
      model <- model %>% layer_activation( activation = "relu" )

      model <- model %>% layer_conv_2d( filters = numberOfFilters, kernel_size = c( 3, 3 ),
        padding = 'same', kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( initialModel, model ) )
      }

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- initialConvolutionLayer( inputs, residualBlockSchedule[1] )
  numberOfConvolutions <- 4

  for( i in seq_len( length( residualBlockSchedule ) ) )
    {
    baseNumberOfFilters <- residualBlockSchedule[i]

    outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters, width = width,
      strides = c( 1, 1 ), dropoutRate = 0.0, expand = TRUE )
    numberOfConvolutions <- numberOfConvolutions + 2

    for( j in seq_len( depth ) )
      {
      outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters,
        width = width, dropoutRate = dropoutRate, expand = FALSE )
      numberOfConvolutions <- numberOfConvolutions + 2
      }

    outputs <- outputs %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    outputs <- outputs %>% layer_activation( activation = "relu" )
    }

  outputs <- outputs %>% layer_average_pooling_2d( pool_size = poolSize )
  outputs <- outputs %>% layer_flatten()

  layerActivation <- ''
  if( mode == 'classification' ) {
    layerActivation <- 'softmax'  
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'  
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>% layer_dense( units = numberOfOutputs,
    kernel_regularizer = regularizer_l2( weightDecay ), activation = layerActivation )

  wideResNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( wideResNetModel )
}

#' 3-D implementation of the Wide ResNet deep learning architecture.
#'
#' Creates a keras model of the Wide ResNet deep learning architecture for image
#' classification/regression.  The paper is available here:
#'
#'         https://arxiv.org/abs/1512.03385
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/titu1994/Wide-Residual-Networks
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfOutputs Number of classification labels.
#' @param depth integer determining the depth of the ntwork.  Related to the
#' actual number of layers by the \code{numberOfLayers = depth * 6 + 4}.
#' Default = 2 (such that \code{numberOfLayers = 16}.)
#' @param width integer determining the width of the network.  Default = 1.
#' @param residualBlockSchedule vector determining the number of filters
#' per convolutional block. Default = \code{c( 16, 32, 64 )}.
#' @param dropoutRate Dropout percentage.  Default = 0.0.
#' @param weightDecay weight for l2 regularizer in convolution layers.
#' Default = 0.0005.
#' @param poolSize pool size for final average pooling layer.  Default = c( 8, 8, 8 ).
#' @param mode 'classification' or 'regression'. 
#'
#' @return a Wide ResNet keras model
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
#' X_trainSmall <- mnistData$train$x[1:10,,]
#' X_trainSmall <- array( data = X_trainSmall, dim = c( dim( X_trainSmall ), 1 ) )
#' Y_trainSmall <- to_categorical( mnistData$train$y[1:10], numberOfLabels )
#'
#' X_testSmall <- mnistData$test$x[1:10,,]
#' X_testSmall <- array( data = X_testSmall, dim = c( dim( X_testSmall ), 1 ) )
#' Y_testSmall <- to_categorical( mnistData$test$y[1:10], numberOfLabels )
#'
#' # We add a dimension of 1 to specify the channel size
#'
#' inputImageSize <- c( dim( X_trainSmall )[2:3], 1 )
#'
#' model <- createWideResNetModel2D( inputImageSize = inputImageSize,
#'   numberOfOutputs = numberOfLabels )
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
#' }
#' @import keras
#' @export
createWideResNetModel3D <- function( inputImageSize,
                                     numberOfOutputs = 1000,
                                     depth = 2,
                                     width = 1,
                                     residualBlockSchedule = c( 16, 32, 64 ),
                                     poolSize = c( 8, 8, 8 ),
                                     dropoutRate = 0.0,
                                     weightDecay = 0.0005,
                                     mode = c( 'classification', 'regression' )
                                   )
{

  mode <- match.arg( mode )

  channelAxis <- 1
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1
    }

  initialConvolutionLayer <- function( model, numberOfFilters )
    {
    model <- model %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = c( 3, 3, 3 ), padding = 'same',
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    return( model )
    }

  customConvolutionLayer <- function( initialModel, base, width, strides = c( 1, 1, 1 ),
    dropoutRate = 0.0, expand = TRUE )
    {
    numberOfFilters <- as.integer( base * width )

    if( expand == TRUE )
      {
      model <- initialModel %>% layer_conv_3d( filters = numberOfFilters, kernel_size = c( 3, 3, 3 ),
        padding = 'same', strides = strides, kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )
      } else {
      model <- initialModel
      }

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    model <- model %>% layer_conv_3d( filters = numberOfFilters, kernel_size = c( 3, 3, 3 ),
      padding = 'same', kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

    if( expand == TRUE )
      {
      skipLayer <- initialModel %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = c( 1, 1, 1 ), padding = 'same', strides = strides,
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( model, skipLayer ) )
      } else {
      if( dropoutRate > 0.0 )
        {
        model <- model %>% layer_dropout( rate = dropoutRate )
        }

      model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
        epsilon = 1.0e-5, gamma_initializer = "uniform" )
      model <- model %>% layer_activation( activation = "relu" )

      model <- model %>% layer_conv_3d( filters = numberOfFilters, kernel_size = c( 3, 3, 3 ),
        padding = 'same', kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( initialModel, model ) )
      }

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- initialConvolutionLayer( inputs, residualBlockSchedule[1] )
  numberOfConvolutions <- 4

  for( i in seq_len( length( residualBlockSchedule ) ) )
    {
    baseNumberOfFilters <- residualBlockSchedule[i]

    outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters, width = width,
      strides = c( 1, 1, 1 ), dropoutRate = 0.0, expand = TRUE )
    numberOfConvolutions <- numberOfConvolutions + 2

    for( j in seq_len( depth ) )
      {
      outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters,
        width = width, dropoutRate = dropoutRate, expand = FALSE )
      numberOfConvolutions <- numberOfConvolutions + 2
      }

    outputs <- outputs %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    outputs <- outputs %>% layer_activation( activation = "relu" )
    }

  outputs <- outputs %>% layer_average_pooling_3d( pool_size = poolSize )
  outputs <- outputs %>% layer_flatten()

  layerActivation <- ''
  if( mode == 'classification' ) {
    layerActivation <- 'softmax'  
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'  
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>% layer_dense( units = numberOfOutputs,
    kernel_regularizer = regularizer_l2( weightDecay ), activation = layerActivation )

  wideResNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( wideResNetModel )
}

