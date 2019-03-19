#' 2-D implementation of the DenseNet deep learning architecture.
#'
#' Creates a keras model of the DenseNet deep learning architecture for image
#' recognition based on the paper
#'
#' G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected
#'   Convolutional Networks Networks
#'
#' available here:
#'
#'         https://arxiv.org/abs/1608.06993
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param numberOfFilters number of filters
#' @param depth number of layers---must be equal to 3 * N + 4 where
#' N is an integer (default = 7).
#' @param numberOfDenseBlocks number of dense blocks to add to the end
#' (default = 1).
#' @param growthRate number of filters to add for each dense block layer
#' (default = 12).
#' @param dropoutRate = per drop out layer rate (default = 0.2)
#' @param weightDecay = weight decay (default = 1e-4)
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#' @return an DenseNet keras model
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
#' model <- createDenseNetModel2D( inputImageSize = inputImageSize,
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
createDenseNetModel2D <- function( inputImageSize,
                                   numberOfClassificationLabels = 1000,
                                   numberOfFilters = 16,
                                   depth = 7,
                                   numberOfDenseBlocks = 1,
                                   growthRate = 12,
                                   dropoutRate = 0.2,
                                   weightDecay = 1e-4,
                                   mode = 'classification'
                                 )
{
  K <- keras::backend()

  concatenationAxis <- 1
  if( K$image_data_format() == 'channels_last' )
    {
    concatenationAxis <- -1
    }

  convolutionFactory2D <- function( model, numberOfFilters, kernelSize = c( 3, 3 ),
                                    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
      gamma_regularizer = regularizer_l2( weightDecay ),
      beta_regularizer = regularizer_l2( weightDecay ) )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = kernelSize, kernel_initializer = 'he_uniform', padding = 'same',
      use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }
    return( model )
    }

  transition2D <- function( model, numberOfFilters, dropoutRate = 0.0,
                            weightDecay = 1e-4 )
    {
    model <- convolutionFactory2D( model, numberOfFilters, kernelSize = c( 1, 1 ),
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    model <- model %>% layer_average_pooling_2d( pool_size = c( 2, 2 ),
      strides = c( 2, 2 ) )
    return( model )
    }

  createDenseBlocks2D <- function( model, numberOfFilters, depth, growthRate,
    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    denseBlockLayers <- list( model )
    for( i in seq_len( depth ) )
      {
      model <- convolutionFactory2D( model, numberOfFilters = growthRate,
        kernelSize = c( 3, 3 ), dropoutRate = dropoutRate, weightDecay = weightDecay )
      denseBlockLayers[[i + 1]] <- model
      model <- layer_concatenate( denseBlockLayers, axis = concatenationAxis )
      numberOfFilters <- numberOfFilters + growthRate
      }

    return( list( model = model, numberOfFilters = numberOfFilters ) )
    }

  if( ( depth - 4 ) %% 3 != 0 )
    {
    stop( "Depth must be equal to 3*N+4 where N is an integer." )
    }
  numberOfLayers = as.integer( ( depth - 4 ) / 3 )

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs %>% layer_conv_2d( filters = numberOfFilters,
    kernel_size = c( 3, 3 ), kernel_initializer = 'he_uniform', padding = 'same',
    use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )

  # Add dense blocks

  nFilters <- numberOfFilters

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks2D( outputs, numberOfFilters = nFilters,
      depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model
    nFilters <- denseBlockLayer$numberOfFilters

    outputs <- transition2D( outputs, numberOfFilters = nFilters,
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    }

  denseBlockLayer <- createDenseBlocks2D( outputs, numberOfFilters = nFilters,
    depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters

  outputs <- outputs %>% layer_batch_normalization( axis = concatenationAxis,
    gamma_regularizer = regularizer_l2( weightDecay ),
    beta_regularizer = regularizer_l2( weightDecay ) )

  outputs <- outputs %>% layer_activation( activation = 'relu' )
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

  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels,
    activation = layerActivation,
    kernel_regularizer = regularizer_l2( weightDecay ),
    bias_regularizer = regularizer_l2( weightDecay ) )

  denseNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseNetModel )
}

#' 3-D implementation of the DenseNet deep learning architecture.
#'
#' Creates a keras model of the DenseNet deep learning architecture for image
#' recognition based on the paper
#'
#' G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected
#'   Convolutional Networks Networks
#'
#' available here:
#'
#'         https://arxiv.org/abs/1608.06993
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param numberOfFilters number of filters
#' @param depth number of layers---must be equal to 3 * N + 4 where
#' N is an integer (default = 7).
#' @param numberOfDenseBlocks number of dense blocks to add to the end
#' (default = 1).
#' @param growthRate number of filters to add for each dense block layer
#' (default = 12).
#' @param dropoutRate = per drop out layer rate (default = 0.2)
#' @param weightDecay = weight decay (default = 1e-4)
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#' @return an DenseNet keras model
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
#' model <- createDenseNetModel2D( inputImageSize = inputImageSize,
#'   numberOfClassificationLabels = numberOfLabels )
#'
#' model %>% compile( loss = 'categorical_crossentropy',
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( 'categorical_crossentropy', 'accuracy' ) )
#'
#' track <- model %>% fit( X_trainSmall, Y_trainSmall, verbose = 1,
#'   epochs = 1, batch_size = 2, shuffle = TRUE, validation_split = 0.5 )
#'
#' # Now test the model
#'
#' testingMetrics <- model %>% evaluate( X_testSmall, Y_testSmall )
#' predictedData <- model %>% predict( X_testSmall, verbose = 1 )
#'
#' }
#' @import keras
#' @export
createDenseNetModel3D <- function( inputImageSize,
                                   numberOfClassificationLabels = 1000,
                                   numberOfFilters = 16,
                                   depth = 7,
                                   numberOfDenseBlocks = 1,
                                   growthRate = 12,
                                   dropoutRate = 0.2,
                                   weightDecay = 1e-4,
                                   mode = 'classification'
                                 )
{
  K <- keras::backend()

  concatenationAxis <- 1
  if( K$image_data_format() == 'channels_last' )
    {
    concatenationAxis <- -1
    }

  convolutionFactory3D <- function( model, numberOfFilters, kernelSize = c( 3, 3, 3 ),
                                    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
      gamma_regularizer = regularizer_l2( weightDecay ),
      beta_regularizer = regularizer_l2( weightDecay ) )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, kernel_initializer = 'he_uniform', padding = 'same',
      use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }
    return( model )
    }

  transition3D <- function( model, numberOfFilters, dropoutRate = 0.0,
                            weightDecay = 1e-4 )
    {
    model <- convolutionFactory3D( model, numberOfFilters, kernelSize = c( 1, 1, 1 ),
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    model <- model %>% layer_average_pooling_3d( pool_size = c( 2, 2, 2 ),
      strides = c( 2, 2, 2 ) )
    return( model )
    }

  createDenseBlocks3D <- function( model, numberOfFilters, depth, growthRate,
    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    denseBlockLayers <- list( model )
    for( i in seq_len( depth ) )
      {
      model <- convolutionFactory3D( model, numberOfFilters = growthRate,
        kernelSize = c( 3, 3, 3 ), dropoutRate = dropoutRate, weightDecay = weightDecay )
      denseBlockLayers[[i + 1]] <- model
      model <- layer_concatenate( denseBlockLayers, axis = concatenationAxis )
      numberOfFilters <- numberOfFilters + growthRate
      }

    return( list( model = model, numberOfFilters = numberOfFilters ) )
    }

  if( ( depth - 4 ) %% 3 != 0 )
    {
    stop( "Depth must be equal to 3*N+4 where N is an integer." )
    }
  numberOfLayers = as.integer( ( depth - 4 ) / 3 )

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs %>% layer_conv_3d( filters = numberOfFilters,
    kernel_size = c( 3, 3, 3 ), kernel_initializer = 'he_uniform',
    padding = 'same', use_bias = FALSE, kernel_regularizer =
    regularizer_l2( weightDecay ) )

  # Add dense blocks

  nFilters <- numberOfFilters

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters,
      depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model
    nFilters <- denseBlockLayer$numberOfFilters

    outputs <- transition3D( outputs, numberOfFilters = nFilters,
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    }

  denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters,
    depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters

  outputs <- outputs %>% layer_batch_normalization( axis = concatenationAxis,
    gamma_regularizer = regularizer_l2( weightDecay ),
    beta_regularizer = regularizer_l2( weightDecay ) )

  outputs <- outputs %>% layer_activation( activation = 'relu' )
  outputs <- outputs %>% layer_global_average_pooling_3d()

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

  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels,
    activation = layerActivation,
    kernel_regularizer = regularizer_l2( weightDecay ),
    bias_regularizer = regularizer_l2( weightDecay ) )

  denseNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseNetModel )
}
