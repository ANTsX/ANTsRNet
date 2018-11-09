#' 2-D implementation of the ResNet deep learning architecture.
#'
#' Creates a keras model of the ResNet deep learning architecture for image
#' classification.  The paper is available here:
#'
#'         https://arxiv.org/abs/1512.03385
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param residualBlockSchedule vector defining the how many residual blocks
#' repeats.
#' @param lowestResolution number of filters at the beginning and end of
#' the 'U'.
#' @param cardinality perform  ResNet (cardinality = 1) or ResNeXt
#' (cardinality != 1 but powers of 2---try '32' )
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return an ResNet keras model
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
#' model <- createResNetModel2D( inputImageSize = inputImageSize,
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
createResNetModel2D <- function( inputImageSize,
                                 numberOfClassificationLabels = 1000,
                                 layers = 1:4,
                                 residualBlockSchedule = c( 3, 4, 6, 3 ),
                                 lowestResolution = 64,
                                 cardinality = 1,
                                 mode = 'classification'
                               )
{

  addCommonLayers <- function( model )
    {
    model <- model %>% layer_batch_normalization()
    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  groupedConvolutionLayer2D <- function( model, numberOfFilters, strides )
    {
    K <- keras::backend()

    # Per standard ResNet, this is just a 2-D convolution
    if( cardinality == 1 )
      {
      groupedModel <- model %>% layer_conv_2d( filters = numberOfFilters,
        kernel_size = c( 3, 3 ), strides = strides, padding = 'same' )
      return( groupedModel )
      }

    if( numberOfFilters %% cardinality != 0 )
      {
      stop( "numberOfFilters %% cardinality != 0" )
      }

    numberOfGroupFilters <- as.integer( numberOfFilters / cardinality )

    convolutionLayers <- list()
    for( j in 1:cardinality )
      {
      convolutionLayers[[j]] <- model %>% layer_lambda( function( z )
        {
        K$set_image_data_format( 'channels_last' )
        z[,,, ( ( j - 1 ) * numberOfGroupFilters + 1 ):( j * numberOfGroupFilters )]
        } )
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>%
        layer_conv_2d( filters = numberOfGroupFilters,
          kernel_size = c( 3, 3 ), strides = strides, padding = 'same' )
      }

    groupedModel <- layer_concatenate( convolutionLayers )
    return( groupedModel )
    }

  residualBlock2D <- function( model, numberOfFiltersIn, numberOfFiltersOut,
    strides = c( 1, 1 ), projectShortcut = FALSE )
    {
    shortcut <- model

    model <- model %>% layer_conv_2d( filters = numberOfFiltersIn,
      kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same' )
    model <- addCommonLayers( model )

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer2D( model, numberOfFilters = numberOfFiltersIn,
      strides = strides )
    model <- addCommonLayers( model )

    model <- model %>% layer_conv_2d( filters = numberOfFiltersOut,
      kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same' )
    model <- model %>% layer_batch_normalization()

    if( projectShortcut == TRUE || prod( strides == c( 1, 1 ) ) == 0 )
      {
      shortcut <- shortcut %>% layer_conv_2d( filters = numberOfFiltersOut,
        kernel_size = c( 1, 1 ), strides = strides, padding = 'same' )
      shortcut <- shortcut %>% layer_batch_normalization()
      }

    model <- layer_add( list( shortcut, model ) )

    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  nFilters <- lowestResolution

  outputs <- inputs %>% layer_conv_2d( filters = nFilters,
    kernel_size = c( 7, 7 ), strides = c( 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ), padding = 'same' )

  for( i in 1:length( layers ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in 1:residualBlockSchedule[i] )
      {
      projectShortcut <- FALSE
      if( i == 1 && j == 1 )
        {
        projectShortcut <- TRUE
        }
      if( i > 1 && j == 1 )
        {
        strides <- c( 2, 2 )
        } else {
        strides <- c( 1, 1 )
        }
      outputs <- residualBlock2D( outputs, numberOfFiltersIn = nFiltersIn,
        numberOfFiltersOut = nFiltersOut, strides = strides,
        projectShortcut = projectShortcut )
      }
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

  resNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( resNetModel )
}

#' 3-D implementation of the ResNet deep learning architecture.
#'
#' Creates a keras model of the ResNet deep learning architecture for image
#' classification.  The paper is available here:
#'
#'         https://arxiv.org/abs/1512.03385
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param residualBlockSchedule vector defining the how many residual blocks
#' repeats.
#' @param lowestResolution number of filters at the beginning and end of
#' the 'U'.
#' @param cardinality perform  ResNet (cardinality = 1) or ResNeXt
#' (cardinality != 1 but powers of 2---try '32' )
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return an ResNet keras model
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
#' model <- createResNetModel2D( inputImageSize = inputImageSize,
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
createResNetModel3D <- function( inputImageSize,
                                 numberOfClassificationLabels = 1000,
                                 layers = 1:4,
                                 residualBlockSchedule = c( 3, 4, 6, 3 ),
                                 lowestResolution = 64,
                                 cardinality = 1,
                                 mode = 'classification'
                               )
{

  addCommonLayers <- function( model )
    {
    model <- model %>% layer_batch_normalization()
    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  groupedConvolutionLayer3D <- function( model, numberOfFilters, strides )
    {
    K <- keras::backend()

    # Per standard ResNet, this is just a 2-D convolution
    if( cardinality == 1 )
      {
      groupedModel <- model %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = c( 3, 3, 3 ), strides = strides, padding = 'same' )
      return( groupedModel )
      }

    if( numberOfFilters %% cardinality != 0 )
      {
      stop( "numberOfFilters %% cardinality != 0" )
      }

    numberOfGroupFilters <- as.integer( numberOfFilters / cardinality )

    convolutionLayers <- list()
    for( j in 1:cardinality )
      {
      convolutionLayers[[j]] <- model %>% layer_lambda( function( z )
        {
        K$set_image_data_format( 'channels_last' )
        z[,,,, ( ( j - 1 ) * numberOfGroupFilters + 1 ):( j * numberOfGroupFilters )]
        } )
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>%
        layer_conv_3d( filters = numberOfGroupFilters,
          kernel_size = c( 3, 3, 3 ), strides = strides, padding = 'same' )
      }

    groupedModel <- layer_concatenate( convolutionLayers )
    return( groupedModel )
    }

  residualBlock3D <- function( model, numberOfFiltersIn, numberOfFiltersOut,
    strides = c( 1, 1, 1 ), projectShortcut = FALSE )
    {
    shortcut <- model

    model <- model %>% layer_conv_3d( filters = numberOfFiltersIn,
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
    model <- addCommonLayers( model )

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer3D( model, numberOfFilters = numberOfFiltersIn,
      strides = strides )
    model <- addCommonLayers( model )

    model <- model %>% layer_conv_3d( filters = numberOfFiltersOut,
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
    model <- model %>% layer_batch_normalization()

    if( projectShortcut == TRUE || prod( strides == c( 1, 1, 1 ) ) == 0 )
      {
      shortcut <- shortcut %>% layer_conv_3d( filters = numberOfFiltersOut,
        kernel_size = c( 1, 1, 1 ), strides = strides, padding = 'same' )
      shortcut <- shortcut %>% layer_batch_normalization()
      }

    model <- layer_add( list( shortcut, model ) )

    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  nFilters <- lowestResolution

  outputs <- inputs %>% layer_conv_3d( filters = nFilters,
    kernel_size = c( 7, 7, 7 ), strides = c( 2, 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ), padding = 'same' )

  for( i in 1:length( layers ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in 1:residualBlockSchedule[i] )
      {
      projectShortcut <- FALSE
      if( i == 1 && j == 1 )
        {
        projectShortcut <- TRUE
        }
      if( i > 1 && j == 1 )
        {
        strides <- c( 2, 2, 2 )
        } else {
        strides <- c( 1, 1, 1 )
        }
      outputs <- residualBlock3D( outputs, numberOfFiltersIn = nFiltersIn,
        numberOfFiltersOut = nFiltersOut, strides = strides,
        projectShortcut = projectShortcut )
      }
    }
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

  outputs <- outputs %>%
    layer_dense( units = numberOfClassificationLabels, activation = layerActivation )

  resNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( resNetModel )
}