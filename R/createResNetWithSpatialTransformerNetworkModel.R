#' 2-D implementation of the ResNet deep learning architecture with a
#' preceding spatial transformer network layer.
#'
#' Creates a keras model of the ResNet deep learning architecture for image
#' classification with a spatial transformer network (STN) layer.  The paper
#' is available here:
#'
#'         https://arxiv.org/abs/1512.03385
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
#' @param lowestResolution number of filters at the initial layer.
#' @param cardinality perform  ResNet (cardinality = 1) or ResNeXt
#' (cardinality != 1 but powers of 2---try '32' )
#' @param resampledSize output image size of the spatial transformer network.
#' @param numberOfSpatialTransformerUnits number of units in the dense layer.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return an STN + ResNet keras model
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
#' model <- createResNetWithSpatialTransformerNetworkModel2D(
#'   inputImageSize = inputImageSize,
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
#' }
#' @import keras
#' @export
createResNetWithSpatialTransformerNetworkModel2D <- function( inputImageSize,
                                                              numberOfClassificationLabels = 1000,
                                                              layers = 1:4,
                                                              residualBlockSchedule = c( 3, 4, 6, 3 ),
                                                              lowestResolution = 64,
                                                              cardinality = 1,
                                                              numberOfSpatialTransformerUnits = 50,
                                                              resampledSize = c( 64, 64 ),
                                                              mode = 'classification'
                                                             )
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


  # The spatial transformer network part

  localizationModel <- createResNetModel2D( inputImageSize = inputImageSize,
     numberOfClassificationLabels = numberOfSpatialTransformerUnits,
     mode = "regression" )
  localization <- localizationModel$output %>% layer_activation( 'relu' )

  # inputs <- layer_input( shape = inputImageSize )

  # localization <- inputs
  # localization <- localization %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  # localization <- localization %>% layer_conv_2d( filters = 20, kernel_size = c( 5, 5 ) )
  # localization <- localization %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
  # localization <- localization %>% layer_conv_2d( filters = 20, kernel_size = c( 5, 5 ) )

  # localization <- localization %>% layer_flatten()
  # localization <- localization %>% layer_dense( units = 50L )
  # localization <- localization %>% layer_activation( 'relu' )

  weights <- getInitialWeights2D( outputSize = numberOfSpatialTransformerUnits )
  localization <- localization %>% layer_dense( units = 6L, weights = weights )

  outputs <- layer_spatial_transformer_2d(
    list( localizationModel$input, localization ),
    resampledSize, transformType = 'affine', interpolatorType = 'linear',
    name = "layer_spatial_transformer" )

  # outputs <- outputs %>%
  #   layer_conv_2d( filters = 32L, kernel_size = c( 3, 3 ), padding = 'same' )

  # The ResNet part

  nFilters <- lowestResolution

  outputs <- outputs %>% layer_conv_2d( filters = nFilters,
    kernel_size = c( 7, 7 ), strides = c( 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ), padding = 'same' )

  for( i in seq_len( length( layers ) ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in seq_len( residualBlockSchedule[i] ) )
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

  resNetModel <- keras_model( inputs = localizationModel$inputs, outputs = outputs )

  return( resNetModel )
}

#' 3-D implementation of the ResNet deep learning architecture with a
#' preceding spatial transformer network layer.
#'
#' Creates a keras model of the ResNet deep learning architecture for image
#' classification with a spatial transformer network (STN) layer.  The paper
#' is available here:
#'
#'         https://arxiv.org/abs/1512.03385
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
#' @param lowestResolution number of filters at the initial layer.
#' @param cardinality perform  ResNet (cardinality = 1) or ResNeXt
#' (cardinality != 1 but powers of 2---try '32' )
#' @param resampledSize output image size of the spatial transformer network.
#' @param numberOfSpatialTransformerUnits number of units in the dense layer.
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
#' model <- createResNetWithSpatialTransformerNetworkModel2D(
#'   inputImageSize = inputImageSize,
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
createResNetWithSpatialTransformerNetworkModel3D <- function( inputImageSize,
                                                              numberOfClassificationLabels = 1000,
                                                              layers = 1:4,
                                                              residualBlockSchedule = c( 3, 4, 6, 3 ),
                                                              lowestResolution = 64,
                                                              cardinality = 1,
                                                              resampledSize = c( 64, 64, 64 ),
                                                              numberOfSpatialTransformerUnits = 50,
                                                              mode = 'classification'
                                                            )
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

  # The spatial transformer network part

  localizationModel <- createResNetModel3D( inputImageSize = inputImageSize,
     numberOfClassificationLabels = numberOfSpatialTransformerUnits,
     mode = "regression" )
  localization <- localizationModel$output %>% layer_activation( 'relu' )

  # inputs <- layer_input( shape = inputImageSize )

  # localization <- inputs
  # localization <- localization %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
  # localization <- localization %>% layer_conv_3d( filters = 20, kernel_size = c( 5, 5, 5 ) )
  # localization <- localization %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
  # localization <- localization %>% layer_conv_3d( filters = 20, kernel_size = c( 5, 5, 5 ) )

  # localization <- localization %>% layer_flatten()
  # localization <- localization %>% layer_dense( units = numberOfSpatialTransformerUnits )
  # localization <- localization %>% layer_activation( 'relu' )

  weights <- getInitialWeights3D( outputSize = numberOfSpatialTransformerUnits )
  localization <- localization %>% layer_dense( units = 12L, weights = weights )

  outputs <- layer_spatial_transformer_3d(
    list( localizationModel$input, localization ),
    resampledSize, transformType = 'affine', interpolatorType = 'linear',
    name = "layer_spatial_transformer" )

  # outputs <- outputs %>%
  #   layer_conv_3d( filters = 32L, kernel_size = c( 3, 3 ), padding = 'same' )

  # The ResNet part

  nFilters <- lowestResolution

  outputs <- inputs %>% layer_conv_3d( filters = nFilters,
    kernel_size = c( 7, 7, 7 ), strides = c( 2, 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ), padding = 'same' )

  for( i in seq_len( length( layers ) ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in seq_len( residualBlockSchedule[i] ) )
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

  resNetModel <- keras_model( inputs = inputs = localizationModel$inputs, outputs = outputs )

  return( resNetModel )
}
