#' 2-D implementation of the dense U-net deep learning architecture.
#'
#' Creates a keras model of the dense U-net deep learning architecture for
#' image segmentation
#'
#' X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, P.-A. Heng. H-DenseUNet: Hybrid
#' Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes
#'
#' available here:
#'
#'         https://arxiv.org/pdf/1709.07330.pdf
#'
#' with the author's implementation available at:
#'
#'         https://github.com/xmengli999/H-DenseUNet
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param numberOfOutputs Meaning depends on the \code{mode}.  For
#' 'classification' this is the number of segmentation labels.  For 'regression'
#' this is the number of outputs.
#' @param numberOfLayersPerDenseBlocks number of dense blocks per layer.
#' @param growthRate number of filters to add for each dense block layer
#' (default = 48).
#' @param initialNumberOfFilters number of filters at the beginning
#'  (default = 96).
#' @param reductionRate reduction factor of transition blocks
#' @param depth number of layers---must be equal to 3 * N + 4 where
#' N is an integer (default = 7).
#' @param dropoutRate drop out layer rate (default = 0.2).
#' @param weightDecay weight decay (default = 1e-4).
#' @return an DenseUnet keras model
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsR )
#' library( ANTsRNet )
#' library( keras )
#'
#' imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
#' trainingBatchSize <- length( imageIDs )
#'
#' # Perform simple 3-tissue segmentation.
#'
#' segmentationLabels <- c( 1, 2, 3 )
#' numberOfLabels <- length( segmentationLabels )
#' initialization <- paste0( 'KMeans[', numberOfLabels, ']' )
#'
#' domainImage <- antsImageRead( getANTsRData( imageIDs[1] ) )
#'
#' X_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ), 1 ) )
#' Y_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ) ) )
#'
#' images <- list()
#' segmentations <- list()
#'
#' for( i in seq_len( trainingBatchSize ) )
#'   {
#'   cat( "Processing image", imageIDs[i], "\n" )
#'   image <- antsImageRead( getANTsRData( imageIDs[i] ) )
#'   mask <- getMask( image )
#'   segmentation <- atropos( image, mask, initialization )$segmentation
#'
#'   X_train[i,,, 1] <- as.array( image )
#'   Y_train[i,,] <- as.array( segmentation )
#'   }
#' Y_train <- encodeUnet( Y_train, segmentationLabels )
#'
#' # Perform a simple normalization
#'
#' X_train <- ( X_train - mean( X_train ) ) / sd( X_train )
#'
#' # Create the model
#'
#' # Create the model
#'
#' model <- createDenseUnetModel2D( c( dim( domainImage ), 1 ),
#'   numberOfOutputs = numberOfLabels )
#'
#' metric_multilabel_dice_coefficient <-
#'   custom_metric( "multilabel_dice_coefficient",
#'     multilabel_dice_coefficient )
#'
#' loss_dice <- function( y_true, y_pred ) {
#'   -multilabel_dice_coefficient(y_true, y_pred)
#' }
#' attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"
#'
#' model %>% compile( loss = loss_dice,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( metric_multilabel_dice_coefficient,
#'     metric_categorical_crossentropy ) )
#'
#' # Comment out the rest due to travis build constraints
#'
#' # Fit the model
#'
#' # track <- model %>% fit( X_train, Y_train,
#' #              epochs = 100, batch_size = 4, verbose = 1, shuffle = TRUE,
#' #              callbacks = list(
#' #                callback_model_checkpoint( "unetModelInterimWeights.h5",
#' #                    monitor = 'val_loss', save_best_only = TRUE ),
#' #                callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#' #              ),
#' #              validation_split = 0.2 )
#'
#' # Save the model and/or save the model weights
#'
#' # save_model_hdf5( model, filepath = 'unetModel.h5' )
#' # save_model_weights_hdf5( unetModel, filepath = 'unetModelWeights.h5' ) )
#'
#' @import keras
#' @export
createDenseUnetModel2D <- function( inputImageSize,
                                    numberOfOutputs = 1,
                                    numberOfLayersPerDenseBlock = c( 6, 12, 36, 24 ),
                                    growthRate = 48,
                                    initialNumberOfFilters = 96,
                                    reductionRate = 0.0,
                                    depth = 7,
                                    dropoutRate = 0.0,
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
    # Bottleneck layer

    model <- model %>% layer_batch_normalization( axis = concatenationAxis )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_2d( filters = numberOfFilters * 4,
      kernel_size = c( 1, 1 ), use_bias = FALSE )

    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }

    # Convolution layer

    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
      epsilon = 1.1e-5 )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_zero_padding_2d( padding = c( 1, 1 ) )
    model <- model %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = kernelSize, use_bias = FALSE )

    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }

    return( model )
    }

  transition2D <- function( model, numberOfFilters, compressionRate = 1.0,
                            dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
      gamma_regularizer = regularizer_l2( weightDecay ),
      beta_regularizer = regularizer_l2( weightDecay ) )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_2d( filters =
      as.integer( numberOfFilters * compressionRate ),
      kernel_size = c( 1, 1 ), use_bias = FALSE )

    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }

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
        kernelSize = c( 3, 3 ), dropoutRate = dropoutRate,
        weightDecay = weightDecay )
      denseBlockLayers[[i+1]] <- model
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

  numberOfDenseBlocks <- length( numberOfLayersPerDenseBlock )

  inputs <- layer_input( shape = inputImageSize )

  boxLayers <- list()
  boxCount <- 1

  # Initial convolution

  outputs <- inputs %>% layer_zero_padding_2d( padding = c( 3, 3 ) )
  outputs <- outputs %>% layer_conv_2d( filters = initialNumberOfFilters,
    kernel_size = c( 7, 7 ), strides = c( 2, 2 ), use_bias = FALSE )
  outputs <- outputs %>% layer_batch_normalization( epsilon = 1.1e-5,
    axis = concatenationAxis )
  outputs <- outputs %>% layer_scale( axis = concatenationAxis )
  outputs <- outputs %>% layer_activation( activation = "relu" )

  boxLayers[[boxCount]] <- outputs
  boxCount <- boxCount + 1

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 1, 1 ) )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ) )

  # Add dense blocks

  nFilters <- initialNumberOfFilters

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks2D( outputs,
      numberOfFilters = nFilters, depth = numberOfLayersPerDenseBlock[i],
      growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model

    boxLayers[[boxCount]] <- outputs
    boxCount <- boxCount + 1

    outputs <- transition2D( outputs,
      numberOfFilters = denseBlockLayer$numberOfFilters,
      compressionRate = 1.0 - reductionRate,
      dropoutRate = dropoutRate,
      weightDecay = weightDecay )

    nFilters <- as.integer( denseBlockLayer$numberOfFilters * ( 1 - reductionRate ) )
    }

  denseBlockLayer <- createDenseBlocks2D( outputs, numberOfFilters = nFilters,
    depth = numberOfLayersPerDenseBlock[numberOfDenseBlocks],
    growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters

  outputs <- outputs %>% layer_batch_normalization( epsilon = 1.1e-5,
    axis = concatenationAxis )
  outputs <- outputs %>% layer_scale( axis = concatenationAxis )
  outputs <- outputs %>% layer_activation( activation = "relu" )

  boxLayers[[boxCount]] <- outputs
  boxCount <- boxCount - 1

  localNumberOfFilters <- tail( unlist( K$int_shape( boxLayers[[boxCount+1]] ) ), 1 )
  localLayer <- boxLayers[[boxCount]] %>% layer_conv_2d( filters = localNumberOfFilters,
    kernel_size = c( 1, 1 ), padding = 'same', kernel_initializer = 'normal' )
  boxCount <- boxCount - 1

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
    {
    upsamplingLayer <- outputs %>% layer_upsampling_2d( size = c( 2, 2 ) )
    outputs <- layer_add( list( localLayer, upsamplingLayer ) )

    localLayer <- boxLayers[[boxCount]]
    boxCount <- boxCount - 1

    localNumberOfFilters <- tail( unlist( K$int_shape( boxLayers[[boxCount+1]] ) ), 1 )
    outputs <- outputs %>% layer_conv_2d( filters = localNumberOfFilters,
      kernel_size = c( 3, 3 ), padding = 'same', kernel_initializer = 'normal' )

    if( i == numberOfDenseBlocks - 1 )
      {
      outputs <- outputs %>% layer_dropout( rate = 0.3 )
      }

    outputs <- outputs %>% layer_batch_normalization()
    outputs <- outputs %>% layer_activation( activation = "relu" )
    }

  convActivation <- ''
  if( mode == 'classification' )
    {
    if( numberOfOutputs == 2 )
      {
      convActivation <- 'sigmoid'
      } else {
      convActivation <- 'softmax'
      }
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }
  outputs <- outputs %>%
    layer_conv_2d( filters = numberOfOutputs,
      kernel_size = c( 1, 1 ), activation = convActivation,
      kernel_initializer = 'normal' )

  denseUnetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseUnetModel )
}

#' 3-D implementation of the dense U-net deep learning architecture.
#'
#' Creates a keras model of the dense U-net deep learning architecture for
#' image segmentation
#'
#' X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, P.-A. Heng. H-DenseUNet: Hybrid
#' Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes
#'
#' available here:
#'
#'         https://arxiv.org/pdf/1709.07330.pdf
#'
#' with the author's implementation available at:
#'
#'         https://github.com/xmengli999/H-DenseUNet
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param numberOfOutputs Meaning depends on the \code{mode}.  For
#' 'classification' this is the number of segmentation labels.  For 'regression'
#' this is the number of outputs.
#' @param numberOfLayersPerDenseBlocks number of dense blocks per layer.
#' @param growthRate number of filters to add for each dense block layer
#' (default = 48).
#' @param initialNumberOfFilters number of filters at the beginning
#' (default = 96).
#' @param reductionRate reduction factor of transition blocks
#' @param depth number of layers---must be equal to 3 * N + 4 where
#' N is an integer (default = 7).
#' @param dropoutRate drop out layer rate (default = 0.2).
#' @param weightDecay weight decay (default = 1e-4).
#' @return an DenseUnet keras model
#' @author Tustison NJ
#' @examples
#'
#' @examples
#'
#' library( ANTsR )
#' library( ANTsRNet )
#' library( keras )
#'
#' imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
#' trainingBatchSize <- length( imageIDs )
#'
#' # Perform simple 3-tissue segmentation.
#'
#' segmentationLabels <- c( 1, 2, 3 )
#' numberOfLabels <- length( segmentationLabels )
#' initialization <- paste0( 'KMeans[', numberOfLabels, ']' )
#'
#' domainImage <- antsImageRead( getANTsRData( imageIDs[1] ) )
#'
#' X_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ), 1 ) )
#' Y_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ) ) )
#'
#' images <- list()
#' segmentations <- list()
#'
#' for( i in seq_len( trainingBatchSize ) )
#'   {
#'   cat( "Processing image", imageIDs[i], "\n" )
#'   image <- antsImageRead( getANTsRData( imageIDs[i] ) )
#'   mask <- getMask( image )
#'   segmentation <- atropos( image, mask, initialization )$segmentation
#'
#'   X_train[i,,, 1] <- as.array( image )
#'   Y_train[i,,] <- as.array( segmentation )
#'   }
#' Y_train <- encodeUnet( Y_train, segmentationLabels )
#'
#' # Perform a simple normalization
#'
#' X_train <- ( X_train - mean( X_train ) ) / sd( X_train )
#'
#' # Create the model
#'
#' model <- createDenseUnetModel2D( c( dim( domainImage ), 1 ),
#'   numberOfOutputs = numberOfLabels )
#'
#' metric_multilabel_dice_coefficient <-
#'   custom_metric( "multilabel_dice_coefficient",
#'     multilabel_dice_coefficient )
#'
#' loss_dice <- function( y_true, y_pred ) {
#'   -multilabel_dice_coefficient(y_true, y_pred)
#' }
#' attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"
#'
#' model %>% compile( loss = loss_dice,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( metric_multilabel_dice_coefficient,
#'     metric_categorical_crossentropy ) )
#'
#' # Comment out the rest due to travis build constraints
#'
#' # Fit the model
#'
#' # track <- model %>% fit( X_train, Y_train,
#' #              epochs = 100, batch_size = 4, verbose = 1, shuffle = TRUE,
#' #              callbacks = list(
#' #                callback_model_checkpoint( "unetModelInterimWeights.h5",
#' #                    monitor = 'val_loss', save_best_only = TRUE ),
#' #                callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#' #              ),
#' #              validation_split = 0.2 )
#'
#' # Save the model and/or save the model weights
#'
#' # save_model_hdf5( model, filepath = 'unetModel.h5' )
#' # save_model_weights_hdf5( unetModel, filepath = 'unetModelWeights.h5' ) )
#'
#' @import keras
#' @export
createDenseUnetModel3D <- function( inputImageSize,
                                    numberOfOutputs = 1,
                                    numberOfLayersPerDenseBlock = c( 3, 4, 12, 8 ),
                                    growthRate = 48,
                                    initialNumberOfFilters = 96,
                                    reductionRate = 0.0,
                                    depth = 7,
                                    dropoutRate = 0.0,
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
    # Bottleneck layer

    model <- model %>% layer_batch_normalization( axis = concatenationAxis )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_3d( filters = numberOfFilters * 4,
      kernel_size = c( 1, 1, 1 ), use_bias = FALSE )

    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }

    # Convolution layer

    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
      epsilon = 1.1e-5 )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_zero_padding_3d( padding = c( 1, 1, 1 ) )
    model <- model %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, use_bias = FALSE )

    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }

    return( model )
    }

  transition3D <- function( model, numberOfFilters, compressionRate = 1.0,
                            dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
      gamma_regularizer = regularizer_l2( weightDecay ),
      beta_regularizer = regularizer_l2( weightDecay ) )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_3d( filters =
      as.integer( numberOfFilters * compressionRate ),
      kernel_size = c( 1, 1, 1 ), use_bias = FALSE )

    if( dropoutRate > 0.0 )
      {
      model <- model %>% layer_dropout( rate = dropoutRate )
      }

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
        kernelSize = c( 3, 3, 3 ), dropoutRate = dropoutRate,
        weightDecay = weightDecay )
      denseBlockLayers[[i+1]] <- model
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

  numberOfDenseBlocks <- length( numberOfLayersPerDenseBlock )

  inputs <- layer_input( shape = inputImageSize )

  boxLayers <- list()
  boxCount <- 1

  # Initial convolution

  outputs <- inputs %>% layer_zero_padding_3d( padding = c( 3, 3, 3 ) )
  outputs <- outputs %>% layer_conv_3d( filters = initialNumberOfFilters,
    kernel_size = c( 7, 7, 7 ), strides = c( 2, 2, 2 ), use_bias = FALSE )
  outputs <- outputs %>% layer_batch_normalization( epsilon = 1.1e-5,
    axis = concatenationAxis )
  outputs <- outputs %>% layer_scale( axis = concatenationAxis )
  outputs <- outputs %>% layer_activation( activation = "relu" )

  boxLayers[[boxCount]] <- outputs
  boxCount <- boxCount + 1

  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 1, 1 ) )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ) )

  # Add dense blocks

  nFilters <- initialNumberOfFilters

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks3D( outputs,
      numberOfFilters = nFilters, depth = numberOfLayersPerDenseBlock[i],
      growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model

    boxLayers[[boxCount]] <- outputs
    boxCount <- boxCount + 1

    outputs <- transition3D( outputs,
      numberOfFilters = denseBlockLayer$numberOfFilters,
      compressionRate = 1.0 - reductionRate,
      dropoutRate = dropoutRate,
      weightDecay = weightDecay )

    nFilters <- as.integer( denseBlockLayer$numberOfFilters * ( 1 - reductionRate ) )
    }

  denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters,
    depth = numberOfLayersPerDenseBlock[numberOfDenseBlocks],
    growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters

  outputs <- outputs %>% layer_batch_normalization( epsilon = 1.1e-5,
    axis = concatenationAxis )
  outputs <- outputs %>% layer_scale( axis = concatenationAxis )
  outputs <- outputs %>% layer_activation( activation = "relu" )

  boxLayers[[boxCount]] <- outputs
  boxCount <- boxCount - 1

  localNumberOfFilters <- tail( unlist( K$int_shape( boxLayers[[boxCount+1]] ) ), 1 )
  localLayer <- boxLayers[[boxCount]] %>% layer_conv_3d( filters = localNumberOfFilters,
    kernel_size = c( 1, 1, 1 ), padding = 'same', kernel_initializer = 'normal' )
  boxCount <- boxCount - 1

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
    {
    upsamplingLayer <- outputs %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
    outputs <- layer_add( list( localLayer, upsamplingLayer ) )

    localLayer <- boxLayers[[boxCount]]
    boxCount <- boxCount - 1

    localNumberOfFilters <- tail( unlist( K$int_shape( boxLayers[[boxCount+1]] ) ), 1 )
    outputs <- outputs %>% layer_conv_3d( filters = localNumberOfFilters,
      kernel_size = c( 3, 3, 3 ), padding = 'same', kernel_initializer = 'normal' )

    if( i == numberOfDenseBlocks )
      {
      outputs <- outputs %>% layer_dropout( rate = 0.3 )
      }

    outputs <- outputs %>% layer_batch_normalization()
    outputs <- outputs %>% layer_activation( activation = "relu" )
    }

  convActivation <- ''
  if( mode == 'classification' )
    {
    if( numberOfOutputs == 2 )
      {
      convActivation <- 'sigmoid'
      } else {
      convActivation <- 'softmax'
      }
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }
  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = c( 1, 1, 1 ), activation = convActivation,
      kernel_initializer = 'normal' )

  denseUnetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseUnetModel )
}


