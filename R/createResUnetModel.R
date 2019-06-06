#' 2-D implementation of the Resnet + U-net deep learning architecture.
#'
#' Creates a keras model of the U-net + ResNet deep learning architecture for
#' image segmentation and regression with the paper available here:
#'
#'         \url{https://arxiv.org/abs/1608.04117}
#'
#' This particular implementation was ported from the following python
#' implementation:
#'
#'         \url{https://github.com/veugene/fcn_maker/}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfOutputs Meaning depends on the \code{mode}.  For
#' 'classification' this is the number of segmentation labels.  For 'regression'
#' this is the number of outputs.
#' @param numberOfFiltersAtBaseLayer number of filters at the beginning and end
#' of the \verb{'U'}.  Doubles at each descending/ascending layer.
#' @param bottleNeckBlockDepthSchedule vector that provides the encoding layer
#' schedule for the number of bottleneck blocks per long skip connection.
#' @param convolutionKernelSize 2-d vector defining the kernel size
#' during the encoding path
#' @param deconvolutionKernelSize 2-d vector defining the kernel size
#' during the decoding
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param weightDecay weighting parameter for L2 regularization of the
#' kernel weights of the convolution layers.  Default = 0.0.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return a res/u-net keras model
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
#' model <- createResUnetModel2D( c( dim( domainImage ), 1 ),
#'   numberOfOutputs = numberOfLabels )
#'
#' model %>% compile( loss = loss_multilabel_dice_coefficient_error,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( multilabel_dice_coefficient ) )
#'
#' # Comment out the rest due to travis build constraints
#'
#' # Fit the model
#'
#' # track <- model %>% fit( X_train, Y_train,
#' #              epochs = 100, batch_size = 4, verbose = 1, shuffle = TRUE,
#' #              callbacks = list(
#' #                callback_model_checkpoint( "resUnetModelInterimWeights.h5",
#' #                    monitor = 'val_loss', save_best_only = TRUE ),
#' #                callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#' #              ),
#' #              validation_split = 0.2 )
#'
#' # Save the model and/or save the model weights
#'
#' # save_model_hdf5( model, filepath = 'resUnetModel.h5' )
#' # save_model_weights_hdf5( unetModel, filepath = 'resUnetModelWeights.h5' ) )
#'
#' @import keras
#' @export
createResUnetModel2D <- function( inputImageSize,
                                  numberOfOutputs = 1,
                                  numberOfFiltersAtBaseLayer = 32,
                                  bottleNeckBlockDepthSchedule = c( 3, 4 ),
                                  convolutionKernelSize = c( 3, 3 ),
                                  deconvolutionKernelSize = c( 2, 2 ),
                                  dropoutRate = 0.0,
                                  weightDecay = 0.0001,
                                  mode = 'classification'
                                )
{

  simpleBlock2D <- function( input, numberOfFilters, downsample = FALSE,
    upsample = FALSE, convolutionKernelSize = c( 3, 3 ),
    deconvolutionKernelSize = c( 2, 2 ), weightDecay = 0.0, dropoutRate = 0.0 )
    {
    numberOfOutputFilters <- numberOfFilters

    output <- input %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    if( downsample )
      {
      output <- output %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
      }

    output <- output %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, padding = 'same',
      kernel_regularizer = regularizer_l2( weightDecay ) )

    if( upsample )
      {
      output <- output %>%
        layer_conv_2d_transpose( filters = numberOfFilters,
          kernel_size = deconvolutionKernelSize, padding = 'same',
          kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( weightDecay ) )
      output <- output %>% layer_upsampling_2d( size = c( 2, 2 ) )
      }

    if( dropoutRate > 0.0 )
      {
      output <- output %>% layer_dropout( rate = dropoutRate )
      }

    # Modify the input so that it has the same size as the output

    if( downsample )
      {
      input <- input %>% layer_conv_2d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1 ), strides = c( 2, 2 ), padding = 'same' )
      } else if( upsample ) {
      input <- input %>%
        layer_conv_2d_transpose( filters = numberOfOutputFilters,
          kernel_size = c( 1, 1 ), padding = 'same' )
      input <- input %>% layer_upsampling_2d( size = c( 2, 2 ) )
      } else if( numberOfFilters != numberOfOutputFilters ) {
      input <- input %>% layer_conv_2d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1 ), padding = 'same' )
      }

    output <- skipConnection( input, output )

    return( output )
    }

  bottleNeckBlock2D <- function( input, numberOfFilters, downsample = FALSE,
    upsample = FALSE, deconvolutionKernelSize = c( 2, 2 ), weightDecay = 0.0,
    dropoutRate = 0.0 )
    {
    output <- input

    numberOfOutputFilters <- numberOfFilters

    if( downsample )
      {
      output <- output %>% layer_batch_normalization()
      output <- output %>% layer_activation_thresholded_relu( theta = 0 )

      output <- output %>% layer_conv_2d(
        filters = numberOfFilters,
        kernel_size = c( 1, 1 ), strides = c( 2, 2 ),
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }

    output <- output %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    output <- output %>% layer_conv_2d(
      filters = numberOfFilters, kernel_size = c( 1, 1 ),
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    output <- output %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    if( upsample )
      {
      output <- output %>%
        layer_conv_2d_transpose( filters = numberOfFilters,
          kernel_size = deconvolutionKernelSize, padding = 'same',
          kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( weightDecay ) )
      output <- output %>% layer_upsampling_2d( size = c( 2, 2 ) )
      }

    output <- output %>% layer_conv_2d(
      filters = numberOfFilters * 4, kernel_size = c( 1, 1 ),
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    numberOfOutputFilters <- numberOfFilters * 4

    if( dropoutRate > 0.0 )
      {
      output <- output %>% layer_dropout( rate = dropoutRate )
      }

    # Modify the input so that it has the same size as the output

    if( downsample )
      {
      input <- input %>% layer_conv_2d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1 ), strides = c( 2, 2 ), padding = 'same' )
      } else if( upsample ) {
      input <- input %>%
        layer_conv_2d_transpose( filters = numberOfOutputFilters,
          kernel_size = c( 1, 1 ), padding = 'same' )
      input <- input %>% layer_upsampling_2d( size = c( 2, 2 ) )
      } else if( numberOfFilters != numberOfOutputFilters ) {
      input <- input %>% layer_conv_2d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1 ), padding = 'valid' )
      }

    output <- skipConnection( input, output )

    return( output )
    }

  skipConnection <- function( source, target, mergeMode = 'sum' )
    {
    layerList <- list( source, target )

    if( mergeMode == 'sum' )
      {
      output <- layer_add( layerList )
      } else {
      channelAxis <- 1
      if( keras::backend()$image_data_format() == "channels_last" )
        {
        channelAxis <- -1
        }
      output <- layer_concatenate( layerList, axis = channelAxis )
      }

    return( output )
    }

  inputs <- layer_input( shape = inputImageSize )

  encodingLayersWithLongSkipConnections <- list()
  encodingLayerCount <- 1

  # Preprocessing layer

  model <- inputs %>% layer_conv_2d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same',
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( weightDecay ) )

  encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
  encodingLayerCount <- encodingLayerCount + 1

  # Encoding initialization path

  model <- model %>% simpleBlock2D( numberOfFiltersAtBaseLayer,
    downsample = TRUE,
    convolutionKernelSize = convolutionKernelSize,
    deconvolutionKernelSize = deconvolutionKernelSize,
    weightDecay = weightDecay, dropoutRate = dropoutRate )

  encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
  encodingLayerCount <- encodingLayerCount + 1

  # Encoding main path

  numberOfBottleNeckLayers <- length( bottleNeckBlockDepthSchedule )
  for( i in seq_len( numberOfBottleNeckLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    for( j in seq_len( bottleNeckBlockDepthSchedule[i] ) )
      {
      if( j == 1 )
        {
        doDownsample <- TRUE
        } else {
        doDownsample <- FALSE
        }
      model <- model %>% bottleNeckBlock2D( numberOfFilters = numberOfFilters,
        downsample = doDownsample,
        deconvolutionKernelSize = deconvolutionKernelSize,
        weightDecay = weightDecay, dropoutRate = dropoutRate )

      if( j == bottleNeckBlockDepthSchedule[i] )
        {
        encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
        encodingLayerCount <- encodingLayerCount + 1
        }
      }
    }
  encodingLayerCount <- encodingLayerCount - 1

  # Transition path

  numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ numberOfBottleNeckLayers
  model <- model %>%
    bottleNeckBlock2D( numberOfFilters = numberOfFilters,
      downsample = TRUE,
      deconvolutionKernelSize = deconvolutionKernelSize,
      weightDecay = weightDecay, dropoutRate = dropoutRate )

  model <- model %>%
    bottleNeckBlock2D( numberOfFilters = numberOfFilters,
      upsample = TRUE,
      deconvolutionKernelSize = deconvolutionKernelSize,
      weightDecay = weightDecay, dropoutRate = dropoutRate )

  # Decoding main path

  numberOfBottleNeckLayers <- length( bottleNeckBlockDepthSchedule )
  for( i in seq_len( numberOfBottleNeckLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer *
      2 ^ ( numberOfBottleNeckLayers - i )

    for( j in seq_len( bottleNeckBlockDepthSchedule[numberOfBottleNeckLayers - i + 1] ) )
      {
      if( j == bottleNeckBlockDepthSchedule[numberOfBottleNeckLayers - i + 1] )
        {
        doUpsample <- TRUE
        } else {
        doUpsample <- FALSE
        }
      model <- model %>% bottleNeckBlock2D( numberOfFilters = numberOfFilters,
        upsample = doUpsample,
        deconvolutionKernelSize = deconvolutionKernelSize,
        weightDecay = weightDecay, dropoutRate = dropoutRate )

      if( j == 1 )
        {
        model <- model %>% layer_conv_2d( filters = numberOfFilters * 4,
          kernel_size = c( 1, 1 ), padding = 'same' )
        model <- skipConnection( encodingLayersWithLongSkipConnections[[encodingLayerCount]], model )
        encodingLayerCount <- encodingLayerCount - 1
        }
      }
    }

  # Decoding initialization path

  model <- model %>% simpleBlock2D( numberOfFiltersAtBaseLayer,
    upsample = TRUE,
    convolutionKernelSize = convolutionKernelSize,
    deconvolutionKernelSize = deconvolutionKernelSize,
    weightDecay = weightDecay, dropoutRate = dropoutRate )

  # Postprocessing layer

  model <- model %>% layer_conv_2d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, activation = 'relu',
    padding = 'same',
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( weightDecay ) )

  encodingLayerCount <- encodingLayerCount - 1

  model <- skipConnection( encodingLayersWithLongSkipConnections[[encodingLayerCount]], model )

  model <- model %>% layer_batch_normalization()
  model <- model %>% layer_activation_thresholded_relu( theta = 0 )

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
  outputs <- model %>%
    layer_conv_2d( filters = numberOfOutputs,
      kernel_size = c( 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

#' 3-D implementation of the Resnet + U-net deep learning architecture.
#'
#' Creates a keras model of the U-net + ResNet deep learning architecture for
#' image segmentation and regression with the paper available here:
#'
#'         \url{https://arxiv.org/abs/1608.04117}
#'
#' This particular implementation was ported from the following python
#' implementation:
#'
#'         \url{https://github.com/veugene/fcn_maker/}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfOutputs Meaning depends on the \code{mode}.  For
#' 'classification' this is the number of segmentation labels.  For 'regression'
#' this is the number of outputs.
#' @param numberOfFiltersAtBaseLayer number of filters at the beginning and end
#' of the \verb{'U'}.  Doubles at each descending/ascending layer.
#' @param bottleNeckBlockDepthSchedule vector that provides the encoding layer
#' schedule for the number of bottleneck blocks per long skip connection.
#' @param convolutionKernelSize 2-d vector defining the kernel size
#' during the encoding path
#' @param deconvolutionKernelSize 2-d vector defining the kernel size
#' during the decoding
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param weightDecay weighting parameter for L2 regularization of the
#' kernel weights of the convolution layers.  Default = 0.0.
#' @param mode 'classification' or 'regression'.  Default = 'classification'.
#'
#' @return a res/u-net keras model
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
#' model <- createResUnetModel2D( c( dim( domainImage ), 1 ),
#'   numberOfOutputs = numberOfLabels )
#'
#' model %>% compile( loss = loss_multilabel_dice_coefficient_error,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( multilabel_dice_coefficient ) )
#'
#' # Comment out the rest due to travis build constraints
#'
#' # Fit the model
#'
#' # track <- model %>% fit( X_train, Y_train,
#' #              epochs = 100, batch_size = 4, verbose = 1, shuffle = TRUE,
#' #              callbacks = list(
#' #                callback_model_checkpoint( "resUnetModelInterimWeights.h5",
#' #                    monitor = 'val_loss', save_best_only = TRUE ),
#' #                callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#' #              ),
#' #              validation_split = 0.2 )
#'
#' # Save the model and/or save the model weights
#'
#' # save_model_hdf5( model, filepath = 'resUnetModel.h5' )
#' # save_model_weights_hdf5( unetModel, filepath = 'resUnetModelWeights.h5' ) )
#'
#' @import keras
#' @export
createResUnetModel3D <- function( inputImageSize,
                                  numberOfOutputs = 1,
                                  numberOfFiltersAtBaseLayer = 32,
                                  bottleNeckBlockDepthSchedule = c( 3, 4 ),
                                  convolutionKernelSize = c( 3, 3, 3 ),
                                  deconvolutionKernelSize = c( 2, 2, 2 ),
                                  dropoutRate = 0.0,
                                  weightDecay = 0.0001,
                                  mode = 'classification'
                                )
{

  simpleBlock3D <- function( input, numberOfFilters, downsample = FALSE,
    upsample = FALSE, convolutionKernelSize = c( 3, 3, 3 ),
    deconvolutionKernelSize = c( 2, 2, 2 ), weightDecay = 0.0, dropoutRate = 0.0 )
    {
    numberOfOutputFilters <- numberOfFilters

    output <- input %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    if( downsample )
      {
      output <- output %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
      }

    output <- output %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, padding = 'same',
      kernel_regularizer = regularizer_l2( weightDecay ) )

    if( upsample )
      {
      output <- output %>%
        layer_conv_3d_transpose( filters = numberOfFilters,
          kernel_size = deconvolutionKernelSize, padding = 'same',
          kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( weightDecay ) )
      output <- output %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      }

    if( dropoutRate > 0.0 )
      {
      output <- output %>% layer_dropout( rate = dropoutRate )
      }

    # Modify the input so that it has the same size as the output

    if( downsample )
      {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), strides = c( 2, 2, 2 ), padding = 'same' )
      } else if( upsample ) {
      input <- input %>%
        layer_conv_3d_transpose( filters = numberOfOutputFilters,
          kernel_size = c( 1, 1, 1 ), padding = 'same' )
      input <- input %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      } else if( numberOfFilters != numberOfOutputFilters ) {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), padding = 'same' )
      }

    output <- skipConnection( input, output )

    return( output )
    }

  bottleNeckBlock3D <- function( input, numberOfFilters, downsample = FALSE,
    upsample = FALSE, deconvolutionKernelSize = c( 2, 2, 2 ), weightDecay = 0.0,
    dropoutRate = 0.0 )
    {
    output <- input

    numberOfOutputFilters <- numberOfFilters

    if( downsample )
      {
      output <- output %>% layer_batch_normalization()
      output <- output %>% layer_activation_thresholded_relu( theta = 0 )

      output <- output %>% layer_conv_3d(
        filters = numberOfFilters,
        kernel_size = c( 1, 1, 1 ), strides = c( 2, 2, 2 ),
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }

    output <- output %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    output <- output %>% layer_conv_3d(
      filters = numberOfFilters, kernel_size = c( 1, 1, 1 ),
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    output <- output %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    if( upsample )
      {
      output <- output %>%
        layer_conv_3d_transpose( filters = numberOfFilters,
          kernel_size = deconvolutionKernelSize, padding = 'same',
          kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( weightDecay ) )
      output <- output %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      }

    output <- output %>% layer_conv_3d(
      filters = numberOfFilters * 4, kernel_size = c( 1, 1, 1 ),
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    numberOfOutputFilters <- numberOfFilters * 4

    if( dropoutRate > 0.0 )
      {
      output <- output %>% layer_dropout( rate = dropoutRate )
      }

    # Modify the input so that it has the same size as the output

    if( downsample )
      {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), strides = c( 2, 2, 2 ), padding = 'same' )
      } else if( upsample ) {
      input <- input %>%
        layer_conv_3d_transpose( filters = numberOfOutputFilters,
          kernel_size = c( 1, 1, 1 ), padding = 'same' )
      input <- input %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      } else if( numberOfFilters != numberOfOutputFilters ) {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), padding = 'valid' )
      }

    output <- skipConnection( input, output )

    return( output )
    }

  skipConnection <- function( source, target, mergeMode = 'sum' )
    {
    layerList <- list( source, target )

    if( mergeMode == 'sum' )
      {
      output <- layer_add( layerList )
      } else {
      channelAxis <- 1
      if( keras::backend()$image_data_format() == "channels_last" )
        {
        channelAxis <- -1
        }
      output <- layer_concatenate( layerList, axis = channelAxis )
      }

    return( output )
    }

  inputs <- layer_input( shape = inputImageSize )

  encodingLayersWithLongSkipConnections <- list()
  encodingLayerCount <- 1

  # Preprocessing layer

  model <- inputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same',
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( weightDecay ) )

  encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
  encodingLayerCount <- encodingLayerCount + 1

  # Encoding initialization path

  model <- model %>% simpleBlock3D( numberOfFiltersAtBaseLayer,
    downsample = TRUE,
    convolutionKernelSize = convolutionKernelSize,
    deconvolutionKernelSize = deconvolutionKernelSize,
    weightDecay = weightDecay, dropoutRate = dropoutRate )

  encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
  encodingLayerCount <- encodingLayerCount + 1

  # Encoding main path

  numberOfBottleNeckLayers <- length( bottleNeckBlockDepthSchedule )
  for( i in seq_len( numberOfBottleNeckLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    for( j in seq_len( bottleNeckBlockDepthSchedule[i] ) )
      {
      if( j == 1 )
        {
        doDownsample <- TRUE
        } else {
        doDownsample <- FALSE
        }
      model <- model %>% bottleNeckBlock3D( numberOfFilters = numberOfFilters,
        downsample = doDownsample,
        deconvolutionKernelSize = deconvolutionKernelSize,
        weightDecay = weightDecay, dropoutRate = dropoutRate )

      if( j == bottleNeckBlockDepthSchedule[i] )
        {
        encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
        encodingLayerCount <- encodingLayerCount + 1
        }
      }
    }
  encodingLayerCount <- encodingLayerCount - 1

  # Transition path

  numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ numberOfBottleNeckLayers
  model <- model %>%
    bottleNeckBlock3D( numberOfFilters = numberOfFilters,
      downsample = TRUE,
      deconvolutionKernelSize = deconvolutionKernelSize,
      weightDecay = weightDecay, dropoutRate = dropoutRate )

  model <- model %>%
    bottleNeckBlock3D( numberOfFilters = numberOfFilters,
      upsample = TRUE,
      deconvolutionKernelSize = deconvolutionKernelSize,
      weightDecay = weightDecay, dropoutRate = dropoutRate )

  # Decoding main path

  numberOfBottleNeckLayers <- length( bottleNeckBlockDepthSchedule )
  for( i in seq_len( numberOfBottleNeckLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer *
      2 ^ ( numberOfBottleNeckLayers - i )

    for( j in seq_len( bottleNeckBlockDepthSchedule[numberOfBottleNeckLayers - i + 1] ) )
      {
      if( j == bottleNeckBlockDepthSchedule[numberOfBottleNeckLayers - i + 1] )
        {
        doUpsample <- TRUE
        } else {
        doUpsample <- FALSE
        }
      model <- model %>% bottleNeckBlock3D( numberOfFilters = numberOfFilters,
        upsample = doUpsample,
        deconvolutionKernelSize = deconvolutionKernelSize,
        weightDecay = weightDecay, dropoutRate = dropoutRate )

      if( j == 1 )
        {
        model <- model %>% layer_conv_3d( filters = numberOfFilters * 4,
          kernel_size = c( 1, 1, 1 ), padding = 'same' )
        model <- skipConnection( encodingLayersWithLongSkipConnections[[encodingLayerCount]], model )
        encodingLayerCount <- encodingLayerCount - 1
        }
      }
    }

  # Decoding initialization path

  model <- model %>% simpleBlock3D( numberOfFiltersAtBaseLayer,
    upsample = TRUE,
    convolutionKernelSize = convolutionKernelSize,
    deconvolutionKernelSize = deconvolutionKernelSize,
    weightDecay = weightDecay, dropoutRate = dropoutRate )

  # Postprocessing layer

  model <- model %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, activation = 'relu',
    padding = 'same',
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( weightDecay ) )

  encodingLayerCount <- encodingLayerCount - 1

  model <- skipConnection( encodingLayersWithLongSkipConnections[[encodingLayerCount]], model )

  model <- model %>% layer_batch_normalization()
  model <- model %>% layer_activation_thresholded_relu( theta = 0 )

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
  outputs <- model %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = c( 1, 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
