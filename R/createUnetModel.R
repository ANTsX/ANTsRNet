#' 2-D implementation of the U-net deep learning architecture.
#'
#' Creates a keras model of the U-net deep learning architecture for image
#' segmentation and regression.  More information is provided at the authors'
#' website:
#'
#'         \url{https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/}
#'
#' with the paper available here:
#'
#'         \url{https://arxiv.org/abs/1505.04597}
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         \url{https://github.com/joelthelion/ultrasound-nerve-segmentation}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfOutputs Meaning depends on the \code{mode}.  For
#' 'classification' this is the number of segmentation labels.  For 'regression'
#' this is the number of outputs.
#' @param numberOfLayers number of encoding/decoding layers.
#' @param numberOfFiltersAtBaseLayer number of filters at the beginning and end
#' of the \verb{'U'}.  Doubles at each descending/ascending layer.
#' @param numberOfFilters vector explicitly setting the number of filters at
#' each layer.  One can either set this or \code{numberOfLayers} and
#' \code{numberOfFiltersAtBaseLayer}.  Default = NULL.
#' @param convolutionKernelSize 2-d vector defining the kernel size
#' during the encoding path.
#' @param deconvolutionKernelSize 2-d vector defining the kernel size
#' during the decoding.
#' @param poolSize 2-d vector defining the region for each pooling layer.
#' @param strides 2-d vector describing the stride length in each direction.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param weightDecay weighting parameter for L2 regularization of the
#' kernel weights of the convolution layers.  Default = 0.0.
#' @param nnUnetActivationStyle boolean for "nnu-net variant" from
#'  \url{https://pubmed.ncbi.nlm.nih.gov/30802813/}.
#' @param addAttentionGating boolean for "attention u-net variant" activation from
#'  \url{https://pubmed.ncbi.nlm.nih.gov/33288961/}.
#' @param mode 'classification' or 'regression' or 'sigmoid'.
#'
#' @return a u-net keras model
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
#' model <- createUnetModel2D( c( dim( domainImage ), 1 ),
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
#'   metrics = metric_multilabel_dice_coefficient )
#'
#' # Comment out the rest due to travis build constraints
#'
#' # Fit the model
#'
#' # track <- model %>% fit( X_train, Y_train,
#' #              epochs = 100, batch_size = 5, verbose = 1, shuffle = TRUE,
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
createUnetModel2D <- function( inputImageSize,
                               numberOfOutputs = 2,
                               numberOfLayers = 4,
                               numberOfFiltersAtBaseLayer = 32,
                               numberOfFilters = NULL,
                               convolutionKernelSize = c( 3, 3 ),
                               deconvolutionKernelSize = c( 2, 2 ),
                               poolSize = c( 2, 2 ),
                               strides = c( 2, 2 ),
                               dropoutRate = 0.0,
                               weightDecay = 0.0,
                               nnUnetActivationStyle = FALSE,
                               addAttentionGating = FALSE,
                               mode = c( 'classification', 'regression', 'sigmoid' )
                             )
{

  nnUnetActivation <- function( x )
    {
    x <- x %>% layer_instance_normalization()
    x <- x %>% layer_activation_leaky_relu( alpha = 0.01 )
    return( x )
    }

  attentionGate2D <- function( x, g, interShape )
    {
    xTheta <- x %>% layer_conv_2d( filters = interShape, kernel_size = c( 1L, 1L ),
      strides = c( 1L, 1L ) )
    gPhi <- g %>% layer_conv_2d( filters = interShape, kernel_size = c( 1L, 1L ),
      strides = c( 1L, 1L ) )
    f <- layer_add( list( xTheta, gPhi ) ) %>% layer_activation_relu()
    fPsi <- f %>% layer_conv_2d( filters = 1L, kernel_size = c( 1L, 1L ),
      strides = c( 1L, 1L ) )
    alpha <- fPsi %>% layer_activation( activation = "sigmoid" )
    attention <- layer_multiply( list( x, alpha ) )
    return( attention )
    }

  mode <- match.arg( mode )

  if( ! is.null( numberOfFilters ) )
    {
    numberOfLayers <- length( numberOfFilters )
    } else {
    numberOfFilters <- rep( 0, numberOfLayers )
    for( i in seq_len( numberOfLayers ) )
      {
      numberOfFilters[i] <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )
      }
    }

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  encodingConvolutionLayers <- list()
  for( i in seq_len( numberOfLayers ) )
    {
    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_2d( filters = numberOfFilters[i],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      } else {
      conv <- pool %>% layer_conv_2d( filters = numberOfFilters[i],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }
    if( nnUnetActivationStyle == TRUE )
      {
      conv <- nnUnetActivation( conv )
      } else {
      conv <- conv %>% layer_activation_relu()
      }
    if( dropoutRate > 0.0 )
      {
      conv <- conv %>% layer_dropout( rate = dropoutRate )
      }
    conv <- conv %>% layer_conv_2d( filters = numberOfFilters[i],
      kernel_size = convolutionKernelSize, padding = 'same' )
    if( nnUnetActivationStyle == TRUE )
      {
      conv <- nnUnetActivation( conv )
      } else {
      conv <- conv %>% layer_activation_relu()
      }

    encodingConvolutionLayers[[i]] <- conv

    if( i < numberOfLayers )
      {
      pool <- encodingConvolutionLayers[[i]] %>%
        layer_max_pooling_2d( pool_size = poolSize, strides = strides,
        padding = 'same' )
      }
    }

  # Decoding path

  outputs <- encodingConvolutionLayers[[numberOfLayers]]
  for( i in 2:numberOfLayers )
    {
    deconv <- outputs %>%
      layer_conv_2d_transpose( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = deconvolutionKernelSize,
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    if( nnUnetActivationStyle == TRUE )
      {
      deconv <- nnUnetActivation( deconv )
      }
    deconv <- deconv %>% layer_upsampling_2d( size = poolSize )

    if( addAttentionGating == TRUE )
      {
      outputs <- attentionGate2D( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]],
        as.integer( numberOfFilters[numberOfLayers - i + 1] / 4 ) )
      outputs <- layer_concatenate( list( deconv, outputs ), axis = 3 )
      } else {
      outputs <- layer_concatenate( list( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]] ),
        axis = 3 )
      }

    outputs <- outputs %>%
      layer_conv_2d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    if( nnUnetActivationStyle == TRUE )
      {
      outputs <- nnUnetActivation( outputs )
      } else {
      outputs <- outputs %>% layer_activation_relu()
      }

    if( dropoutRate > 0.0 )
      {
      outputs <- outputs %>% layer_dropout( rate = dropoutRate )
      }

    outputs <- outputs %>%
      layer_conv_2d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    if( nnUnetActivationStyle == TRUE )
      {
      outputs <- nnUnetActivation( outputs )
      } else {
      outputs <- outputs %>% layer_activation_relu()
      }
    }

  convActivation <- ''
  if( mode == 'sigmoid' )
    {
    convActivation <- 'sigmoid'
    } else if( mode == 'classification' ) {
    convActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>%
    layer_conv_2d( filters = numberOfOutputs,
      kernel_size = c( 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

#' 3-D image segmentation implementation of the U-net deep learning architecture.
#'
#' Creates a keras model of the U-net deep learning architecture for image
#' segmentation and regression.  More information is provided at the authors'
#' website:
#'
#'         \url{https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/}
#'
#' with the paper available here:
#'
#'         \url{https://arxiv.org/abs/1505.04597}
#'
#' This particular implementation was influenced by the following python
#' implementation:
#'
#'         \url{https://github.com/joelthelion/ultrasound-nerve-segmentation}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfOutputs Meaning depends on the \code{mode}.  For
#' 'classification' this is the number of segmentation labels.  For 'regression'
#' this is the number of outputs.
#' @param numberOfLayers number of encoding/decoding layers.
#' @param numberOfFiltersAtBaseLayer number of filters at the beginning and end
#' of the \verb{'U'}.  Doubles at each descending/ascending layer.
#' @param numberOfFilters vector explicitly setting the number of filters at
#' each layer.  One can either set this or \code{numberOfLayers} and
#' @param convolutionKernelSize 3-d vector defining the kernel size
#' during the encoding path.
#' @param deconvolutionKernelSize 3-d vector defining the kernel size
#' during the decoding.
#' @param poolSize 3-d vector defining the region for each pooling layer.
#' @param strides 3-d vector describing the stride length in each direction.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param weightDecay weighting parameter for L2 regularization of the
#' kernel weights of the convolution layers.  Default = 0.0.
#' @param nnUnetActivationStyle boolean for "nnu-net variant" from
#'  \url{https://pubmed.ncbi.nlm.nih.gov/30802813/}.
#' @param addAttentionGating boolean for "attention u-net variant" from
#  https://pubmed.ncbi.nlm.nih.gov/30802813/.
#' @param mode 'classification' or 'regression' or 'sigmoid'.
#'
#' @return a u-net keras model
#' @author Tustison NJ
#' @examples
#' # Simple examples, must run successfully and quickly. These will be tested.
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel3D( c( 64, 64, 64, 1 ) )
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
#'   metrics = metric_multilabel_dice_coefficient )
#'
#' print( model )
#'
#' @import keras
#' @export
createUnetModel3D <- function( inputImageSize,
                               numberOfOutputs = 2,
                               numberOfLayers = 4,
                               numberOfFiltersAtBaseLayer = 32,
                               numberOfFilters = NULL,
                               convolutionKernelSize = c( 3, 3, 3 ),
                               deconvolutionKernelSize = c( 2, 2, 2 ),
                               poolSize = c( 2, 2, 2 ),
                               strides = c( 2, 2, 2 ),
                               dropoutRate = 0.0,
                               weightDecay = 0.0,
                               nnUnetActivationStyle = FALSE,
                               addAttentionGating = FALSE,
                               mode = c( 'classification', 'regression', 'sigmoid' )
                             )
{

  nnUnetActivation <- function( x )
    {
    x <- x %>% layer_instance_normalization()
    x <- x %>% layer_activation_leaky_relu( alpha = 0.01 )
    return( x )
    }

  attentionGate3D <- function( x, g, interShape )
    {
    xTheta <- x %>% layer_conv_3d( filters = interShape, kernel_size = c( 1L, 1L, 1L ),
      strides = c( 1L, 1L, 1L ) )
    gPhi <- g %>% layer_conv_3d( filters = interShape, kernel_size = c( 1L, 1L, 1L ),
      strides = c( 1L, 1L, 1L ) )
    f <- layer_add( list( xTheta, gPhi ) ) %>% layer_activation_relu()
    fPsi <- f %>% layer_conv_3d( filters = 1L, kernel_size = c( 1L, 1L, 1L ),
      strides = c( 1L, 1L, 1L ) )
    alpha <- fPsi %>% layer_activation( activation = "sigmoid" )
    attention <- layer_multiply( list( x, alpha ) )
    return( attention )
    }

  mode <- match.arg( mode )

  if( ! is.null( numberOfFilters ) )
    {
    numberOfLayers <- length( numberOfFilters )
    } else {
    numberOfFilters <- rep( 0, numberOfLayers )
    for( i in seq_len( numberOfLayers ) )
      {
      numberOfFilters[i] <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )
      }
    }

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  encodingConvolutionLayers <- list()
  for( i in seq_len( numberOfLayers ) )
    {
    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_3d( filters = numberOfFilters[i],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      } else {
      conv <- pool %>% layer_conv_3d( filters = numberOfFilters[i],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }
    if( nnUnetActivationStyle == TRUE )
      {
      conv <- nnUnetActivation( conv )
      } else {
      conv <- conv %>% layer_activation_relu()
      }
    if( dropoutRate > 0.0 )
      {
      conv <- conv %>% layer_dropout( rate = dropoutRate )
      }
    conv <- conv %>% layer_conv_3d( filters = numberOfFilters[i],
      kernel_size = convolutionKernelSize, padding = 'same' )
    if( nnUnetActivationStyle == TRUE )
      {
      conv <- nnUnetActivation( conv )
      } else {
      conv <- conv %>% layer_activation_relu()
      }

    encodingConvolutionLayers[[i]] <- conv

    if( i < numberOfLayers )
      {
      pool <- encodingConvolutionLayers[[i]] %>%
        layer_max_pooling_3d( pool_size = poolSize, strides = strides,
        padding = 'same' )
      }
    }

  # Decoding path

  outputs <- encodingConvolutionLayers[[numberOfLayers]]
  for( i in 2:numberOfLayers )
    {
    deconv <- outputs %>%
      layer_conv_3d_transpose( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = deconvolutionKernelSize,
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    if( nnUnetActivationStyle == TRUE )
      {
      deconv <- nnUnetActivation( deconv )
      }
    deconv <- deconv %>% layer_upsampling_3d( size = poolSize )

    if( addAttentionGating == TRUE )
      {
      outputs <- attentionGate3D( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]],
        as.integer( numberOfFilters[numberOfLayers - i + 1] / 4 ) )
      outputs <- layer_concatenate( list( deconv, outputs ), axis = 4 )
      } else {
      outputs <- layer_concatenate( list( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]] ),
        axis = 4 )
      }

    outputs <- outputs %>%
      layer_conv_3d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    if( nnUnetActivationStyle == TRUE )
      {
      outputs <- nnUnetActivation( outputs )
      } else {
      outputs <- outputs %>% layer_activation_relu()
      }

    if( dropoutRate > 0.0 )
      {
      outputs <- outputs %>% layer_dropout( rate = dropoutRate )
      }

    outputs <- outputs %>%
      layer_conv_3d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = convolutionKernelSize, padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    if( nnUnetActivationStyle == TRUE )
      {
      outputs <- nnUnetActivation( outputs )
      } else {
      outputs <- outputs %>% layer_activation_relu()
      }
    }

  convActivation <- ''
  if( mode == 'sigmoid' )
    {
    convActivation <- 'sigmoid'
    } else if( mode == 'classification' ) {
    convActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = c( 1, 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
