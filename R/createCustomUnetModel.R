#' Implementation of the "NoBrainer" U-net architecture
#'
#' Creates a keras model implementation of the u-net architecture
#' avaialable here:
#'
#'         \url{https://github.com/neuronets/nobrainer/}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#'
#' @return a u-net keras model
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#'
#' model <- createNoBrainerUnetModel3D( list( NULL, NULL, NULL, 1 ) )
#'
#' @import keras
#' @export
createNoBrainerUnetModel3D <- function( inputImageSize )
{

  numberOfOutputs <- 1
  numberOfFiltersAtBaseLayer <- 16
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  outputs <- inputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 16,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  # Decoding path

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 16,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip3, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip2, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip1, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  convActivation <- ''
  if( numberOfOutputs == 1 )
    {
    convActivation <- 'sigmoid'
    } else {
    convActivation <- 'softmax'
    }

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = 1, activation = convActivation )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}


#' Implementation of the "HippMapp3r" U-net architecture
#'
#' Creates a keras model implementation of the u-net architecture
#' described here:
#'
#'     \url{https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.24811}
#'
#' with the implementation available here:
#'
#'     \url{https://github.com/mgoubran/HippMapp3r}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param doFirstNetwork boolean dictating if the model built should be the
#' first (initial) network or second (refinement) network.
#'
#' @return a u-net keras model
#' @author Tustison NJ
#' @examples
#' \dontrun{
#'
#' model1 <- createHippMapp3rUnetModel3D( c( 160, 160, 128, 1 ), doFirstNetwork = TRUE )
#' model2 <- createHippMapp3rUnetModel3D( c( 112, 112, 64, 1 ), doFirstNetwork = FALSE )
#'
#' json_config <- model_to_json( model1 )
#' writeLines( json_config, "/Users/ntustison/Desktop/model1_config.json" )
#'
#' }
#' @import keras
#' @export
createHippMapp3rUnetModel3D <- function( inputImageSize,
                                         doFirstNetwork = TRUE )
{
  layer_convB_3d <- function( input, numberOfFilters, kernelSize = 3, strides = 1 )
    {
    block <- input %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides, padding = 'same' )
    block <- block %>% layer_instance_normalization( axis = 5 )
    block <- block %>% layer_activation_leaky_relu()

    return( block )
    }

  residualBlock3D <- function( input, numberOfFilters )
    {
    block <- layer_convB_3d( input, numberOfFilters )
    block <- block %>% layer_spatial_dropout_3d( rate = 0.3 )
    block <- layer_convB_3d( block, numberOfFilters )

    return( block )
    }

  upsampleBlock3D <- function( input, numberOfFilters )
    {
    block <- input %>% layer_upsampling_3d()
    block <- layer_convB_3d( block, numberOfFilters )

    return( block )
    }

  featureBlock3D <- function( input, numberOfFilters )
    {
    block <- layer_convB_3d( input, numberOfFilters )
    block <- layer_convB_3d( block, numberOfFilters, kernelSize = 1 )

    return( block )
    }

  numberOfFiltersAtBaseLayer <- 16

  numberOfLayers <- 6
  if( doFirstNetwork == FALSE )
    {
    numberOfLayers <- 5
    }

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  add <- NULL

  encodingConvolutionLayers <- list()
  for( i in seq_len( numberOfLayers ) )
    {
    numberOfFilters = numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    conv <- NULL
    if( i == 1 )
      {
      conv <- layer_convB_3d( inputs, numberOfFilters )
      } else {
      conv <- layer_convB_3d( add, numberOfFilters, strides = 2 )
      }
    residualBlock <- residualBlock3D( conv, numberOfFilters )
    add <- list( conv, residualBlock ) %>% layer_add()

    encodingConvolutionLayers[[i]] <- add
    }

  # Decoding path

  outputs <- unlist( tail( encodingConvolutionLayers, 1 ) )[[1]]

  # 256
  numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( numberOfLayers - 2 )
  outputs <- upsampleBlock3D( outputs, numberOfFilters )

  if( doFirstNetwork == TRUE )
    {
    # 256, 128
    outputs <- list( encodingConvolutionLayers[[5]], outputs ) %>%
      layer_concatenate()
    outputs <- featureBlock3D( outputs, numberOfFilters )
    numberOfFilters <- numberOfFilters / 2
    outputs <- upsampleBlock3D( outputs, numberOfFilters )
    }

  # 128, 64
  outputs <- list( encodingConvolutionLayers[[4]], outputs ) %>%
    layer_concatenate()
  outputs <- featureBlock3D( outputs, numberOfFilters )
  numberOfFilters <- numberOfFilters / 2
  outputs <- upsampleBlock3D( outputs, numberOfFilters )

  # 64, 32
  outputs <- list( encodingConvolutionLayers[[3]], outputs ) %>%
    layer_concatenate()
  feature64 <- featureBlock3D( outputs, numberOfFilters )
  numberOfFilters <- numberOfFilters / 2
  outputs <- upsampleBlock3D( feature64, numberOfFilters )
  back64 <- NULL
  if( doFirstNetwork == TRUE )
    {
    back64 <- layer_convB_3d( feature64, 1, 1 )
    } else {
    back64 <- feature64 %>% layer_conv_3d( filters = 1, kernel_size = 1 )
    }
  back64 <- back64 %>% layer_upsampling_3d()

  # 32, 16
  outputs <- list( encodingConvolutionLayers[[2]], outputs ) %>%
    layer_concatenate()
  feature32 <- featureBlock3D( outputs, numberOfFilters )
  numberOfFilters <- numberOfFilters / 2
  outputs <- upsampleBlock3D( feature32, numberOfFilters )
  back32 <- NULL
  if( doFirstNetwork == TRUE )
    {
    back32 <- layer_convB_3d( feature32, 1, 1 )
    } else {
    back32 <- feature32 %>% layer_conv_3d( filters = 1, kernel_size = 1 )
    }
  back32 <- list( back64, back32 ) %>% layer_add()
  back32 <- back32 %>% layer_upsampling_3d()

  # final
  outputs <- list( encodingConvolutionLayers[[1]], outputs ) %>%
    layer_concatenate()
  outputs <- layer_convB_3d( outputs, numberOfFilters, 3 )
  outputs <- layer_convB_3d( outputs, numberOfFilters, 1 )
  if( doFirstNetwork == TRUE )
    {
    outputs <- layer_convB_3d( outputs, 1, 1 )
    } else {
    outputs <- outputs %>% layer_conv_3d( filters = 1, kernel_size = 1 )
    }
  outputs <- list( back32, outputs ) %>% layer_add()
  outputs <- outputs %>% layer_activation( 'sigmoid' )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

#' Implementation of the sysu_media U-net architecture
#'
#' Creates a keras model implementation of the u-net architecture
#' in the 2017 MICCAI WMH challenge by the sysu_medial team described
#' here:
#'
#'     \url{https://pubmed.ncbi.nlm.nih.gov/30125711/}
#'
#' or targeting the claustrum:
#'
#'     \url{https://arxiv.org/abs/2008.03465}
#'
#' with the original implementations available at
#'
#'     \url{https://github.com/hongweilibran/wmh_ibbmTum}
#'
#' and
#'
#'     \url{https://github.com/hongweilibran/claustrum_multi_view},
#'
#' respectively.
#'
#' @param inputImageSize Used for specifying the input tensor shape.
#' This will be \code{c(200, 200, 2)} for t1/flair input and
#' \code{c(200, 200, 1)} for flair-only input for wmhs.  For the
#' claustrum, it is \code{c(180, 180, 1)}.
#'
#' @return a u-net keras model
#' @author Tustison NJ
#' @examples
#' \dontrun{
#'
#' model <- createSysuMediaUnetModel2D( c( 200, 200, 1 ) )
#'
#' }
#' @import keras
#' @export
createSysuMediaUnetModel2D <- function( inputImageSize, anatomy = c( "wmh", "claustrum" ) )
{
  getCropShape <- function( targetLayer, referenceLayer )
    {
    K <- keras::backend()

    cropShape <- list()

    delta <- K$int_shape( targetLayer )[[2]] - K$int_shape( referenceLayer )[[2]]
    if( delta %% 2 != 0 )
      {
      cropShape[[1]] <- c( as.integer( delta / 2 ), as.integer( delta / 2 ) + 1L )
      } else {
      cropShape[[1]] <- c( as.integer( delta / 2 ), as.integer( delta / 2 ) )
      }

    delta <- K$int_shape( targetLayer )[[3]] - K$int_shape( referenceLayer )[[3]]
    if( delta %% 2 != 0 )
      {
      cropShape[[2]] <- c( as.integer( delta / 2 ), as.integer( delta / 2 ) + 1L )
      } else {
      cropShape[[2]] <- c( as.integer( delta / 2 ), as.integer( delta / 2 ) )
      }

    return( cropShape )
    }

  inputs <- layer_input( shape = inputImageSize )

  if( anatomy == "wmh" )
    {
    numberOfFilters <- as.integer( c( 64, 96, 128, 256, 512 ) )
    } else if( anatomy == "claustrum" ) {
    numberOfFilters <- as.integer( c( 32, 64, 96, 128, 256 ) )
    } else {
    stop( "Unrecognized anatomy" )
    }

  # encoding layers

  encodingLayers <- list()

  outputs <- inputs
  for( i in seq.int( length( numberOfFilters ) ) )
    {
    kernel1 <- 3L
    kernel2 <- 3L
    if( i == 1 && anatomy == "wmh" )
      {
      kernel1 <- 5L
      kernel2 <- 5L
      } else if( i == 4 ) {
      kernel1 <- 3L
      kernel2 <- 4L
      }
    outputs <- outputs %>% layer_conv_2d( numberOfFilters[i], kernel_size = kernel1, padding = 'same' )
    outputs <- outputs %>% layer_activation_relu()
    outputs <- outputs %>% layer_conv_2d( numberOfFilters[i], kernel_size = kernel2, padding = 'same' )
    outputs <- outputs %>% layer_activation_relu()
    encodingLayers[[i]] <- outputs
    if( i < 5 )
      {
      outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2L, 2L ) )
      }
    }

  # decoding layers

  for( i in seq.int( from = length( encodingLayers ) - 1, to = 1, by = -1 ) )
    {
    upsampleLayer <- outputs %>% layer_upsampling_2d( size = c( 2L, 2L ) )
    cropShape <- getCropShape( encodingLayers[[i]], upsampleLayer )
    croppedLayer <- encodingLayers[[i]] %>% layer_cropping_2d( cropping = cropShape )
    outputs <- layer_concatenate( list( upsampleLayer, croppedLayer ), axis = -1L )
    outputs <- outputs %>% layer_conv_2d( numberOfFilters[i], kernel_size = 3L, padding = 'same' )
    outputs <- outputs %>% layer_activation_relu()
    outputs <- outputs %>% layer_conv_2d( numberOfFilters[i], kernel_size = 3L, padding = 'same' )
    outputs <- outputs %>% layer_activation_relu()
    }

  # final

  cropShape <- getCropShape( inputs, outputs )
  outputs <- outputs %>% layer_zero_padding_2d( padding = cropShape )
  outputs <- outputs %>% layer_conv_2d( 1L, kernel_size = 1L, activation = 'sigmoid', padding = 'same' )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}