#' 2-D implementation of the deep denoise image super resolution architecture.
#'
#' Creates a keras model of the expanded image super resolution deep learning
#' framework based on the following python implementation:
#'
#'         \url{https://github.com/titu1994/Image-Super-Resolution}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param layers number of architecture layers.
#' @param lowestResolution number of filters at the beginning and end of
#' the architecture.
#' @param convolutionKernelSize 2-D vector defining the kernel size
#' during the encoding path
#' @param poolSize 2-D vector defining the region for each pooling layer.
#' @param strides 2-D vector describing the stride length in each direction.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' }
#' @import keras
#' @importFrom ANTsRCore antsRegistration
#' @export
createDeepDenoiseSuperResolutionModel2D <- function( inputImageSize,
  layers = 2,
  lowestResolution = 64,
  convolutionKernelSize = c( 3, 3 ),
  poolSize = c( 2, 2 ),
  strides = c( 2, 2 ) )
{
  inputs <- layer_input( shape = inputImageSize )

  # encoding layers

  encodingConvolutionLayers <- list()
  for( i in 1:length( 1:layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( i - 1 )

    if( i == 1 )
      {
      conv <- inputs %>%
        layer_conv_2d( filters = numberOfFilters,
          kernel_size = convolutionKernelSize, activation = 'relu',
          padding = 'same' )
      } else {
      conv <- pool %>%
        layer_conv_2d( filters = numberOfFilters,
          kernel_size = convolutionKernelSize, activation = 'relu',
          padding = 'same' )
      }

    encodingConvolutionLayers[[i]] <- conv %>%
      layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same' )

    pool <- encodingConvolutionLayers[[i]] %>%
      layer_max_pooling_2d( pool_size = poolSize, strides = strides,
        padding = 'same' )
    }

  numberOfFilters <- lowestResolution * 2 ^ ( layers )

  outputs <- pool %>%
    layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize,
      activation = 'relu', padding = 'same' )

  # upsampling layers
  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers - i )

    outputs <- outputs %>% layer_upsampling_2d()

    conv <- outputs %>%
      layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same' )
    conv <- conv %>%
      layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same' )

    outputs <- outputs %>%
      layer_add( list( encodingConvolutionLayers[[layers - i + 1]], conv ))
    outputs <- outputs %>% layer_upsampling_2d()
    }

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_2d( filters = numberOfChannels,
    kernel_size = convolutionKernelSize[[2]], activation = "linear",
    padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}

#' 3-D implementation of the deep denoise image super resolution architecture.
#'
#' Creates a keras model of the expanded image super resolution deep learning
#' framework based on the following python implementation:
#'
#'         \url{https://github.com/titu1994/Image-Super-Resolution}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param layers number of architecture layers.
#' @param lowestResolution number of filters at the beginning and end of
#' the architecture.
#' @param convolutionKernelSize 3-D vector defining the kernel size
#' during the encoding path
#' @param poolSize 3-D vector defining the region for each pooling layer.
#' @param strides 3-D vector describing the stride length in each direction.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' }
#' @import keras
#' @export
createDeepDenoiseSuperResolutionModel3D <- function( inputImageSize,
  layers = 2,
  lowestResolution = 64,
  convolutionKernelSize = c( 3, 3, 3 ),
  poolSize = c( 2, 2, 2 ),
  strides = c( 2, 2, 2 ) )
{
  inputs <- layer_input( shape = inputImageSize )

  # encoding layers

  encodingConvolutionLayers <- list()
  for( i in 1:length( 1:layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( i - 1 )

    if( i == 1 )
      {
      conv <- inputs %>%
        layer_conv_3d( filters = numberOfFilters,
          kernel_size = convolutionKernelSize, activation = 'relu',
          padding = 'same' )
      } else {
      conv <- pool %>%
        layer_conv_3d( filters = numberOfFilters,
          kernel_size = convolutionKernelSize, activation = 'relu',
          padding = 'same' )
      }

    encodingConvolutionLayers[[i]] <- conv %>%
      layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same' )

    pool <- encodingConvolutionLayers[[i]] %>%
      layer_max_pooling_3d( pool_size = poolSize, strides = strides,
        padding = 'same' )
    }

  numberOfFilters <- lowestResolution * 2 ^ ( layers )

  outputs <- pool %>%
    layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize,
      activation = 'relu', padding = 'same' )

  # upsampling layers
  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers - i )

    outputs <- outputs %>% layer_upsampling_3d()

    conv <- outputs %>%
      layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same' )
    conv <- conv %>%
      layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same' )

    outputs <- outputs %>%
      layer_add( list( encodingConvolutionLayers[[layers - i + 1]], conv ))
    outputs <- outputs %>% layer_upsampling_3d()
    }

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfChannels,
    kernel_size = convolutionKernelSize[[2]], activation = "linear",
    padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}
