#' 2-D implementation of the ResNet image super resolution architecture.
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
#' @param convolutionKernelSize a vector specifying the kernel size for
#' convolution.
#' @param numberOfFilters the number of filters for each encoding layer.
#' @param numberOfResidualBlocks the number of residual blocks.
#' @param numberOfResNetBlocks the number of resNet blocks.  Each block
#' will double the upsampling amount.
#'
#' @return a keras model for ResNet image super resolution
#' @author Tustison NJ
#' @examples
#' createResNetSuperResolutionModel2D(c(256, 256, 3))
#' @import keras
#' @export
createResNetSuperResolutionModel2D <- function( inputImageSize,
  convolutionKernelSize = c( 3, 3 ), numberOfFilters = 64,
  numberOfResidualBlocks = 5, numberOfResNetBlocks = 1 )
{

  residualBlock2D <- function( model, numberOfFilters, convolutionKernelSize )
    {
    block <- model %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'linear',
      padding = 'same' )

    block <- block %>% layer_batch_normalization()
    block <- block %>% layer_activation( activation = 'relu' )
    block <- block %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'linear',
      padding = 'same' )
    block <- block %>% layer_batch_normalization()
    block <- layer_add( list( model, block ) )

    return( block )
    }

  upscaleBlock2D <- function( model, numberOfFilters, convolutionKernelSize )
    {
    block <- model %>% layer_upsampling_2d()
    block <- block %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'relu',
      padding = 'same' )
    return( block )
    }

  resNetBlock2D <- function( inputs,  numberOfFilters,
    convolutionKernelSize, numberOfResidualBlocks  )
  {
  outputs <- inputs %>% layer_conv_2d( filters = numberOfFilters,
    kernel_size = convolutionKernelSize, activation = 'relu',
    padding = 'same' )

  residualBlocks <- residualBlock2D( outputs,
    numberOfFilters = numberOfFilters,
    convolutionKernelSize = convolutionKernelSize )
  for( i in seq_len( numberOfResidualBlocks ) )
    {
    residualBlocks <- residualBlock2D( residualBlocks, numberOfFilters,
      convolutionKernelSize )
    }
  outputs <- layer_add( list( residualBlocks, outputs ) )

  outputs <- upscaleBlock2D( outputs, numberOfFilters,
    convolutionKernelSize )
  return( outputs )
  }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- resNetBlock2D( inputs, numberOfFilters,
    convolutionKernelSize, numberOfResidualBlocks  )
  if( numberOfResNetBlocks > 1 )
    {
    for( n in 2:numberOfResNetBlocks )
      {
      outputs <- resNetBlock2D( outputs,  numberOfFilters,
        convolutionKernelSize, numberOfResidualBlocks )
      }
    }

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_2d( filters = numberOfChannels,
    kernel_size = convolutionKernelSize, activation = 'linear',
    padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}

#' 3-D implementation of the ResNet image super resolution architecture.
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
#' @param convolutionKernelSize a vector specifying the kernel size for
#' convolution.
#' @param numberOfFilters the number of filters for each encoding layer.
#' @param numberOfResidualBlocks the number of residual blocks.
#' @param numberOfResNetBlocks the number of resNet blocks.  Each block
#' will double the upsampling amount.
#'
#' @return a keras model for ResNet image super resolution
#' @author Tustison NJ
#' @examples
#' inputImageSize = c(256, 256, 30, 1)
#' createResNetSuperResolutionModel3D(inputImageSize)
#' @import keras
#' @export
createResNetSuperResolutionModel3D <- function( inputImageSize,
  convolutionKernelSize = c( 3, 3, 3 ), numberOfFilters = 64,
  numberOfResidualBlocks = 5, numberOfResNetBlocks = 1 )
{

  residualBlock3D <- function( model, numberOfFilters, convolutionKernelSize )
    {
    block <- model %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'linear',
      padding = 'same' )

    block <- block %>% layer_batch_normalization()
    block <- block %>% layer_activation( activation = 'relu' )
    block <- block %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'linear',
      padding = 'same' )
    block <- block %>% layer_batch_normalization()
    block <- layer_add( list( model, block ) )

    return( block )
    }

  upscaleBlock3D <- function( model, numberOfFilters, convolutionKernelSize )
    {
    block <- model %>% layer_upsampling_3d()
    block <- block %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'relu',
      padding = 'same' )

    return( block )
    }

  resNetBlock3D <- function( model,  numberOfFilters,
    convolutionKernelSize, numberOfResidualBlocks  )
    {
    outputs <- model %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'relu',
      padding = 'same' )

    residualBlocks <- residualBlock3D( outputs, numberOfFilters, convolutionKernelSize )
    for( i in seq_len( numberOfResidualBlocks ) )
      {
      residualBlocks <- residualBlock3D( residualBlocks, numberOfFilters,
        convolutionKernelSize )
      }
    outputs <- layer_add( list( residualBlocks, outputs ) )

    outputs <- upscaleBlock3D( outputs, numberOfFilters,
      convolutionKernelSize )
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- resNetBlock3D( inputs, numberOfFilters,
    convolutionKernelSize, numberOfResidualBlocks  )
  if( numberOfResNetBlocks > 1 )
    {
    for( n in 2:numberOfResNetBlocks )
      {
      outputs <- resNetBlock3D( outputs,  numberOfFilters,
        convolutionKernelSize, numberOfResidualBlocks )
      }
    }

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfChannels,
    kernel_size = convolutionKernelSize, activation = 'linear',
    padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}
