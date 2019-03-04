#' 2-D implementation of the EDSR super resolution architecture.
#'
#' Creates a keras model of the expanded image super resolution deep learning
#' framework based on EDSR.
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param convolutionKernelSize a vector specifying the kernel size for
#' convolution.
#' @param numberOfFilters the number of filters for each encoding layer.
#' @param numberOfResidualBlocks the number of residual blocks.
#' @param scale the upsampling amount, 2, 4 or 8
#' @param numberOfOutputs the number of data targets, e.g. 2 for 2 targets
#'
#' @return a keras model for EDSR image super resolution
#' @author Tustison NJ, Avants BB
#' @examples
#' \dontrun{
#' }
#' @import keras
#' @export
createEnhancedDeepSuperResolutionModel2D <- function(
  inputImageSize,
  convolutionKernelSize = c( 3, 3 ),
  numberOfFilters = 256,
  numberOfResidualBlocks = 32,
  scale = 2,
  numberOfOutputs = 1 )
{

  residualBlock2D <- function( model, numberOfFilters, convolutionKernelSize )
    {
    block <- model %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'relu',
      padding = 'same' )
    block <- block %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, activation = 'linear',
      padding = 'same' )
    block <- layer_add( list( model, block ) )

    return( block )
    }

  upscaleBlock2D <- function( model,
    numberOfFilters, nChannels,
    convolutionKernelSize, scale = 2 )
    {
    interp = 'bilinear'
    interp = "nearest"
    block <- model %>% layer_upsampling_2d(
      interpolation = interp )
    if ( scale == 4 )
      block <- block %>% layer_upsampling_2d(
        interpolation = interp )
    if ( scale == 8 )
      block <- block %>% layer_upsampling_2d(
        interpolation = interp )
    return( block )
    }

  upscaleBlock2DConv <- function( model,
    numberOfFilters, nChannels,
    convolutionKernelSize, scale = 2 ) {
    kernelSize = c( 4, 4 )
    strides =  c( 2, 2 )
    block <- model %>% layer_conv_2d_transpose( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )%>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )
    if ( scale == 4 )
        block <- block %>% layer_conv_2d_transpose( filters = numberOfFilters,
          kernel_size = kernelSize, strides = strides,
          kernel_initializer = 'glorot_uniform', padding = 'same' )%>% layer_activation_parametric_relu(
            alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )
    if ( scale == 8 )
        block <- block %>% layer_conv_2d_transpose( filters = numberOfFilters,
          kernel_size = kernelSize, strides = strides,
          kernel_initializer = 'glorot_uniform', padding = 'same' )%>% layer_activation_parametric_relu(
            alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

  }

  inputs <- layer_input( shape = inputImageSize )
  outputsX = residualBlocks = layer_conv_2d( inputs,
    filters = numberOfFilters,
    kernel_size = convolutionKernelSize,
    padding = 'same' )

  for( i in 1:numberOfResidualBlocks )
    {
    residualBlocks <- residualBlock2D(
      residualBlocks, numberOfFilters,
      convolutionKernelSize )
    }
  residualBlocks = layer_conv_2d( residualBlocks,
      filters = numberOfFilters,
      kernel_size = convolutionKernelSize,
      padding = 'same' )
  outputsX = layer_add( list( outputsX, residualBlocks ) )


  outputs <- upscaleBlock2DConv( outputsX, numberOfFilters,
          convolutionKernelSize, scale = scale )

  numberOfChannels <- tail( inputImageSize, 1 )
  outputs <- outputs %>% layer_conv_2d(
    filters = numberOfChannels,
    kernel_size = convolutionKernelSize,
    activation = 'linear',
    padding = 'same' )

  if ( numberOfOutputs == 1 ) {
    srModel <- keras_model(
      inputs = inputs,
      outputs = outputs )
    } else {
      olist = list()
      for ( k in 1:numberOfOutputs ) olist[[ k ]] = outputs
      srModel <- keras_model(
        inputs = inputs,
        outputs = olist )
    }


  return( srModel )
}
