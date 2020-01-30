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
#' @param numberOfLossFunctions the number of data targets, e.g. 2 for 2 targets
#' @param numberOfOutputChannels the number of ouput channels
#' @param doBatchNormalization boolean for include BN in the residual blocks
#' @param interpolation nearest, bilinear or conv for upscaling block
#'
#' @return a keras model for EDSR image super resolution
#' @author Tustison NJ, Avants BB
#' @examples
#' createEnhancedDeepSuperResolutionModel2D(c( 28, 28, 1 ))
#' createEnhancedDeepSuperResolutionModel2D(c( 28, 28, 1 ),
#' doBatchNormalization = TRUE,
#' interpolation = "conv", scale = 4)
#' createEnhancedDeepSuperResolutionModel2D(c( 28, 28, 1 ),
#' doBatchNormalization = TRUE,
#' numberOfLossFunctions = 2,
#' interpolation = "conv", scale = 8)
#' @import keras
#' @export
createEnhancedDeepSuperResolutionModel2D <- function(
  inputImageSize,
  convolutionKernelSize = c( 3, 3 ),
  numberOfFilters = 256,
  numberOfResidualBlocks = 32,
  scale = 2,
  numberOfLossFunctions = 1,
  numberOfOutputChannels = 1,
  doBatchNormalization = FALSE,
  interpolation = c("bilinear", "nearest", "conv")
) {

  interpolation = match.arg(interpolation)
  residualBlock2D <- function(
    model, numberOfFilters, convolutionKernelSize,
    doBatchNormalization = FALSE ) {
    block <- model %>% layer_conv_2d( filters = numberOfFilters,
                                      kernel_size = convolutionKernelSize, activation = 'relu',
                                      padding = 'same' )
    block <- block %>% layer_conv_2d( filters = numberOfFilters,
                                      kernel_size = convolutionKernelSize, activation = 'linear',
                                      padding = 'same' )
    if ( doBatchNormalization ) block <- block %>% layer_batch_normalization()
    block <- layer_add( list( model, block ) )

    return( block )
  }

  upscaleBlock2D <- function(
    model,
    numberOfFilters,
    # nChannels,
    convolutionKernelSize, scale = 2,
    interpolation = "bilinear" )
  {
    block <- model %>% layer_upsampling_2d( size = c( scale, scale ) )
    return( block )
  }

  upscaleBlock2DConv <- function(
    model,
    numberOfFilters, scale = 2 ) {
    kernelSize = c( 4, 4 )
    strides =  c( 2, 2 )
    block <- model %>% layer_conv_2d_transpose( filters = numberOfFilters,
                                                kernel_size = kernelSize, strides = strides, activation = 'relu',
                                                kernel_initializer = 'glorot_uniform', padding = 'same' )
    if ( scale == 4 )
      block <- block %>% layer_conv_2d_transpose( filters = numberOfFilters,
                                                  kernel_size = kernelSize, strides = strides, activation = 'relu',
                                                  kernel_initializer = 'glorot_uniform', padding = 'same' )
    if ( scale == 8 )
      block <- block %>% layer_conv_2d_transpose( filters = numberOfFilters,
                                                  kernel_size = kernelSize, strides = strides, activation = 'relu',
                                                  kernel_initializer = 'glorot_uniform', padding = 'same' )
    return( block )
  }

  inputs <- layer_input( shape = inputImageSize )
  outputsX = residualBlocks = layer_conv_2d(
    inputs,
    filters = numberOfFilters,
    kernel_size = convolutionKernelSize,
    padding = 'same' )

  for( i in 1:numberOfResidualBlocks ){
    residualBlocks <- residualBlock2D(
      residualBlocks, numberOfFilters,
      convolutionKernelSize, doBatchNormalization = doBatchNormalization )
  }
  residualBlocks = layer_conv_2d( residualBlocks,
                                  filters = numberOfFilters,
                                  kernel_size = convolutionKernelSize,
                                  padding = 'same' )
  outputsX = layer_add( list( outputsX, residualBlocks ) )

  if ( interpolation != 'conv' )
    outputs <- upscaleBlock2D( model = outputsX,
                               # nChannels
                               numberOfFilters = numberOfFilters,
                               convolutionKernelSize = convolutionKernelSize,
                               scale = scale, interpolation = interpolation )
  if ( interpolation == 'conv' )
    outputs <- upscaleBlock2DConv( outputsX, numberOfFilters, scale = scale )

  outputs <- outputs %>% layer_conv_2d(
    filters = numberOfOutputChannels,
    kernel_size = c(1L,1L),
    activation = 'linear',
    padding = 'same' )

  if ( numberOfLossFunctions == 1 ) {
    srModel <- keras_model(
      inputs = inputs,
      outputs = outputs )
  } else {
    olist = list()
    for ( k in 1:numberOfLossFunctions ) olist[[ k ]] = outputs
    srModel <- keras_model(
      inputs = inputs,
      outputs = olist )
  }


  return( srModel )
}
