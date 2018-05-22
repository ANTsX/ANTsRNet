#' 2-D implementation of the expanded image super resolution architecture.
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
#' @param convolutionKernelSizes a 2-element list of 2-D vectors specifying the 
#' kernel size at each convolution layer.  The first element is the kernel size
#' of the encoding layers and the 2nd element is the kernel size of the final 
#' convolution layer.
#' @param numberOfEncodingLayers the number of encoding layers.
#' @param numberOfFilters the number of filters for each encoding layer.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#' \dontrun{ 
#' }
#' @import keras
#' @export
createDenoisingAutoEncoderSuperResolutionModel2D <- function( inputImageSize, 
  convolutionKernelSizes = list( c( 3, 3 ), c( 5, 5 ) ), 
  numberOfEncodingLayers = 2,
  numberOfFilters = 64 )
{
  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs
 
  encodingConvolutionLayers <- list()
  for( i in 1:numberOfEncodingLayers )
    {
    if( i == 1 )
      {
      encodingConvolutionLayers[[i]] <- inputs %>% 
        layer_conv_2d( filters = numberOfFilters, activation = 'relu', 
          kernel_size = convolutionKernelSizes[[1]], padding = 'same' )
      } else {
      encodingConvolutionLayers[[i]] <- encodingConvolutionLayers[[i-1]] %>% 
        layer_conv_2d( filters = numberOfFilters, activation = 'relu', 
          kernel_size = convolutionKernelSizes[[1]], padding = 'same' )
      }
    }  

  outputs <- encodingConvolutionLayers[[length( encodingConvolutionLayers )]]
  for( i in 1:numberOfEncodingLayers )
    {
    index <- length( encodingConvolutionLayers ) - i + 1
          
    deconvolution <- outputs %>%
      layer_conv_2d_transpose( filters = numberOfFilters,
        kernel_size = convolutionKernelSizes[[1]],
        padding = 'same' )
    outputs <- layer_add( 
      list( encodingConvolutionLayers[[index]], deconvolution ) )    
    }  

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_2d( filters = numberOfChannels, 
    kernel_size = convolutionKernelSizes[[2]], activation = "linear", 
    padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}

#' 3-D implementation of the expanded image super resolution architecture.
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
#' @param convolutionKernelSizes a 2-element list of 3-D vectors specifying the 
#' kernel size at each convolution layer.  The first element is the kernel size
#' of the encoding layers and the 2nd element is the kernel size of the final 
#' convolution layer.
#' @param numberOfEncodingLayers the number of encoding layers.
#' @param numberOfFilters the number of filters for each encoding layer.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#' \dontrun{ 
#' }
#' @import keras
#' @export
createDenoisingAutoEncoderSuperResolutionModel3D <- function( inputImageSize, 
  convolutionKernelSizes = list( c( 3, 3, 3 ), c( 5, 5, 5 ) ), 
  numberOfEncodingLayers = 2,
  numberOfFilters = 64 )
{
  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs
 
  encodingConvolutionLayers <- list()
  for( i in 1:numberOfEncodingLayers )
    {
    if( i == 1 )
      {
      encodingConvolutionLayers[[i]] <- inputs %>% 
        layer_conv_3d( filters = numberOfFilters, activation = 'relu', 
          kernel_size = convolutionKernelSizes[[1]], padding = 'same' )
      } else {
      encodingConvolutionLayers[[i]] <- encodingConvolutionLayers[[i-1]] %>% 
        layer_conv_3d( filters = numberOfFilters, activation = 'relu', 
          kernel_size = convolutionKernelSizes[[1]], padding = 'same' )
      }
    }  

  outputs <- encodingConvolutionLayers[[length( encodingConvolutionLayers )]]
  for( i in 1:numberOfEncodingLayers )
    {
    index <- length( encodingConvolutionLayers ) - i + 1
          
    deconvolution <- outputs %>%
      layer_conv_3d_transpose( filters = numberOfFilters,
        kernel_size = convolutionKernelSizes[[1]],
        padding = 'same' )
    outputs <- layer_add( 
      list( encodingConvolutionLayers[[index]], deconvolution ) )    
    }  

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfChannels, 
    kernel_size = convolutionKernelSizes[[2]], activation = "linear", 
    padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}

