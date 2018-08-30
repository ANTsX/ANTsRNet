#' 2-D implementation of the image super resolution deep learning architecture.
#'
#' Creates a keras model of the image super resolution deep learning framework.  
#' based on the paper available here:
#' 
#'         \url{https://arxiv.org/pdf/1501.00092}
#'
#' This particular implementation is based on the following python 
#' implementation: 
#' 
#'         \url{https://github.com/titu1994/Image-Super-Resolution}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param convolutionKernelSizes a list of 2-D vectors specifying the kernel
#' size at each convolution layer.  Default values are the same as given in
#' the original paper.  The length of kernel size vectors must be 1 greater
#' than the vector length of the number of filters.
#' @param numberOfFilters a vector containing the number of filters for each
#' convolutional layer.  Default values are the same as given in the original 
#' paper.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#' \dontrun{ 
#' }
#' @import keras
#' @export
createImageSuperResolutionModel2D <- function( inputImageSize, 
  convolutionKernelSizes = list( c( 9, 9 ), c( 1, 1 ), c( 5, 5 ) ),
  numberOfFilters = c( 64, 32 ) )
{
  numberOfConvolutionLayers <- length( convolutionKernelSizes )

  if( length( numberOfFilters ) != numberOfConvolutionLayers - 1 )
    {
    stop( "Error:  the length of the number of filters must be 1   
      less than the length of the convolution vector size" );
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs
  for( i in seq_len( numberOfConvolutionLayers - 1 ) )
    {
    outputs <- outputs %>% layer_conv_2d( filters = numberOfFilters[i], 
      kernel_size = convolutionKernelSizes[[i]], activation = 'relu', padding = 'same' )
    }  

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_2d( filters = numberOfChannels, 
    kernel_size = unlist( tail( convolutionKernelSizes, 1 ) ), padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}


#' 3-D implementation of the image super resolution deep learning architecture.
#'
#' Creates a keras model of the image super resolution deep learning framework.  
#' based on the paper available here:
#' 
#'         \url{https://arxiv.org/pdf/1501.00092}
#'
#' This particular implementation is based on the following python 
#' implementation: 
#' 
#'         \url{https://github.com/titu1994/Image-Super-Resolution}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param convolutionKernelSizes a list of 3-D vectors specifying the kernel
#' size at each convolution layer.  Default values are the same as given in
#' the original paper.  The length of kernel size vectors must be 1 greater
#' than the vector length of the number of filters.
#' @param numberOfFilters a vector containing the number of filters for each
#' convolutional layer.  Default values are the same as given in the original 
#' paper.
#'
#' @return a keras model for image super resolution
#' @author Tustison NJ
#' @examples
#' \dontrun{ 
#' }
#' @import keras
#' @export
createImageSuperResolutionModel3D <- function( inputImageSize, 
  convolutionKernelSizes = list( c( 9, 9, 9 ), c( 1, 1, 1 ), c( 5, 5, 5 ) ),
  numberOfFilters = c( 64, 32 ) )
{
  numberOfConvolutionLayers <- length( convolutionKernelSizes )

  if( length( numberOfFilters ) != numberOfConvolutionLayers - 1 )
    {
    stop( "Error:  the length of the number of filters must be 1   
      less than the length of the convolution vector size." );
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs
  for( i in seq_len( numberOfConvolutionLayers - 1 ) )
    {
    outputs <- outputs %>% layer_conv_3d( filters = numberOfFilters[i], 
      kernel_size = convolutionKernelSizes[[i]], activation = 'relu', padding = 'same' )
    }  

  numberOfChannels <- tail( inputImageSize, 1 )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfChannels, 
    kernel_size = unlist( tail( convolutionKernelSizes, 1 ) ), padding = 'same' )

  srModel <- keras_model( inputs = inputs, outputs = outputs )

  return( srModel )
}

