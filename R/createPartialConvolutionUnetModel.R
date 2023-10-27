#' 2-D implementation of the U-net architecture for inpainting using partial
#'    convolution.
#'
#'         \url{https://arxiv.org/abs/1804.07723}
#'
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfPriors Specify tissue priors for use during the decoding branch.
#' Default = 0.
#' @param numberOfFilters vector explicitly setting the number of filters at
#' each layer.  Defaults to number used in the paper.
#' @param kernelSize single scalar or tuple of same length as the number of filters.
#' Specifies the kernel size schedule for the encoding path.  Defaults to the
#' kernel sizes used in the paper.
#' @param usePartialConv boolean.  Testing.  Switch between vanilla convolution
#' layers and partial convolution layers.
#'
#' @return a u-net keras model
#' @author Tustison NJ
#'
#' @import keras
#' @export
createPartialConvolutionUnetModel2D <- function( inputImageSize, numberOfPriors = 0,
  numberOfFilters = c( 64, 128, 256, 512, 512, 512, 512, 512 ),
  kernelSize = c( 7, 5, 5, 3, 3, 3, 3, 3 ), usePartialConv = TRUE )
{
  if( length( kernelSize ) == 1 )
    {
    kernelSize <- rep( kernelSize, length( numberOfFilters ) )
    } else if( length( kernelSize ) != length( numberOfFilters ) ) {
    stop( "kernelSize must be a scalar or of equal length as the numberOfFilters." )
    }

  inputImage <- layer_input( shape = inputImageSize )
  inputMask <- layer_input( shape = inputImageSize )

  if( numberOfPriors > 0 )
    {
    inputPriors <- layer_input( shape = c( inputImageSize[1], inputImageSize[2], numberOfPriors ) )
    inputs <- list( inputImage, inputMask, inputPriors )
    } else {
    inputs <- list( inputImage, inputMask )
    }

  # Encoding path

  numberOfLayers <- length( numberOfFilters )

  encodingConvolutionLayers <- list()
  pool <- NULL
  mask <- NULL
  for( i in seq_len( numberOfLayers ) )
    {
    if( i == 1 )
      {
      if( usePartialConv )
        {
        convOutputs <- inputs %>% layer_partial_conv_2d( filters = numberOfFilters[i],
          kernelSize = kernelSize[i], padding = 'same' )
        conv <- convOutputs[[1]]
        mask <- convOutputs[[2]]
        } else {
        conv <- inputs %>% layer_conv_2d( filters = numberOfFilters[i],
          kernel_size = kernelSize[i], padding = 'same' )
        }
      } else {
        if( usePartialConv )
          {
          mask <- mask %>% layer_resample_tensor_2d( shape = c( dim( pool )[2], dim( pool )[3] ),
                                            interpolationType = 'nearestNeighbor' )
          convOutputs <- list( pool, mask ) %>% layer_partial_conv_2d( filters = numberOfFilters[i],
            kernelSize = kernelSize[i], padding = 'same' )
          conv <- convOutputs[[1]]
          mask <- convOutputs[[2]]
          } else {
          conv <- inputs %>% layer_conv_2d( filters = numberOfFilters[i],
            kernel_size = kernelSize[i], padding = 'same' )
          }
      }
    conv <- conv %>% layer_activation_relu()

    if( usePartialConv )
      {
      convOutputs <- list( conv, mask ) %>% layer_partial_conv_2d( filters = numberOfFilters[i],
        kernelSize = kernelSize[i], padding = 'same' )
      conv <- convOutputs[[1]]
      mask <- convOutputs[[2]]
      } else {
      conv <- inputs %>% layer_conv_2d( filters = numberOfFilters[i],
        kernel_size = kernelSize[i], padding = 'same' )
      }
    conv <- conv %>% layer_activation_relu()

    encodingConvolutionLayers[[i]] <- conv

    if( i < numberOfLayers )
      {
      pool <- encodingConvolutionLayers[[i]] %>%
        layer_max_pooling_2d( pool_size = c( 2, 2 ), strides = c( 2, 2 ) )
      }
    }

  # Decoding path

  outputs <- encodingConvolutionLayers[[numberOfLayers]]
  for( i in 2:numberOfLayers )
    {
    deconv <- outputs %>%
      layer_conv_2d_transpose( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = 2,
        padding = 'same' )
    deconv <- deconv %>% layer_upsampling_2d( size = c( 2, 2 ) )

    if( usePartialConv )
      {
      mask <- mask %>% layer_upsampling_2d( size = c( 2, 2 ), interpolation = "nearest" )
      }
    outputs <- layer_concatenate( list( deconv, encodingConvolutionLayers[[numberOfLayers - i + 1]] ), axis = 3, trainable = TRUE )
    if( usePartialConv )
      {
      # mask <- list( mask, outputs ) %>% layer_lambda(
      #   f = function( x ) { tensorflow::tf$repeat( tensorflow::tf$gather( x[[1]], list( 0L ), axis = -1L ),
      #     tensorflow::tf$shape( x[[2]] )[4L],
      #     axis = -1 ) } )
      mask <- list( mask, outputs ) %>% layer_lambda(
        f = function( x ) { tensorflow::tf$tile( tensorflow::tf$gather( x[[1]], list( 0L ), axis = -1L ),
          list( 1L, 1L, 1L, tensorflow::tf$shape( x[[2]] )[4L] ) ) } )
      }
    if( numberOfPriors > 0 )
      {
      resampledPriors <- inputPriors %>%
        layer_resample_tensor_2d( shape = c( dim( outputs )[2], dim( outputs )[3] ), interpolationType = "linear" )
      outputs <- list( outputs, resampledPriors ) %>% layer_concatenate( axis = 3, trainable = TRUE )
      if( usePartialConv )
        {
        resampledPriorsMask <- resampledPriors %>% layer_lambda( f = function( x ) {tensorflow::tf$ones_like( x ) } )
        mask = list( mask, resampledPriorsMask ) %>% layer_concatenate( axis = 3, trainable = TRUE )
        }
      }
    if( usePartialConv )
      {
      convOutputs <- list( outputs, mask ) %>% layer_partial_conv_2d(
        filters = numberOfFilters[numberOfLayers - i + 1], kernelSize = 3, padding = 'same' )
      outputs <- convOutputs[[1]]
      mask <- convOutputs[[2]]
      } else {
      outputs <- outputs %>% layer_conv_2d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = 3, padding = 'same' )
      }
    outputs <- outputs %>% layer_activation_relu()

    if( usePartialConv )
      {
      convOutputs <- list( outputs, mask ) %>% layer_partial_conv_2d(
        filters = numberOfFilters[numberOfLayers - i + 1], kernelSize = 3, padding = 'same' )
      outputs <- convOutputs[[1]]
      mask <- convOutputs[[2]]
      } else {
      outputs <- outputs %>% layer_conv_2d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = 3, padding = 'same' )
      }

    outputs <- outputs %>% layer_activation_relu()
    }

  outputs <- outputs %>%
    layer_conv_2d( filters = 1,
      kernel_size = c( 1, 1 ), activation = 'linear' )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}


#' 3-D implementation of the U-net architecture for inpainting using partial
#'    convolution.
#'
#'         \url{https://arxiv.org/abs/1804.07723}
#'
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori.
#' @param numberOfPriors Specify tissue priors for use during the decoding branch.
#' Default = 0.
#' @param numberOfFilters vector explicitly setting the number of filters at
#' each layer.  Defaults to number used in the paper.
#' @param kernelSize single scalar or tuple of same length as the number of filters.
#' Specifies the kernel size schedule for the encoding path.  Defaults to the
#' kernel sizes used in the paper.
#' @param usePartialConv boolean.  Testing.  Switch between vanilla convolution
#' layers and partial convolution layers.
#'
#' @return a u-net keras model
#' @author Tustison NJ
#'
#' @import keras
#' @export
createPartialConvolutionUnetModel3D <- function( inputImageSize, numberOfPriors = 0,
  numberOfFilters = c( 64, 128, 256, 512, 512, 512, 512, 512 ),
  kernelSize = c( 7, 5, 5, 3, 3, 3, 3, 3 ), usePartialConv = TRUE )
{
  if( length( kernelSize ) == 1 )
    {
    kernelSize <- rep( kernelSize, length( numberOfFilters ) )
    } else if( length( kernelSize ) != length( numberOfFilters ) ) {
    stop( "kernelSize must be a scalar or of equal length as the numberOfFilters." )
    }

  inputImage <- layer_input( shape = inputImageSize )
  inputMask <- layer_input( shape = inputImageSize )

  if( numberOfPriors > 0 )
    {
    inputPriors <- layer_input( shape = c( inputImageSize[1], inputImageSize[2], inputImageSize[3], numberOfPriors ) )
    inputs <- list( inputImage, inputMask, inputPriors )
    } else {
    inputs <- list( inputImage, inputMask )
    }

  # Encoding path

  numberOfLayers <- length( numberOfFilters )

  encodingConvolutionLayers <- list()
  pool <- NULL
  mask <- NULL
  for( i in seq_len( numberOfLayers ) )
    {
    if( i == 1 )
      {
      if( usePartialConv )
        {
        convOutputs <- inputs %>% layer_partial_conv_3d( filters = numberOfFilters[i],
          kernelSize = kernelSize[i], padding = 'same' )
        conv <- convOutputs[[1]]
        mask <- convOutputs[[2]]
        } else {
        conv <- inputs %>% layer_conv_3d( filters = numberOfFilters[i],
          kernel_size = kernelSize[i], padding = 'same' )
        }
      } else {
        if( usePartialConv )
          {
          mask <- mask %>% layer_resample_tensor_3d( shape = c( dim( pool )[2], dim( pool )[3], dim( pool )[4] ),
                                            interpolationType = 'nearestNeighbor' )
          convOutputs <- list( pool, mask ) %>% layer_partial_conv_3d( filters = numberOfFilters[i],
            kernelSize = kernelSize[i], padding = 'same' )
          conv <- convOutputs[[1]]
          mask <- convOutputs[[2]]
          } else {
          conv <- inputs %>% layer_conv_3d( filters = numberOfFilters[i],
            kernel_size = kernelSize[i], padding = 'same' )
          }
      }
    conv <- conv %>% layer_activation_relu()

    if( usePartialConv )
      {
      convOutputs <- list( conv, mask ) %>% layer_partial_conv_3d( filters = numberOfFilters[i],
        kernelSize = kernelSize[i], padding = 'same' )
      conv <- convOutputs[[1]]
      mask <- convOutputs[[2]]
      } else {
      conv <- inputs %>% layer_conv_3d( filters = numberOfFilters[i],
        kernel_size = kernelSize[i], padding = 'same' )
      }
    conv <- conv %>% layer_activation_relu()

    encodingConvolutionLayers[[i]] <- conv

    if( i < numberOfLayers )
      {
      pool <- encodingConvolutionLayers[[i]] %>%
        layer_max_pooling_3d( pool_size = c( 2, 2, 2 ), strides = c( 2, 2, 2 ) )
      }
    }

  # Decoding path

  outputs <- encodingConvolutionLayers[[numberOfLayers]]
  for( i in 2:numberOfLayers )
    {
    deconv <- outputs %>%
      layer_conv_3d_transpose( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = 2,
        padding = 'same' )
    deconv <- deconv %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )

    if( usePartialConv )
      {
      mask <- mask %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      }
    outputs <- layer_concatenate( list( deconv, encodingConvolutionLayers[[numberOfLayers - i + 1]] ), axis = 4, trainable = TRUE )
    if( usePartialConv )
      {
      # mask <- list( mask, outputs ) %>% layer_lambda(
      #   f = function( x ) { tensorflow::tf$repeat( tensorflow::tf$gather( x[[1]], list( 0L ), axis = -1L ),
      #     tensorflow::tf$shape( x[[2]] )[5L],
      #     axis = -1 ) } )
      mask <- list( mask, outputs ) %>% layer_lambda(
        f = function( x ) { tensorflow::tf$tile( tensorflow::tf$gather( x[[1]], list( 0L ), axis = -1L ),
          list( 1L, 1L, 1L, 1L, tensorflow::tf$shape( x[[2]] )[5L] ) ) } )
      }
    if( numberOfPriors > 0 )
      {
      resampledPriors <- inputPriors %>%
        layer_resample_tensor_3d( shape = c( dim( outputs )[2], dim( outputs )[3], dim( outputs )[4] ), interpolationType = "linear" )
      outputs <- list( outputs, resampledPriors ) %>% layer_concatenate( axis = 4, trainable = TRUE )
      if( usePartialConv )
        {
        resampledPriorsMask <- resampledPriors %>% layer_lambda( f = function( x ) { tensorflow::tf$ones_like( x ) } )
        mask = list( mask, resampledPriorsMask ) %>% layer_concatenate( axis = 4, trainable = TRUE )
        }
      }
    if( usePartialConv )
      {
      convOutputs <- list( outputs, mask ) %>% layer_partial_conv_3d(
        filters = numberOfFilters[numberOfLayers - i + 1], kernelSize = 3, padding = 'same' )
      outputs <- convOutputs[[1]]
      mask <- convOutputs[[2]]
      } else {
      outputs <- outputs %>% layer_conv_3d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = 3, padding = 'same' )
      }
    outputs <- outputs %>% layer_activation_relu()

    if( usePartialConv )
      {
      convOutputs <- list( outputs, mask ) %>% layer_partial_conv_3d(
        filters = numberOfFilters[numberOfLayers - i + 1], kernelSize = 3, padding = 'same' )
      outputs <- convOutputs[[1]]
      mask <- convOutputs[[2]]
      } else {
      outputs <- outputs %>% layer_conv_3d( filters = numberOfFilters[numberOfLayers - i + 1],
        kernel_size = 3, padding = 'same' )
      }

    outputs <- outputs %>% layer_activation_relu()
    }

  outputs <- outputs %>%
    layer_conv_3d( filters = 1,
      kernel_size = c( 1, 1, 1 ), activation = 'linear' )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
