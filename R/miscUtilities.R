#' Creates an interpolating (or resizing) lambda layer
#'
#' Creates a lambda layer which interpolates/resizes an input tensor based
#' on the specified shape
#'
#' @param shape vector or list of length 2 or 3 specifying the shape of the
#' output tensor
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#'
#' @return a keras lambda layer
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#' K <- keras::backend()
#'
#' inputTensor <- K$ones( c( 2L, 10L, 10L, 10L, 3L ) )
#'
#' interpolatorLayer <- ResizeTensorLayer( c( 15L, 10L, 12L ) )
#' outputTensor <- inputTensor %>% interpolatorLayer
#'
#' @import keras
#' @export
ResizeTensorLayer <- function( shape, interpolationType = 'nearestNeighbor' )
  {
  f <- function( inputTensor )
    {
    K <- keras::backend()

    newSize <- as.integer( shape )
    inputShape <- unlist( K$int_shape( inputTensor ) )

    batchSize <- inputShape[1]
    channelSize <- tail( inputShape, 1 )

    dimensionality <- NULL
    if( length( shape ) == 2 )
      {
      dimensionality <- 2
      } else if( length( shape == 3 ) ) {
      dimensionality <- 3
      } else {
      stop( "\'shape\' should be of length 2 for images or 3 for volumes." )
      }

    oldSize <- inputShape[2:( dimensionality + 1 )]

    if( all( newSize == oldSize ) )
      {
      return( inputTensor )
      }

    resizedTensor <- NULL
    if( dimensionality == 2 )
      {
      if( interpolationType == 'nearestNeighbor' )
        {
        resizedTensor <- tensorflow::tf$image$resize_nearest_neighbor( inputTensor, size = shape )
        } else if( interpolationType == 'linear' ) {
        resizedTensor <- tensorflow::tf$image$resize_bilinear( inputTensor, size = shape )
        } else if( interpolationType == 'cubic' ) {
        resizedTensor <- tensorflow::tf$image$resize_bicubic( inputTensor, size = shape )
        } else {
        stop( "Interpolation type not recognized." )
        }
      } else {
      # Do yz
      squeezeTensor_yz <-
        tensorflow::tf$reshape( inputTensor, c( -1L, oldSize[2], oldSize[3], channelSize ) )

      newShape_yz <- c( newSize[2], newSize[3] )

      resizedTensor_yz <- NULL
      if( interpolationType == 'nearestNeighbor' )
        {
        resizedTensor_yz <-
          tensorflow::tf$image$resize_nearest_neighbor( squeezeTensor_yz, size = newShape_yz )
        } else if( interpolationType == 'linear' ) {
        resizedTensor_yz <-
          tensorflow::tf$image$resize_bilinear( squeezeTensor_yz, size = newShape_yz )
        } else if( interpolationType == 'cubic' ) {
        resizedTensor_yz <-
          tensorflow::tf$image$resize_bicubic( squeezeTensor_yz, size = newShape_yz )
        } else {
        stop( "Interpolation type not recognized." )
        }

      newShape_yz <- c( batchSize, oldSize[1], newSize[2], newSize[3], channelSize )
      resumeTensor_yz <- tensorflow::tf$reshape( resizedTensor_yz, newShape_yz )

      # Do x

      reorientedTensor <- tensorflow::tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

      squeezeTensor_x <- tensorflow::tf$reshape( reorientedTensor,
        c( -1L, newSize[2], oldSize[1], channelSize ) )

      newShape_x <- c( newSize[2], newSize[1] )

      resizedTensor_x <- NULL
      if( interpolationType == 'nearestNeighbor' )
        {
        resizedTensor_x <-
          tensorflow::tf$image$resize_nearest_neighbor( squeezeTensor_x, size = newShape_x )
        } else if( interpolationType == 'linear' ) {
        resizedTensor_x <-
          tensorflow::tf$image$resize_bilinear( squeezeTensor_x, size = newShape_x )
        } else if( interpolationType == 'cubic' ) {
        resizedTensor_x <-
          tensorflow::tf$image$resize_bicubic( squeezeTensor_x, size = newShape_x )
        } else {
        stop( "Interpolation type not recognized." )
        }

      newShape_x <- c( batchSize, newSize[3], newSize[2], newSize[1], channelSize )
      resumeTensor_x <- tensorflow::tf$reshape( resizedTensor_x, newShape_x )

      resizedTensor <- tensorflow::tf$transpose( resumeTensor_x, c( 0L, 3L, 2L, 1L, 4L ) )
      }

    return( resizedTensor )
    }

  return( layer_lambda( f = f ) )
  }

