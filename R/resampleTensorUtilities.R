#' Resamples a spatial tensor.
#'
#' Resamples a spatial tensor based on the specified shape and interpolation type.
#'
#' @param inputTensor tensor to be resampled.
#' @param shape vector or list of length 2 or 3 specifying the shape of the
#' output tensor
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#'
#' @return a tensor
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#' K <- keras::backend()
#'
#' # 2-D spatial tensor
#'
#' inputTensor <- K$ones( c( 2L, 10L, 10L, 3L ) )
#'
#' outputTensor <- resampleTensor( inputTensor, c( 12, 13 ), 'nearestNeighbor' )
#' outputTensor <- resampleTensor( inputTensor, c( 12, 13 ), 'linear' )
#' outputTensor <- resampleTensor( inputTensor, c( 12, 13 ), 'cubic' )
#'
#' # 3-D spatial tensor
#'
#' inputTensor <- K$ones( c( 2L, 10L, 10L, 10L, 3L ) )
#'
#' outputTensor <- resampleTensor( inputTensor, c( 12, 13, 14 ), 'nearestNeighbor' )
#' outputTensor <- resampleTensor( inputTensor, c( 12, 13, 14 ), 'linear' )
#' outputTensor <- resampleTensor( inputTensor, c( 12, 13, 14 ), 'cubic' )
#'
#' @import keras
#' @export
resampleTensor <- function( inputTensor, shape, interpolationType = 'nearestNeighbor' )
  {
  K <- keras::backend()

  newSize <- as.integer( shape )
  inputShape <- K$int_shape( inputTensor )
  inputShape[sapply( inputShape, is.null )] <- NA
  inputShape <- unlist( inputShape )

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

  resampledTensor <- NULL
  if( dimensionality == 2 )
    {
    if( interpolationType == 'nearestNeighbor' )
      {
      resampledTensor <- tensorflow::tf$image$resize_nearest_neighbor( inputTensor, size = newSize, align_corners = TRUE )
      } else if( interpolationType == 'linear' ) {
      resampledTensor <- tensorflow::tf$image$resize_bilinear( inputTensor, size = newSize, align_corners = TRUE )
      } else if( interpolationType == 'cubic' ) {
      resampledTensor <- tensorflow::tf$image$resize_bicubic( inputTensor, size = newSize, align_corners = TRUE )
      } else {
      stop( "Interpolation type not recognized." )
      }
    } else {
    # Do yz
    squeezeTensor_yz <-
      tensorflow::tf$reshape( inputTensor, c( -1L, oldSize[2], oldSize[3], channelSize ) )

    newShape_yz <- c( newSize[2], newSize[3] )

    resampledTensor_yz <- NULL
    if( interpolationType == 'nearestNeighbor' )
      {
      resampledTensor_yz <-
        tensorflow::tf$image$resize_nearest_neighbor( squeezeTensor_yz, size = newShape_yz, align_corners = TRUE )
      } else if( interpolationType == 'linear' ) {
      resampledTensor_yz <-
        tensorflow::tf$image$resize_bilinear( squeezeTensor_yz, size = newShape_yz, align_corners = TRUE )
      } else if( interpolationType == 'cubic' ) {
      resampledTensor_yz <-
        tensorflow::tf$image$resize_bicubic( squeezeTensor_yz, size = newShape_yz, align_corners = TRUE )
      } else {
      stop( "Interpolation type not recognized." )
      }

    newShape_yz <- c( batchSize, oldSize[1], newSize[2], newSize[3], channelSize )
    resumeTensor_yz <- tensorflow::tf$reshape( resampledTensor_yz, newShape_yz )

    # Do x

    reorientedTensor <- tensorflow::tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

    squeezeTensor_x <- tensorflow::tf$reshape( reorientedTensor,
      c( -1L, newSize[2], oldSize[1], channelSize ) )

    newShape_x <- c( newSize[2], newSize[1] )

    resampledTensor_x <- NULL
    if( interpolationType == 'nearestNeighbor' )
      {
      resampledTensor_x <-
        tensorflow::tf$image$resize_nearest_neighbor( squeezeTensor_x, size = newShape_x, align_corners = TRUE )
      } else if( interpolationType == 'linear' ) {
      resampledTensor_x <-
        tensorflow::tf$image$resize_bilinear( squeezeTensor_x, size = newShape_x, align_corners = TRUE )
      } else if( interpolationType == 'cubic' ) {
      resampledTensor_x <-
        tensorflow::tf$image$resize_bicubic( squeezeTensor_x, size = newShape_x, align_corners = TRUE )
      } else {
      stop( "Interpolation type not recognized." )
      }

    newShape_x <- c( batchSize, newSize[3], newSize[2], newSize[1], channelSize )
    resumeTensor_x <- tensorflow::tf$reshape( resampledTensor_x, newShape_x )

    resampledTensor <- tensorflow::tf$transpose( resumeTensor_x, c( 0L, 3L, 2L, 1L, 4L ) )
    }

  return( resampledTensor )
  }

#' Resamples a tensor.
#'
#' Resamples a tensor based on the reference tensor and interpolation type.
#'
#' @param inputTensor tensor to be resampled.
#' @param referenceTensor Reference tensor of rank 4 or 5 (for 2-D or 3-D volumes,
#'                        respectively).
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#'
#' @return a tensor
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#' K <- keras::backend()
#'
#' inputTensor <- K$ones( c( 2L, 10L, 10L, 10L, 3L ) )
#' referenceTensor <- K$ones( c( 2L, 12L, 13L, 14L, 3L ) )
#'
#' outputTensor <- resampleTensorLike( inputTensor, referenceTensor )
#'
#' @import keras
#' @export
resampleTensorLike <- function( inputTensor, referenceTensor, interpolationType = 'nearestNeighbor' )
  {
  K <- keras::backend()

  referenceShape <- K$int_shape( referenceTensor )
  referenceShape[sapply( referenceShape, is.null )] <- NA
  referenceShape <- unlist( referenceShape )

  if( length( referenceShape ) == 4 )
    {
    referenceShape <- referenceShape[2:3]
    } else if( length( referenceShape ) == 5 ) {
    referenceShape <- referenceShape[2:4]
    } else {
    stop( "Reference tensor must be of rank 4 or 5 (for 2-D images or 3-D volumes)." )
    }

  resampledTensor <- resampleTensor( inputTensor, referenceShape, interpolationType )

  return( resampledTensor )
  }

#' Creates a resample tensor lambda layer (2-D)
#'
#' Creates a lambda layer which interpolates/resizes an input tensor based
#' on the specified shape
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{resampledTensor <- inputs %>% layer_resample_tensor_2d( shape,
#'      interpolationType = 'nearestNeighbor' )
#'
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{shape}{A 2-D vector specifying the new shape.}
#'  \item{interpolationType}{Type of interpolation.  Can be
#'    \code{'nearestNeighbor'}, \code{'linear'}, or \code{'cubic'}}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return a resampled version of the input tensor
#'
#' @name ResampleTensorLayer2D
NULL

#' @export
ResampleTensorLayer2D <- R6::R6Class( "ResampleTensorLayer2D",

  inherit = KerasLayer,

  public = list(

    shape = NULL,

    interpolationType = 'nearestNeighbor',

    initialize = function( shape, interpolationType = 'nearestNeighbor' )
      {
      if( length( shape ) != 2 )
        {
        stop( "shape must be of length 2 specifying the width and height
               of the resampled tensor." )
        }
      self$shape <- shape

      allowedTypes <- c( 'nearestNeighbor', 'linear', 'cubic' )
      if( ! interpolationType %in% allowedTypes )
        {
        stop( "interpolationType not one of the allowed types." )
        }
      self$interpolationType <- interpolationType
      },

    compute_output_shape = function( input_shape )
      {
      if( length( input_shape ) != 4 )
        {
        stop( "Input tensor must be of rank 4." )
        }
      return( reticulate::tuple( input_shape[[1]], self$shape[1],
          self$shape[2], input_shape[[4]] ) )
      },

   call = function( x, mask = NULL )
      {
      K <- keras::backend()

      dimensionality <- 2

      newSize <- as.integer( self$shape )
      inputShape <- K$int_shape( x )
      inputShape[sapply( inputShape, is.null )] <- NA
      inputShape <- unlist( inputShape )
      oldSize <- inputShape[2:( dimensionality + 1 )]

      if( all( newSize == oldSize ) )
        {
        return( x + 0 )
        }

      resampledTensor <- NULL
      if( self$interpolationType == 'nearestNeighbor' )
        {
        resampledTensor <- tensorflow::tf$image$resize_nearest_neighbor( x, size = newSize, align_corners = TRUE )
        } else if( self$interpolationType == 'linear' ) {
        resampledTensor <- tensorflow::tf$image$resize_bilinear( x, size = newSize, align_corners = TRUE )
        } else if( self$interpolationType == 'cubic' ) {
        resampledTensor <- tensorflow::tf$image$resize_bicubic( x, size = newSize, align_corners = TRUE )
        }
      return( resampledTensor )
      }
  )
)

#' Resampling a spatial tensor (2-D).
#'
#' Resamples a spatial tensor based on the specified shape and interpolation type.
#'
#' @param inputTensor tensor to be resampled.
#' @param shape vector or list of length 2 specifying the shape of the output tensor.
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#'
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
layer_resample_tensor_2d <- function( object, shape,
    interpolationType = 'nearestNeighbor', name = NULL,
    trainable = FALSE ) {
create_layer( ResampleTensorLayer2D, object,
    list( shape = shape, interpolationType = interpolationType,
        name = name, trainable = trainable ) )
}

#' Creates a resample tensor lambda layer (3-D)
#'
#' Creates a lambda layer which interpolates/resizes an input tensor based
#' on the specified shape
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{resampledTensor <- inputs %>% layer_resample_tensor_3d( shape,
#'      interpolationType = 'nearestNeighbor' )
#'
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{shape}{A 3-D vector specifying the new shape.}
#'  \item{interpolationType}{Type of interpolation.  Can be
#'    \code{'nearestNeighbor'}, \code{'linear'}, or \code{'cubic'}}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return a resampled version of the input tensor
#'
#' @name ResampleTensorLayer3D
NULL

#' @export
ResampleTensorLayer3D <- R6::R6Class( "ResampleTensorLayer3D",

  inherit = KerasLayer,

  public = list(

    shape = NULL,

    interpolationType = 'nearestNeighbor',

    initialize = function( shape, interpolationType = 'nearestNeighbor' )
      {
      if( length( shape ) != 3 )
        {
        stop( "shape must be of length 3 specifying the width, height and
               depth of the resampled tensor." )
        }
      self$shape <- shape

      allowedTypes <- c( 'nearestNeighbor', 'linear', 'cubic' )
      if( ! interpolationType %in% allowedTypes )
        {
        stop( "interpolationType not one of the allowed types." )
        }
      self$interpolationType <- interpolationType
      },

    compute_output_shape = function( input_shape )
      {
      if( length( input_shape ) != 5 )
        {
        stop( "Input tensor must be of rank 5." )
        }
      return( reticulate::tuple( input_shape[[1]], self$shape[1],
          self$shape[2], self$shape[3], input_shape[[5]] ) )
      },

   call = function( x, mask = NULL )
      {
      K <- keras::backend()

      dimensionality <- 3

      newSize <- as.integer( self$shape )
      inputShape <- K$int_shape( x )
      inputShape[sapply( inputShape, is.null )] <- NA
      inputShape <- unlist( inputShape )
      oldSize <- inputShape[2:( dimensionality + 1 )]

      batchSize <- inputShape[1]
      channelSize <- tail( inputShape, 1 )

      if( all( newSize == oldSize ) )
        {
        return( x + 0 )
        }

      resampledTensor <- NULL
      # Do yz
      squeezeTensor_yz <-
        tensorflow::tf$reshape( x, c( -1L, oldSize[2], oldSize[3], channelSize ) )

      newShape_yz <- c( newSize[2], newSize[3] )

      resampledTensor_yz <- NULL
      if( self$interpolationType == 'nearestNeighbor' )
        {
        resampledTensor_yz <-
          tensorflow::tf$image$resize_nearest_neighbor( squeezeTensor_yz, size = newShape_yz, align_corners = TRUE )
        } else if( self$interpolationType == 'linear' ) {
        resampledTensor_yz <-
          tensorflow::tf$image$resize_bilinear( squeezeTensor_yz, size = newShape_yz, align_corners = TRUE )
        } else if( self$interpolationType == 'cubic' ) {
        resampledTensor_yz <-
          tensorflow::tf$image$resize_bicubic( squeezeTensor_yz, size = newShape_yz, align_corners = TRUE )
        } else {
        stop( "Interpolation type not recognized." )
        }

      newShape_yz <- c( batchSize, oldSize[1], newSize[2], newSize[3], channelSize )
      resumeTensor_yz <- tensorflow::tf$reshape( resampledTensor_yz, newShape_yz )

      # Do x

      reorientedTensor <- tensorflow::tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

      squeezeTensor_x <- tensorflow::tf$reshape( reorientedTensor,
        c( -1L, newSize[2], oldSize[1], channelSize ) )

      newShape_x <- c( newSize[2], newSize[1] )

      resampledTensor_x <- NULL
      if( self$interpolationType == 'nearestNeighbor' )
        {
        resampledTensor_x <-
          tensorflow::tf$image$resize_nearest_neighbor( squeezeTensor_x, size = newShape_x, align_corners = TRUE )
        } else if( self$interpolationType == 'linear' ) {
        resampledTensor_x <-
          tensorflow::tf$image$resize_bilinear( squeezeTensor_x, size = newShape_x, align_corners = TRUE )
        } else if( self$interpolationType == 'cubic' ) {
        resampledTensor_x <-
          tensorflow::tf$image$resize_bicubic( squeezeTensor_x, size = newShape_x, align_corners = TRUE )
        } else {
        stop( "Interpolation type not recognized." )
        }

      newShape_x <- c( batchSize, newSize[3], newSize[2], newSize[1], channelSize )
      resumeTensor_x <- tensorflow::tf$reshape( resampledTensor_x, newShape_x )

      resampledTensor <- tensorflow::tf$transpose( resumeTensor_x, c( 0L, 3L, 2L, 1L, 4L ) )

      return( resampledTensor )
      }
  )
)

#' Resampling a spatial tensor (3-D).
#'
#' Resamples a spatial tensor based on the specified shape and interpolation type.
#'
#' @param inputTensor tensor to be resampled.
#' @param shape vector or list of length 3 specifying the shape of the output tensor.
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#'
#' @return a keras layer tensor
#' @export
layer_resample_tensor_3d <- function( object, shape,
    interpolationType = 'nearestNeighbor', name = NULL,
    trainable = FALSE ) {
create_layer( ResampleTensorLayer3D, object,
    list( shape = shape, interpolationType = interpolationType,
        name = name, trainable = trainable ) )
}

