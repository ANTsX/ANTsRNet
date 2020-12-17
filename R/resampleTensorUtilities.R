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
#' rm(K)
#' rm(outputTensor)
#' rm(intputTensor)
#' gc()
#' @import keras
#' @export
resampleTensor <- function(
  inputTensor, shape,
  interpolationType =
    c("nearestNeighbor", "linear",
      "cubic",
      "bicubic",
      "bilinear",
      "nearest"))
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
  } else if ( length( shape ) == 3  ) {
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
  interpolationType = match.arg(interpolationType)
  func = tfResizingFunction(interpolationType = interpolationType)

  if ( dimensionality == 2 ){
    resampledTensor <- func(
      inputTensor,
      size = newSize)
  } else {
    # Do yz
    squeezeTensor_yz <-
      tensorflow::tf$reshape( inputTensor, c( -1L, oldSize[2], oldSize[3], channelSize ) )

    newShape_yz <- c( newSize[2], newSize[3] )

    resampledTensor_yz <- NULL

    resampledTensor_yz <- func(
      squeezeTensor_yz,
      size = newShape_yz)
    newShape_yz <- c( batchSize, oldSize[1], newSize[2], newSize[3], channelSize )
    resumeTensor_yz <- tensorflow::tf$reshape( resampledTensor_yz, newShape_yz )

    # Do x

    reorientedTensor <- tensorflow::tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

    squeezeTensor_x <- tensorflow::tf$reshape( reorientedTensor,
                                               c( -1L, newSize[2], oldSize[1], channelSize ) )

    newShape_x <- c( newSize[2], newSize[1] )

    resampledTensor_x <- func(squeezeTensor_x, size = newShape_x)
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
ResampleTensorLayer2D <- R6::R6Class(
  "ResampleTensorLayer2D",

  inherit = KerasLayer,

  public = list(

    shape = NULL,

    interpolationType = 'nearestNeighbor',

    initialize = function( shape, interpolationType =
                             c( "nearestNeighbor",
                                "nearest",
                                "linear",
                                "cubic",
                                "bicubic",
                                "bilinear"
                                ) )
      {
      if( length( shape ) != 2 )
        {
        stop( "shape must be of length 2 specifying the width and height
               of the resampled tensor." )
        }

      self$shape <- shape

      interpolationType = match.arg( interpolationType )
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
      resampledTensor <- NULL
      func <- tfResizingFunction( self$interpolationType )
      resampledTensor <- func( x, size = self$shape )
      return( resampledTensor )
      }
  )
)

#' Resampling a spatial tensor (2-D).
#'
#' Resamples a spatial tensor based on the specified shape and interpolation type.
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param shape vector or list of length 2 specifying the shape of the output tensor.
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#' @param name The name of the layer
#' @param trainable Whether the layer weights will be updated during training.
#'
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
#' @rdname layer_resample_tensor_2d
layer_resample_tensor_2d <- function(
  object, shape,
  interpolationType = c( "nearestNeighbor",
                         "nearest",
                         "linear",
                         "cubic",
                         "bicubic",
                         "bilinear"
                        ),
  name = NULL,
  trainable = FALSE ) {
  interpolationType = match.arg( interpolationType )
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
ResampleTensorLayer3D <- R6::R6Class(
  "ResampleTensorLayer3D",

  inherit = KerasLayer,

  public = list(

    shape = NULL,

    interpolationType = 'nearestNeighbor',

    initialize = function( shape, interpolationType =
                             c( "nearestNeighbor",
                                "nearest",
                                "linear",
                                "cubic",
                                "bicubic",
                                "bilinear"
                                ) )
      {

      if( length( shape ) != 3 )
        {
        stop( "shape must be of length 3 specifying the width, height and
               depth of the resampled tensor." )
        }
      self$shape <- shape

      interpolationType = match.arg( interpolationType )
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
      newSize <- self$shape
      channelSize <- x$get_shape()[[5]]

      resampledTensor <- NULL

      # Do yz

      # newShape_squeeze_yz <- list( -1L, oldSize[2], oldSize[3], channelSize )
      newShape_squeeze_yz <- reticulate::tuple( -1L, tensorflow::tf$shape( x )[3], tensorflow::tf$shape( x )[4], channelSize )
      squeezeTensor_yz <- tf$reshape( x, newShape_squeeze_yz )

      resampledTensor_yz <- NULL
      newShape_yz <- list( newSize[2], newSize[3] )
      func <- tfResizingFunction( self$interpolationType )
      resampledTensor_yz <- func( squeezeTensor_yz, size = newShape_yz )

      # newShape_yz <- list( -1L, oldSize[1], newSize[2], newSize[3], channelSize )
      newShape_yz <- reticulate::tuple( -1L, tensorflow::tf$shape( x )[2], newSize[2], newSize[3], channelSize )
      resumeTensor_yz <- tensorflow::tf$reshape( resampledTensor_yz, newShape_yz )

      # Do x

      reorientedTensor <- tensorflow::tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

      # newShape_squeeze_x <- list( -1L, newSize[2], oldSize[1], channelSize )
      newShape_squeeze_x <- reticulate::tuple( -1L, newSize[2], tensorflow::tf$shape( x )[2], channelSize ) #######  ********** The problem  *******
      squeezeTensor_x <- tensorflow::tf$reshape( reorientedTensor, newShape_squeeze_x )

      resampledTensor_x <- NULL
      newShape_x <- list( newSize[2], newSize[1] )
      func = tfResizingFunction( self$interpolationType )
      resampledTensor_x <- func( squeezeTensor_x, size = newShape_x )

      newShape_x <- list( -1L, newSize[3], newSize[2], newSize[1], channelSize )
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
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param shape vector or list of length 3 specifying the shape of the output tensor.
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{linear}, or \code{cubic}.
#' @param name The name of the layer
#' @param trainable Whether the layer weights will be updated during training.
#'
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
#' @rdname layer_resample_tensor_3d
layer_resample_tensor_3d <- function(
  object, shape,
  interpolationType = c( "nearestNeighbor",
                         "nearest",
                         "linear",
                         "cubic",
                         "bicubic",
                         "bilinear"
                        ),
  name = NULL,
  trainable = FALSE ) {
  interpolationType = match.arg( interpolationType )
  create_layer( ResampleTensorLayer3D, object,
                list( shape = shape, interpolationType = interpolationType,
                      name = name, trainable = trainable ) )
}


tfResizingFunction = function( interpolationType )
{
  tf_img <- tensorflow::tf$image
  n_image <- names( tf_img )
  if( interpolationType == "linear" )
    {
    interpolationType = "bilinear"
    }
  if( interpolationType == "nearestNeighbor" )
    {
    interpolationType = "nearest"
    }
  if( interpolationType == "cubic" )
    {
    interpolationType = "bicubic"
    }
  if ( "resize" %in% n_image )
    {
    func = function(...)
      {
      tf_img$resize( ..., method = interpolationType )
      }
    } else {
    runFunc <- switch(
      interpolationType,
      nearest = tf_img$resize_nearest_neighbor,
      bilinear = tf_img$resize_bilinear,
      bicubic = tf_img$resize_bicubic
    )
    func = function( ... )
      {
      runFunc( ..., align_corners = TRUE )
      }
    }
  if( is.null( func ) )
    {
    stop( "Function not found in tf, you may need updated tensorflow" )
    }
  return( func )
}
