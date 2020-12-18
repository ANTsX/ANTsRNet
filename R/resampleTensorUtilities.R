#################################################################
#
#  Resampling layers (to a fixed size)
#
#################################################################


#' Creates a resample tensor layer (2-D)
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
#'    \code{'nearestNeighbor'}, \code{'nearestNeighbor'},
#'    \code{'linear'}, \code{'bilinear'},
#'    \code{'cubic'}, or \code{'bicubic'}}
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
                                "bilinear",
                                "cubic",
                                "bicubic"
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

#' Creates a resampled tensor (to fixed size) layer (2-D)
#'
#' Resamples a spatial tensor based on the specified shape and interpolation type.
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param shape vector or list of length 2 specifying the shape of the output tensor.
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{nearest},
#' \code{linear}, \code{bilinear},
#' \code{cubic}, or \code{bicubic}.
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
                         "bilinear",
                         "cubic",
                         "bicubic"
                       ),
  name = NULL,
  trainable = FALSE ) {
  interpolationType = match.arg( interpolationType )
  create_layer( ResampleTensorLayer2D, object,
                list( shape = shape, interpolationType = interpolationType,
                      name = name, trainable = trainable ) )
}


#' Creates a resampled tensor (to fixed size) layer (3-D)
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
#'    \code{'nearestNeighbor'}, \code{'nearestNeighbor'},
#'    \code{'linear'}, \code{'bilinear'},
#'    \code{'cubic'}, or \code{'bicubic'}}
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
                                "bilinear",
                                "cubic",
                                "bicubic"
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
      tf <- tensorflow::tf

      newSize <- self$shape
      channelSize <- x$get_shape()[[5]]

      resampledTensor <- NULL

      # Do yz

      # newShape_squeeze_yz <- list( -1L, oldSize[2], oldSize[3], channelSize )
      newShape_squeeze_yz <- reticulate::tuple( -1L, tf$shape( x )[3], tf$shape( x )[4], channelSize )
      squeezeTensor_yz <- tf$reshape( x, newShape_squeeze_yz )

      resampledTensor_yz <- NULL
      newShape_yz <- list( newSize[2], newSize[3] )
      func <- tfResizingFunction( self$interpolationType )
      resampledTensor_yz <- func( squeezeTensor_yz, size = newShape_yz )

      # newShape_yz <- list( -1L, oldSize[1], newSize[2], newSize[3], channelSize )
      newShape_yz <- reticulate::tuple( -1L, tf$shape( x )[2], newSize[2], newSize[3], channelSize )
      resumeTensor_yz <- tf$reshape( resampledTensor_yz, newShape_yz )

      # Do x

      reorientedTensor <- tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

      # newShape_squeeze_x <- list( -1L, newSize[2], oldSize[1], channelSize )
      newShape_squeeze_x <- reticulate::tuple( -1L, newSize[2], tf$shape( x )[2], channelSize )
      squeezeTensor_x <- tf$reshape( reorientedTensor, newShape_squeeze_x )

      resampledTensor_x <- NULL
      newShape_x <- list( newSize[2], newSize[1] )
      func <- tfResizingFunction( self$interpolationType )
      resampledTensor_x <- func( squeezeTensor_x, size = newShape_x )

      newShape_x <- list( -1L, newSize[3], newSize[2], newSize[1], channelSize )
      resumeTensor_x <- tf$reshape( resampledTensor_x, newShape_x )

      resampledTensor <- tf$transpose( resumeTensor_x, c( 0L, 3L, 2L, 1L, 4L ) )

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
#' \code{nearestNeighbor}, \code{nearest},
#' \code{linear}, \code{bilinear},
#' \code{cubic}, or \code{bicubic}.
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
                         "bilinear",
                         "cubic",
                         "bicubic"
                        ),
  name = NULL,
  trainable = FALSE ) {
  interpolationType = match.arg( interpolationType )
  create_layer( ResampleTensorLayer3D, object,
                list( shape = shape, interpolationType = interpolationType,
                      name = name, trainable = trainable ) )
}

#################################################################
#
#  Resampling layers (to a target tensor)
#
#################################################################

#' Creates a resampled tensor (to target tensor) layer (2-D)
#'
#' Creates a lambda layer which interpolates/resizes an input tensor based
#' on the specified target tensor
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{targetTensor}{tensor of desired size.}
#'  \item{interpolationType}{Type of interpolation.  Can be
#'    \code{'nearestNeighbor'}, \code{'nearestNeighbor'},
#'    \code{'linear'}, \code{'bilinear'},
#'    \code{'cubic'}, or \code{'bicubic'}}
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
#' @name ResampleTensorToTargetTensorLayer2D
NULL

#' @export
ResampleTensorToTargetTensorLayer2D <- R6::R6Class(
  "ResampleTensorToTargetTensorLayer2D",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    interpolationType = 'nearestNeighbor',

    initialize = function( interpolationType =
                             c( "nearestNeighbor",
                                "nearest",
                                "linear",
                                "bilinear",
                                "cubic",
                                "bicubic"
                                ) )
      {
      self$resampledTensor <- NULL

      interpolationType = match.arg( interpolationType )
      self$interpolationType <- interpolationType
      },

    call = function( x, mask = NULL )
      {
      tf <- tensorflow::tf
      K <- tf$keras$backend

      sourceTensor <- x[[1]]
      targetTensor <- x[[2]]

      newShape <- reticulate::tuple( tf$shape( targetTensor )[2], tf$shape( targetTensor )[3] )
      func <- tfResizingFunction( self$interpolationType )
      self$resampledTensor <- func( x[[1]], size = newShape )

      return( self$resampledTensor )
      },

    compute_output_shape = function( input_shape )
      {
      return( tensorflow::tf$keras$backend$int_shape( self$resampledTensor ) )
      }
  )
)

#' Resampling a spatial tensor to a target tensor (2-D).
#'
#' Resamples a spatial tensor based on a target tensor and interpolation type.
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{nearest},
#' \code{linear}, \code{bilinear},
#' \code{cubic}, or \code{bicubic}.
#' @param name The name of the layer
#' @param trainable Whether the layer weights will be updated during training.
#'
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
#' @rdname layer_resample_tensor_to_target_tensor_2d
layer_resample_tensor_to_target_tensor_2d <- function(
  object,
  interpolationType = c( "nearestNeighbor",
                         "nearest",
                         "linear",
                         "bilinear",
                         "cubic",
                         "bicubic"
                        ),
  name = NULL,
  trainable = FALSE ) {
  interpolationType = match.arg( interpolationType )
  create_layer( ResampleTensorToTargetTensorLayer2D, object,
                list( interpolationType = interpolationType,
                      name = name, trainable = trainable ) )
}


#' Creates a resampled tensor (to target tensor) layer (3-D)
#'
#' Creates a lambda layer which interpolates/resizes an input tensor based
#' on the specified target tensor
#'
#' @docType class
#'
#'
#' @section Arguments:
#' \describe{
#'  \item{interpolationType}{Type of interpolation.  Can be
#'    \code{'nearestNeighbor'}, \code{'nearestNeighbor'},
#'    \code{'linear'}, \code{'bilinear'},
#'    \code{'cubic'}, or \code{'bicubic'}}
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
#' @name ResampleTensorToTargetTensorLayer3D
NULL

#' @export
ResampleTensorToTargetTensorLayer3D <- R6::R6Class(
  "ResampleTensorToTargetTensorLayer3D",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    interpolationType = 'nearestNeighbor',

    initialize = function( interpolationType =
                             c( "nearestNeighbor",
                                "nearest",
                                "linear",
                                "bilinear",
                                "cubic",
                                "bicubic"
                                ) )
      {
      self$resampledTensor <- NULL

      interpolationType = match.arg( interpolationType )
      self$interpolationType <- interpolationType
      },

    call = function( x, mask = NULL )
      {
      tf <- tensorflow::tf
      K <- tf$keras$backend

      sourceTensor <- x[[1]]
      targetTensor <- x[[2]]

      channelSize <- sourceTensor$get_shape()[[5]]

      # Do yz

      newShape_squeeze_yz <- reticulate::tuple( -1L, tf$shape( sourceTensor )[3], tf$shape( sourceTensor )[4], channelSize )
      squeezeTensor_yz <- tf$reshape( sourceTensor, newShape_squeeze_yz )

      resampledTensor_yz <- NULL
      newShape_yz <- reticulate::tuple( tf$shape( targetTensor )[3], tf$shape( targetTensor )[4] )
      func <- tfResizingFunction( self$interpolationType )
      resampledTensor_yz <- func( squeezeTensor_yz, size = newShape_yz )

      newShape_yz <- reticulate::tuple( -1L, tf$shape( sourceTensor )[2], tf$shape( targetTensor )[3], tf$shape( targetTensor )[4], channelSize )
      resumeTensor_yz <- tf$reshape( resampledTensor_yz, newShape_yz )

      # Do x

      reorientedTensor <- tf$transpose( resumeTensor_yz, c( 0L, 3L, 2L, 1L, 4L ) )

      newShape_squeeze_x <- reticulate::tuple( -1L, tf$shape( targetTensor )[3], tf$shape( sourceTensor )[2], channelSize )
      squeezeTensor_x <- tf$reshape( reorientedTensor, newShape_squeeze_x )

      resampledTensor_x <- NULL
      newShape_x <- reticulate::tuple( tf$shape( targetTensor )[3], tf$shape( targetTensor )[2] )
      func = tfResizingFunction( self$interpolationType )
      resampledTensor_x <- func( squeezeTensor_x, size = newShape_x )

      newShape_x <- reticulate::tuple( -1L, tf$shape( targetTensor )[4], tf$shape( targetTensor )[3], tf$shape( targetTensor )[2], channelSize )
      resumeTensor_x <- tf$reshape( resampledTensor_x, newShape_x )

      self$resampledTensor <- tf$transpose( resumeTensor_x, c( 0L, 3L, 2L, 1L, 4L ) )

      return( self$resampledTensor )
      },

    compute_output_shape = function( input_shape )
      {
      return( tensorflow::tf$keras$backend$int_shape( self$resampledTensor ) )
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
#' @param interpolationType type of interpolation for resampling.  Can be
#' \code{nearestNeighbor}, \code{nearest},
#' \code{linear}, \code{bilinear},
#' \code{cubic}, or \code{bicubic}.
#' @param name The name of the layer
#' @param trainable Whether the layer weights will be updated during training.
#'
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
#' @rdname layer_resample_tensor_to_target_tensor_3d
layer_resample_tensor_to_target_tensor_3d <- function(
  object,
  interpolationType = c( "nearestNeighbor",
                         "nearest",
                         "linear",
                         "bilinear",
                         "cubic",
                         "bicubic"
                        ),
  name = NULL,
  trainable = FALSE ) {
  interpolationType = match.arg( interpolationType )
  create_layer( ResampleTensorToTargetTensorLayer3D, object,
                list( interpolationType = interpolationType,
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

