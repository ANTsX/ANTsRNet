#' Creates a log softmax layer
#'
#' Creates a log softmax layer taken from 
#'
#'   \url{https://github.com/tensorflow/tensorflow/pull/25514/files}
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'   \item{axis}{Integer specifying the axis.}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'   \code{$call} main body.
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return a log softmax layer
#'
#' @name LogSoftmaxLayer
NULL

#' @export
LogSoftmaxLayer <- R6::R6Class( "LogSoftmaxLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    axis = -1L,

    initialize = function( axis = -1L )
      {
      self$axis = axis
      },

    call = function( inputs, mask = NULL )
      {
      return( tensorflow::tf$nn$log_softmax( inputs, axis = self$axis ) )
      },

    compute_output_shape = function( input_shape ) 
      {
      return( input_shape )
      }  
  )
)

#' Log softmax layer
#'
#' Creates a log softmax layer
#'
#' @param axis Integer specifying which axis.
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
layer_activation_log_softmax <- function( object, axis = -1, trainable = TRUE ) {
create_layer( LogSoftmaxLayer, object,
    list( axis = axis, trainable = trainable ) )
}
