#' Custom scale layer
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{outputs <- layer_scale( inputs, resampledSize )}
#'
#' @section Arguments:
#' \describe{
#'  \item{axis}{integer specifying which axis to normalize.}
#'  \item{momentum}{momentum value used for computation of the exponential
#'                  average of the mean and standard deviation.}
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
#' @return .
#'
#' @name ScaleLayer
NULL

#' @export
ScaleLayer <- R6::R6Class( "ScaleLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    axis = -1L,

    momentum = 0.9,

    initialize = function( axis = -1L, momentum = 0.9 )
      {
      self$axis <- axis
      self$momentum <- momentum
      },

    build = function( input_shape )
      {
      K <- keras::backend()

      self$inputShape <- input_shape

      index <- self$axis
      if( self$axis == -1 )
        {
        index <- length( self$inputShape )
        }
      output_shape <- reticulate::tuple( input_shape[[index]] )

      self$gamma <- self$add_weight(
        name = "gamma",
        shape = output_shape,
        initializer = initializer_ones(),
        trainable = TRUE )
      self$beta <- self$add_weight(
        name = "beta",
        shape = output_shape,
        initializer = initializer_zeros(),
        trainable = TRUE )
      },

    call = function( inputs, mask = NULL )
      {
      K <- keras::backend()

      broadcastShape <- as.list( rep( 1, length( self$inputShape ) ) )

      index <- self$axis
      if( self$axis == -1 )
        {
        index <- length( self$inputShape )
        }
      broadcastShape[[index]] <- self$inputShape[[index]]
      broadcastShape <- K$cast( broadcastShape, dtype = "int32" )

      output <- K$reshape( self$gamma, broadcastShape ) * inputs +
        K$reshape( self$beta, broadcastShape )

      return( output )
      }

  )
)

layer_scale <- function( objects,
  axis = -1, momentum = 0.9 ) {
create_layer( ScaleLayer, objects,
    list( axis = axis, momentum = momentum )
    )
}
