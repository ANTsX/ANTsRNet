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
#'  \item{weights}{initialization weights with shapes.}
#'  \item{betaInitializer}{initial bias parameter. Only used if weights
#'                         aren't specified.}
#'  \item{gammaInitializer}{initial scale parameter. Only used if weights
#'                          aren't specified.}
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

    betaInitializer = NULL,

    gammaInitializer = NULL,

    weights = NULL,

    initialize = function( axis = -1L, momentum = 0.9,
      betaInitializer = initializer_zeros(),
      gammaInitializer = initializer_ones(), weights = NULL )
      {
      self$axis <- axis
      self$momentum <- momentum
      self$betaInitializer <- betaInitializer
      self$gammaInitializer <- gammaInitializer
      self$initial_weights <- weights
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
      shape <- reticulate::tuple( input_shape[[index]] )

      self$gamma <- K$variable( value = self$gammaInitializer( shape ) )
      self$beta <- K$variable( self$betaInitializer( shape ) )
      self$trainable_weights <- list( self$gamma, self$beta )

      if( ! is.null( self$initial_weights ) )
        {
        self$set_weights( self$initial_weights )
        self$initial_weights <- NULL
        }
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
  axis = -1, momentum = 0.9, betaInitializer = initializer_zeros(),
  gammaInitializer = initializer_ones(), weights = NULL ) {
create_layer( ScaleLayer, objects,
    list( axis = axis, momentum = momentum, betaInitializer = betaInitializer,
      gammaInitializer = gammaInitializer, weights = weights )
    )
}
