#' Creates an instance normalization layer
#'
#' Creates an instance normalization layer as described in the paper
#'
#'   \url{https://arxiv.org/abs/1701.02096}
#'
#' with the implementation ported from the following python implementation
#'
#'   \url{https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py}
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{inputs %>% layer_instance_normalization()
#'              }
#'
#' @section Arguments:
#' \describe{
#'   \item{axis}{Integer specifying which axis should be normalized, typically
#'               the feature axis.  For example, after a Conv2D layer with
#'               `channels_first`, set axis = 1.  Setting `axis=-1L` will
#'               normalize all values in each instance of the batch.  Axis 0
#'               is the batch dimension for tensorflow backend so we throw an
#'               error if `axis = 0`.}
#'   \item{epsilon}{Small float added to the variance to avoid dividing by 0.}
#'   \item{center}{If TRUE, add `beta` offset to normalized tensor.}
#'   \item{scale}{If TRUE, multiply by `gamma`.}
#'   \item{betaInitializer}{Intializer for the beta weight.}
#'   \item{gammaInitializer}{Intializer for the gamma weight.}
#'   \item{betaRegularizer}{Regularizer for the beta weight.}
#'   \item{gammaRegularizer}{Regularizer for the gamma weight.}
#'   \item{betaConstraint}{Optional constraint for the beta weight.}
#'   \item{gammaConstraint}{Optional constraint for the gamma weight.}
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
#' @return an instance normalization layer
#'
#' @name InstanceNormalizationLayer
NULL

#' @export
InstanceNormalizationLayer <- R6::R6Class( "InstanceNormalizationLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    axis = NULL,

    epsilon = 1e-3,

    center = TRUE,

    scale = TRUE,

    betaInitializer = "zeros",

    gammaInitializer = "ones",

    betaRegularizer = NULL,

    gammaRegularizer = NULL,

    betaConstraint = NULL,

    gammaConstraint = NULL,

    initialize = function( axis = NULL, epsilon = 1e-3, center = TRUE, scale = TRUE,
      betaInitializer = "zeros", gammaInitializer = "ones",  betaRegularizer = NULL,
      gammaRegularizer = NULL, betaConstraint = NULL, gammaConstraint = NULL )
      {
      self$axis = axis
      if( ! is.null( self$axis ) && self$axis == 1L )
        {
        stop( "Error:  axis can't be 1." )
        }
      self$epsilon = epsilon
      self$center = center
      self$scale = scale
      self$betaInitializer = betaInitializer
      self$gammaInitializer = gammaInitializer
      self$betaRegularizer = betaRegularizer
      self$gammaRegularizer = gammaRegularizer
      self$betaConstraint = betaConstraint
      self$gammaConstraint = gammaConstraint
      },

    build = function( input_shape )
      {
      dimensionality <- as.integer( length( input_shape ) )

      if( ( ! is.null( self$axis ) ) && ( dimensionality == 2L ) )
        {
        stop( "Error:  Cannot specify an axis for rank 1 tensor." )
        }

      shape <- NULL
      if( is.null( self$axis ) )
        {
        shape <- shape( 1L )
        } else {
        shape <- shape( input_shape[self$axis] )
        }

      if( self$scale )
        {
        self$gamma <- self$add_weight(
          name = "gamma",
          shape = shape,
          initializer = self$gammaInitializer,
          regularizer = self$gammaRegularizer,
          constraint = self$gammaConstraint,
          trainable = TRUE )
        } else {
        self$gamma <- NULL
        }

      if( self$center )
        {
        self$beta <- self$add_weight(
          name = "beta",
          shape = shape,
          initializer = self$betaInitializer,
          regularizer = self$betaRegularizer,
          constraint = self$betaConstraint,
          trainable = TRUE )
        } else {
        self$beta <- NULL
        }
      },

   call = function( inputs, mask = NULL )
      {
      K <- keras::backend()

      inputShape <- K$int_shape( inputs )
      reductionAxes <- as.list( seq( from = 0,
        to = as.integer( length( inputShape ) - 1 ) ) )

      if( ! is.null( self$axis ) )
        {
        reductionAxes[[self$axis]] <- NULL
        }
      reductionAxes[[1]] <- NULL

      mean <- K$mean( inputs, reductionAxes, keepdims = TRUE )
      stddev <- K$std( inputs, reductionAxes, keepdims = TRUE )

      normed <- ( inputs - mean ) / ( stddev + self$epsilon )

      broadcastShape <- rep( 1L, length( inputShape ) )
      if( ! is.null( self$axis ) )
        {
        broadcastShape[self$axis] <- inputShape[self$axis]
        }

      if( self$scale == TRUE )
        {
        broadcastGamma <- K$reshape( self$gamma, broadcastShape )
        normed <- normed * broadcastGamma
        }
      if( self$center == TRUE )
        {
        broadcastBeta <- K$reshape( self$beta, broadcastShape )
        normed <- normed + broadcastBeta
        }
      return( normed )
      }
  )
)

#' Instance normalization layer
#'
#' Creates an instance normalization layer
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param axis Integer specifying which axis should be normalized, typically
#'               the feature axis.  For example, after a Conv2D layer with
#'               `channels_first`, set axis = 1.  Setting `axis=-1L` will
#'               normalize all values in each instance of the batch.  Axis 0
#'               is the batch dimension for tensorflow backend so we throw an
#'               error if `axis = 0`.
#' @param epsilon Small float added to the variance to avoid dividing by 0.
#' @param center If TRUE, add `beta` offset to normalized tensor.
#' @param scale If TRUE, multiply by `gamma`.
#' @param betaInitializer Intializer for the beta weight.
#' @param gammaInitializer Intializer for the gamma weight.
#' @param betaRegularizer Regularizer for the beta weight.
#' @param gammaRegularizer Regularizer for the gamma weight.
#' @param betaConstraint Optional constraint for the beta weight.
#' @param gammaConstraint Optional constraint for the gamma weight.
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
layer_instance_normalization <- function( object, axis = NULL,
  epsilon = 1e-3, center = TRUE, scale = TRUE,
  betaInitializer = "zeros", gammaInitializer = "ones",
  betaRegularizer = NULL, gammaRegularizer = NULL,
  betaConstraint = NULL, gammaConstraint = NULL, trainable = TRUE ) {
create_layer( InstanceNormalizationLayer, object,
    list( axis = axis, epsilon = epsilon, center = center,
      scale = scale, betaInitializer = "zeros",
      gammaInitializer = "ones", betaRegularizer = NULL,
      gammaRegularizer = NULL, betaConstraint = NULL,
      gammaConstraint = NULL, trainable = trainable ) )
}
