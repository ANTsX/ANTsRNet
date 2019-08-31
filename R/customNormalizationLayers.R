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

  public = list(

    axis = -1L,

    epsilon = 1e-3,

    center = TRUE,

    scale = TRUE,

    betaInitializer = "zeros",

    gammaInitializer = "ones",

    betaRegularizer = NULL,

    gammaRegularizer = NULL,

    betaConstraint = NULL,

    gammaConstraint = NULL,

    interpolationType = 'nearestNeighbor',

    initialize = function( axis = NULL, epsilon = 1e-3, center = TRUE, scale = TRUE,
      betaInitializer = "zeros", gammaInitializer = "ones",  betaRegularizer = NULL,
      gammaRegularizer = NULL, betaConstraint = NULL, gammaConstraint = NULL )
      {
      self$axis = axis
      if( self$axis == 1 )
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
      dimensionality <- length( input_shape )

      if( ( self$axis != -1L ) && ( dimensionality == 2 ) )
        {
        stop( "Error:  Cannot specify an axis for rank 1 tensor." )
        }

      shape <- NULL
      if( self$axis == -1L )
        {
        shape <- shape( 1 )
        } else {
        shape <- shape( input_shape[self$axis] )
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

      },

   call = function( inputs, mask = NULL )
      {
      K <- keras::backend()

      inputShape <- K$int_shape( inputs )
      reductionAxes <- list( seq( from = 1, to = length( inputShape ) ) )

      if( self$axis != -1 )
        {
        reductionAxes[[self$axis]] <- NULL
        }
      reductionAxes[[1]] <- NULL

      mean <- K$mean( inputs, reductionAxes, keepdims = TRUE )
      stddev <- K$sd( inputs, reductionAxes, keepdims = TRUE )

      normed <- ( inputs - mean ) / ( stddev + epsilon )

      broadcastShape <- rep( 1, length( inputShape ) )
      if( self$axis != -1L )
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
layer_instance_normalization <- function( object, shape,
    interpolationType = 'nearestNeighbor', name = NULL,
    trainable = FALSE ) {
create_layer( ResampleTensorLayer2D, object,
    list( shape = shape, interpolationType = interpolationType,
        name = name, trainable = trainable ) )
}
