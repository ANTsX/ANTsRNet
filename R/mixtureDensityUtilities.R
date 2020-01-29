#' Mixture density network layer
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{outputs <- layer_mixture_density( inputs, resampledSize )}
#'
#' @section Arguments:
#' \describe{
#'  \item{outputDimensino}{}
#'  \item{numberOfMixtures}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' Create custom density layers for each parameter of the
#' mixed Gaussians. (mu, sigma, pi). I could not get the approach
#' from the original implementation to work:
#'
#'   https://github.com/cpmpercussion/keras-mdn-layer/blob/master/mdn/__init__.py#L28-L73
#'
#' where the author used the keras dense layers to create the
#' custom MDN layer and assign the trainable weights directly
#' thus circumventing the add_weight() function.  Instead, I
#' recreated dense layer functionality using the keras definition
#' here:
#'
#'   https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L796-L937
#'
#' @author Tustison NJ
#'
#' @return .
#'
#' @name MixtureDensityNetworkLayer
NULL

#' @export
MixtureDensityNetworkLayer <- R6::R6Class( "MixtureDensityNetworkLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    outputDimension = NULL,

    numberOfMixtures = NULL,

    initialize = function( outputDimension, numberOfMixtures )
      {
      K <- keras::backend()
      if( K$backend() != 'tensorflow' )
        {
        stop( "Error:  tensorflow is required for this implementation." )
        }

      self$outputDimension <- as.integer( outputDimension )
      self$numberOfMixtures <- as.integer( numberOfMixtures )
      },

    build = function( input_shape )
      {
      inputDimension <- input_shape[-1]

      units1 <- self$outputDimension * self$numberOfMixtures

      self$muKernel <- self$add_weight(
        name = "mu_kernel",
        shape = shape( inputDimension, units1 ),
        initializer = initializer_random_normal(),
        trainable = TRUE )
      self$muBias <- self$add_weight(
        name = "mu_bias",
        shape = shape( units1 ),
        initializer = initializer_zeros(),
        trainable = TRUE )

      self$sigmaKernel <- self$add_weight(
        name = "sigma_kernel",
        shape = shape( inputDimension, units1 ),
        initializer = initializer_random_normal(),
        trainable = TRUE )
      self$sigmaBias <- self$add_weight(
        name = "sigma_bias",
        shape = shape( units1 ),
        initializer = initializer_zeros(),
        trainable = TRUE )

      units2 <- self$numberOfMixtures

      self$piKernel <- self$add_weight(
        name = "pi_kernel",
        shape = shape( inputDimension, units2 ),
        initializer = initializer_random_normal(),
        trainable = TRUE )
      self$piBias <- self$add_weight(
        name = "pi_bias",
        shape = shape( units2 ),
        initializer = initializer_zeros(),
        trainable = TRUE )
      },

    call = function( inputs, mask = NULL )
      {
      K <- keras::backend()

      # dense layer for mu (mean) of the gaussians
      muOutput <- K$dot( inputs, self$muKernel )
      muOutput <- K$bias_add( muOutput, self$muBias,
        data_format = 'channels_last' )

      # dense layer for sigma (variance) of the gaussians
      sigmaOutput <- K$dot( inputs, self$sigmaKernel )
      sigmaOutput <- K$bias_add( sigmaOutput, self$sigmaBias,
        data_format = 'channels_last' )

      # Avoid NaN's by pushing sigma through the following custom
      #  activation
      sigmaOutput <- K$elu( sigmaOutput ) + 1 + K$epsilon()

      # dense layer for pi (amplitude) of the gaussians
      piOutput <- K$dot( inputs, self$piKernel )
      piOutput <- K$bias_add( piOutput, self$piBias,
        data_format = 'channels_last' )

      output <- layer_concatenate( list( muOutput, sigmaOutput, piOutput ),
        name = "mdn_ouputs" )

      return( output )
      },

    compute_output_shape = function( input_shape )
      {
      return( list( unlist( input_shape[[1]] ),
        as.integer( 2L * self$outputDimension * self$numberOfMixtures +
          self$numberOfMixtures ) ) )
      }

  )
)

#' Mixture density layer
#'
#' Wraps a custom mixture density layer.
#'
#' @param outputDimension output dimension
#' @param numberOfMixtures number of Gaussians used to model the function
#'
#' @return a keras layer tensor
#' @export
layer_mixture_density <- function( objects,
  outputDimension, numberOfMixtures, trainable = TRUE ) {
create_layer( MixtureDensityNetworkLayer, objects,
    list( outputDimension = outputDimension,
      numberOfMixtures = numberOfMixtures, trainable = TRUE )
    )
}

#' Returns a loss function for the mixture density.
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param outputDimension output dimension
#' @param numberOfMixtures number of mixture components
#' @return a function providing the mean square error accuracy
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
getMixtureDensityLossFunction <- function( outputDimension, numberOfMixtures )
{
  lossFunction <- function( y_true, y_pred )
    {
    outputDimension <- as.integer( outputDimension )
    numberOfMixtures <- as.integer( numberOfMixtures )
    dimension <- as.integer( numberOfMixtures * outputDimension )

    y_pred <- tensorflow::tf$reshape( y_pred,
      list( -1L, ( 2L * dimension ) + numberOfMixtures ),
      name = 'reshape_ypred_loss' )
    y_true <- tensorflow::tf$reshape( y_true,
      list( -1L, outputDimension ), name = 'reshape_ytrue_loss' )

    splitTensors <- tensorflow::tf$split( y_pred,
      num_or_size_splits = list( dimension, dimension, numberOfMixtures ),
      axis = -1L, name = "mdn_coef_split" )

    outputMu <- splitTensors[[1]]
    outputSigma <- splitTensors[[2]]
    outputPi <- splitTensors[[3]]

    # Construct the mixture models

    tfp <- tensorflow::import( "tensorflow_probability" )
    tfd <- tfp$distributions

    categoricalDistribution <- tfd$Categorical( logits = outputPi )
    componentSplits <- rep.int( outputDimension, numberOfMixtures )
    mu <- tensorflow::tf$split( outputMu,
      num_or_size_splits = componentSplits, axis = 1L )
    sigma <- tensorflow::tf$split( outputSigma,
      num_or_size_splits = componentSplits, axis = 1L )

    components <- list()
    for( i in seq_len( length( mu ) ) )
      {
      components[[i]] <- tfd$MultivariateNormalDiag(
        loc = mu[[i]], scale_diag = sigma[[i]] )
      }
    mixture <- tfd$Mixture( cat = categoricalDistribution,
      components = components )

    loss <- mixture$log_prob( y_true )
    loss <- tensorflow::tf$negative( loss )
    loss <- tensorflow::tf$reduce_mean( loss )

    return( loss )
    }

  with( tensorflow::tf$name_scope( "MixtureDensityNetwork" ),
    {
    return( lossFunction )
    } )
}

#' Returns a sampling function for the mixture density.
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param outputDimension output dimension
#' @param numberOfMixtures number of mixture components
#' @return a function for sampling a mixture density
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
getMixtureDensitySamplingFunction <- function( outputDimension, numberOfMixtures )
{
  samplingFunction <- function( y_pred )
    {
    outputDimension <- as.integer( outputDimension )
    numberOfMixtures <- as.integer( numberOfMixtures )
    dimension <- as.integer( numberOfMixtures * outputDimension )

    y_pred <- tensorflow::tf$reshape( y_pred,
      c( -1L, ( 2L * dimension ) + numberOfMixtures ),
      name = 'reshape_ypred' )

    splitTensors <- tensorflow::tf$split( y_pred,
      num_or_size_splits = c( dimension, dimension, numberOfMixtures ),
      axis = 1L, name = "mdn_coef_split" )

    outputMu <- splitTensors[[1]]
    outputSigma <- splitTensors[[2]]
    outputPi <- splitTensors[[3]]

    # Construct the mixture models

    tfp <- tensorflow::import( "tensorflow_probability" )
    tfd <- tfp$distributions

    categoricalDistribution <- tfd$Categorical( logits = outputPi )
    componentSplits <- rep.int( outputDimension, numberOfMixtures )
    mu <- tensorflow::tf$split( outputMu,
      num_or_size_splits = componentSplits, axis = 1L )
    sigma <- tensorflow::tf$split( outputSigma,
      num_or_size_splits = componentSplits, axis = 1L )

    components <- list()
    for( i in seq_len( length( mu ) ) )
      {
      components[[i]] <- tfd$MultivariateNormalDiag(
        loc = mu[[i]], scale_diag = sigma[[i]] )
      }
    mixture <- tfd$Mixture( cat = categoricalDistribution,
      components = components )

    sample <- mixture$sample()

    return( sample )
    }

  with( tensorflow::tf$name_scope( "MixtureDensityNetwork" ),
    {
    return( samplingFunction )
    } )
}

#' Returns a MSE accuracy function for the mixture density.
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param outputDimension output dimension
#' @param numberOfMixtures number of mixture components
#' @return a function providing the mean square error accuracy
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
getMixtureDensityMseAccuracyFunction <- function( outputDimension, numberOfMixtures )
{
  mseAccuracyFunction <- function( y_true, y_pred )
    {
    outputDimension <- as.integer( outputDimension )
    numberOfMixtures <- as.integer( numberOfMixtures )
    dimension <- as.integer( numberOfMixtures * outputDimension )

    y_pred <- tensorflow::tf$reshape( y_pred,
      c( -1L, ( 2L * dimension ) + numberOfMixtures ),
      name = 'reshape_ypred' )
    y_true <- tensorflow::tf$reshape( y_true,
      c( -1L, outputDimension ), name = 'reshape_ytrue' )

    splitTensors <- tensorflow::tf$split( y_pred,
      num_or_size_splits = c( dimension, dimension, numberOfMixtures ),
      axis = 1L, name = "mdn_coef_split" )

    outputMu <- splitTensors[[1]]
    outputSigma <- splitTensors[[2]]
    outputPi <- splitTensors[[3]]

    # Construct the mixture models

    tfp <- tensorflow::import( "tensorflow_probability" )
    tfd <- tfp$distributions

    categoricalDistribution <- tfd$Categorical( logits = outputPi )
    componentSplits <- rep.int( outputDimension, numberOfMixtures )
    mu <- tensorflow::tf$split( outputMu,
      num_or_size_splits = componentSplits, axis = 1L )
    sigma <- tensorflow::tf$split( outputSigma,
      num_or_size_splits = componentSplits, axis = 1L )

    components <- list()
    for( i in seq_len( length( mu ) ) )
      {
      components[[i]] <- tfd$MultivariateNormalDiag(
        loc = mu[[i]], scale_diag = sigma[[i]] )
      }
    mixture <- tfd$Mixture( cat = categoricalDistribution,
      components = components )

    sample <- mixture$sample()
    mse <- tensorflow::tf$reduce_mean(
      tensorflow::tf$square( sample - y_true ), axis = -1L )

    return( mse )
    }

  with( tensorflow::tf$name_scope( "MixtureDensityNetwork" ),
    {
    return( mseAccuracyFunction )
    } )
}

#' Splits the mixture parameters.
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param parameters vector parameter to split
#' @param outputDimension output dimension
#' @param numberOfMixtures number of mixture components
#' @return separate mixture parameters
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
splitMixtureParameters <- function(
   parameters, outputDimension, numberOfMixtures )
{
  dimension <- as.integer( numberOfMixtures * outputDimension )
  mu <- parameters[, 1:dimension, drop = FALSE]
  sigma <- parameters[, ( dimension + 1 ):( 2 * dimension ), drop = FALSE]
  pi_logits <- parameters[, ( 2 * dimension + 1 ):( 2 * dimension + numberOfMixtures ), drop = FALSE]
  return( list( mu = mu, sigma = sigma, pi = pi_logits ) )
}

#' Softmax function for mixture density with temperature adjustment
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param logits input
#' @return softmax loss value
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
mixture_density_network_softmax <- function( logits, temperature = 1.0 )
{
  np <- reticulate::import( "numpy" )

  e <- np$array( logits ) / temperature
  e <- e - np$max( e )
  e <- np$exp( e )
  distribution <- e / np$sum( e )

  return( distribution )
}

#' Sample from a categorical distribution
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param distribution input categorical distribution from which
#' to sample.
#' @return a single sample
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
sampleFromCategoricalDistribution <- function( distribution )
{
  r <- runif( 1 )

  accumulate <- 0
  for( i in seq_len( length( distribution ) ) )
    {
    accumulate <- accumulate + distribution[i]
    if( accumulate >= r )
      {
      return( i )
      }
    }
  tensorflow::tf$logging$info( 'Error: sampling categorical model.' )

  return( -1 )
}

#' Sample from a distribution
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param distribution input distribution from which to sample.
#' @return a single sample
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
sampleFromOutput <- function( parameters, outputDimension,
  numberOfMixtures, temperature = 1.0, sigmaTemperature = 1.0 )
{
  splits <- splitMixtureParameters( parameters, outputDimension,
    numberOfMixtures )
  piSoftmax <- mixture_density_network_softmax( splits$pi, temperature = temperature )
  m <- sampleFromCategoricalDistribution( piSoftmax )

  muVector <-
    splits$mu[( ( m - 1 ) * outputDimension + 1 ):( m * outputDimension ), drop = FALSE]
  sigmaVector <-
    splits$sigma[( ( m - 1 ) * outputDimension + 1 ):( m * outputDimension ), drop = FALSE] *
    sigmaTemperature

  sample <- NA
  if( length( muVector ) == 1 )
    {
    sample <- rnorm( 1, mean = muVector[1], sd = sigmaVector[1] )
    } else {
    sample <- mvtnorm::rmvnorm( 1, mean = muVector, sigma = diag( sigmaVector ) )
    }
  return( sample )
}
