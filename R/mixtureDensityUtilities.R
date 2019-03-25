#' Returns a custom activation functions to prevent NaNs in loss function.
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param x input tensor
#' @return an activation function
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#' K <- keras::backend()
#' x <- K$ones( list( 1, 1 ) )
#' activation <- activation_elu_plus_one_plus_epsilon( x )
#'
#' @import keras
#' @export
activation_elu_plus_one_plus_epsilon <- function( x )
{
  K <- keras::backend()
  return( K$elu( x ) + 1 + K$epsilon() )
}

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

      self$outputDimension <- outputDimension
      self$numberOfMixtures <- numberOfMixtures

      with( tensorflow::tf$name_scope( "MixtureDensityNetwork" ),
        {
        self$mu <- layer_dense( units = self$numberOfMixtures * self$outputDimension,
          name = "mdn_mu" )
        self$sigma <- layer_dense( units = self$numberOfMixtures * self$outputDimension,
          activation = activation_elu_plus_one_plus_epsilon, name = "mdn_sigma" )
        self$pi <- layer_dense( units = self$numberOfMixtures, name = "mdn_pi" )
        } )
      },

    build = function( input_shape )
      {
      self$mu$build( input_shape )
      self$sigma$build( input_shape )
      self$pi$build( input_shape )

      self$trainable_weights <- list( self$mu$trainable_weights,
        self$sigma$trainable_weights, self$pi$trainable_weights )
      self$non_trainable_weights <- list( self$mu$non_trainable_weights,
        self$sigma$non_trainable_weights, self$pi$non_trainable_weights )
      },

    call = function( inputs, mask = NULL )
      {
      with( tensorflow::tf$name_scope( "MixtureDensityNetwork" ),
        {
        output <- layer_concatenate( list( self$mu( inputs ),
          self$sigma( inputs ), self$pi( inputs ) ), name = "mnd_ouputs" )
        } )
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

layer_mixture_density <- function( objects,
  outputDimension, numberOfMixtures ) {
create_layer( MixtureDensityNetworkLayer, objects,
    list( outputDimension = outputDimension,
      numberOfMixtures = numberOfMixtures )
    )
}

#' Returns a loss function for the mixture density.
#'
#' Ported from:
#'
#'         https://github.com/cpmpercussion/keras-mdn-layer/
#'
#' @param outputDimension output dimension
#' @param numberOfMixes number of mixture components
#' @return a function providing the mean square error accuracy
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
getMixtureDensityLossFunction <- function( outputDimension, numberOfMixes )
{
  lossFunction <- function( y_true, y_pred )
    {
    outputDimension <- as.integer( outputDimension )
    numberOfMixes <- as.integer( numberOfMixes )
    dimension <- as.integer( numberOfMixes * outputDimension )

    y_pred <- tensorflow::tf$reshape( y_pred,
      c( -1L, ( 2L * dimension ) + numberOfMixes ),
      name = 'reshape_ypred_loss' )
    y_true <- tensorflow::tf$reshape( y_true,
      c( -1L, outputDimension ), name = 'reshape_ytrue_loss' )

    splitTensors <- tensorflow::tf$split( y_pred,
      num_or_size_splits = c( dimension, dimension, numberOfMixes ),
      axis = -1L, name = "mdn_coef_split" )

    outputMu <- splitTensors[[1]]
    outputSigma <- splitTensors[[2]]
    outputPi <- splitTensors[[3]]

    # Construct the mixture models

    tfp <- tensorflow::import( "tensorflow_probability" )
    tfd <- tfp$distributions

    categoricalDistribution <- tfd$Categorical( logits = outputPi )
    componentSplits <- rep.int( outputDimension, numberOfMixes )
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

    loss <- tensorflow::tf$reduce_mean(
      tensorflow::tf$negative( mixture$log_prob( y_true ) ) )

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
#' @param numberOfMixes number of mixture components
#' @return a function for sampling a mixture density
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
getMixtureDensitySamplingFunction <- function( outputDimension, numberOfMixes )
{
  samplingFunction <- function( y_pred )
    {
    outputDimension <- as.integer( outputDimension )
    numberOfMixes <- as.integer( numberOfMixes )
    dimension <- as.integer( numberOfMixes * outputDimension )

    y_pred <- tensorflow::tf$reshape( y_pred,
      c( -1L, ( 2L * dimension ) + numberOfMixes ),
      name = 'reshape_ypred' )

    splitTensors <- tf$split( y_pred,
      num_or_size_splits = c( dimension, dimension, numberOfMixes ),
      axis = 1L, name = "mdn_coef_split" )

    outputMu <- splitTensors[[1]]
    outputSigma <- splitTensors[[2]]
    outputPi <- splitTensors[[3]]

    # Construct the mixture models

    tfp <- tensorflow::import( "tensorflow_probability" )
    tfd <- tfp$distributions

    categoricalDistribution <- tfd$Categorical( logits = outputPi )
    componentSplits <- rep.int( outputDimension, numberOfMixes )
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
#' @param numberOfMixes number of mixture components
#' @return a function providing the mean square error accuracy
#' @author Tustison NJ
#' @examples
#'
#' library( keras )
#'
#'
#' @import keras
#' @export
getMixtureDensityMseAccuracyFunction <- function( outputDimension, numberOfMixes )
{
  mseAccuracyFunction <- function( y_pred )
    {
    outputDimension <- as.integer( outputDimension )
    numberOfMixes <- as.integer( numberOfMixes )
    dimension <- as.integer( numberOfMixes * outputDimension )

    y_pred <- tensorflow::tf$reshape( y_pred,
      c( -1L, ( 2L * dimension ) + numberOfMixes ),
      name = 'reshape_ypred' )
    y_true <- tensorflow::tf$reshape( y_true,
      c( -1L, outputDimension ), name = 'reshape_ytrue' )

    splitTensors <- tf$split( y_pred,
      num_or_size_splits = c( dimension, dimension, numberOfMixes ),
      axis = 1L, name = "mdn_coef_split" )

    outputMu <- splitTensors[[1]]
    outputSigma <- splitTensors[[2]]
    outputPi <- splitTensors[[3]]

    # Construct the mixture models

    tfp <- tensorflow::import( "tensorflow_probability" )
    tfd <- tfp$distributions

    categoricalDistribution <- tfd$Categorical( logits = outputPi )
    componentSplits <- rep.int( outputDimension, numberOfMixes )
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
#' @param numberOfMixes number of mixture components
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
   parameters, outputDimension, numberOfMixes )
{
  dimension <- as.integer( numberOfMixes * outputDimension )
  mu <- parameters[1:dimension]
  sigma <- parameters[( dimension + 1 ):( 2 * dimension )]
  pi_logits <- parameters[( 2 * dimension + 1 ):length( parameters )]
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

  e <- np$array( logits )
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
  for( i in seq_length( length( distribution ) ) )
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
  m <- sampleFromCategricalDistribution( piSoftmax )

  muVector <-
    splits$mu[( m * outputDimension ):( ( m + 1 ) * outputDimension )]
  sigmaVector <-
    splits$sigma[( m * outputDimension ):( ( m + 1 ) * outputDimension )] *
    sigmaTemperature

  sample <- mvtnorm::rmvnorm( 1, mean = muVector, sigma = diag( sigmaVector ) )
}
