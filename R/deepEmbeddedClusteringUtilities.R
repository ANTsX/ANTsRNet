#' Function for creating a symmetric autoencoder model.
#'
#' Builds an autoencoder based on the specified array definining the
#' number of units in the encoding branch.  Ported to Keras R from the
#' Keras python implementation here:
#'
#' \url{https://github.com/XifengGuo/DEC-keras/blob/master/DEC.py}
#'
#' @param numberOfUnitsLayer an array of defining the number of units
#' in the encoding branch
#'
#' @return two models:  the encoder and auto-encoder
#'
#' @author Tustison NJ
#' @export

createAutoencoderModel <- function( numberOfUnitsPerLayer,
                                    activation = 'relu',
                                    initializer = 'glorot_uniform' )
{
  numberOfEncodingLayers <- length( numberOfUnitsPerLayer ) - 1

  inputs <- layer_input( shape = numberOfUnitsPerLayer[1] )

  encoderModel <- inputs

  for( i in seq_len( numberOfEncodingLayers - 1 ) )
    {
    encoderModel <- encoderModel %>%
      layer_dense( units = numberOfUnitsPerLayer[i+1],
         activation = activation, kernel_initializer = initializer )
    }

  encoderModel <- encoderModel %>%
    layer_dense( units = tail( numberOfUnitsPerLayer, 1 ) )

  autoencoderModel <- encoderModel

  for( i in seq( from = numberOfEncodingLayers, to = 2, by = -1 ) )
    {
    autoencoderModel <- autoencoderModel %>%
      layer_dense( units = numberOfUnitsPerLayer[i],
         activation = activation, kernel_initializer = initializer )
    }

  autoencoderModel <- autoencoderModel %>%
    layer_dense( numberOfUnitsPerLayer[1], kernel_initializer = initializer )

  return( list(
    AutoencoderModel = keras_model( inputs = inputs, outputs = autoencoderModel ),
    EncoderModel = keras_model( inputs = inputs, outputs = encoderModel ) ) )
}

#' Clustering layer for Deep Embedded Clustering
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{outputs <- layer_clustering( numberOfClusters )}
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfClusters}{number of clusters.}
#'  \item{initialClusterWeights}{}
#'  \item{alpha}{parameter}
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
#' @return clustering layer
#'
#' @name ClusteringLayer
NULL

#' @export
ClusteringLayer <- R6::R6Class( "ClusteringLayer",

  inherit = KerasLayer,

  public = list(

    numberOfClusters = 10,

    initialClusterWeights = NULL,

    alpha = 1.0,

    name = '',

    initialize = function( numberOfClusters, initialClusterWeights = NULL, alpha = 1.0 )
      {
      self$numberOfClusters <- numberOfClusters
      self$initialClusterWeights <- initialClusterWeights
      self$alpha <- alpha
      self$name <- name
      },

    build = function( input_shape )
      {
      if( length( input_shape ) != 2 )
        {
        stop( paste0( "input_shape is not of length 2." ) )
        }
      self$clusters <- self$add_weight(
        self$numberOfClusters, shape = input_shape[1], initializer = 'glorot_uniform' )

      if( ! is.null( sefl$initialClusterWeights ) )
        {
        self$set_weights( self$initialClusterWeights )
        self$initialClusterWeights <- NULL
        }
      self$built <- TRUE
      },

    call = function( inputs, mask = NULL )
      {
      # Uses Student t-distribution (same as t-SNE)
      # inputs are the variable containing the data, shape = ( numberOfSamples, numberOfFeatures )

      K <- keras::backend()

      q <- 1.0 / ( 1.0 + ( K$sum( K$square(
        K$expand_dims( inputs, axis = 1 ) - self$clusters ), axis = 2 ) / self$alpha ) )
      q <- q^( ( self$alpha + 1.0 ) / 2.0 )
      q <- K$transpose( K$transpose( q ) / K$sum( q, axis = 1 ) )

      return( q )
      },

    compute_output_shape = function( input_shape )
      {
      if( length( input_shape ) != 2 )
        {
        stop( paste0( "input_shape is not of length 2." ) )
        }

      numberOfChannels <- as.integer( tail( unlist( input_shape[[1]] ), 1 ) )

      return( list( input_shape[1], self$numberOfClusters ) )
      }
  )
)

layer_clustering <- function( objects,
  numberOfClusters, initialClusterWeights, alpha, name )
{
  create_layer( ClusteringLayer, objects,
      list( numberOfClusters = numberOfClusters,
            initialClusterWeights = initialClusterWeights,
            alpha = alpha, name = '' )
      )
}

#' Deep embedded clustering (DEC) model class
#'
#' @docType class
#'
#' @section Usage:
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfUnitsPerLayer}{array describing the auteoencoder.}
#'  \item{numberOfClusters}{number of clusters.}
#'  \item{alpha}{parameter}
#'  \item{initializer}{initializer for autoencoder}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$pretrain}
#'
#'   \code{$loadWeights}
#'
#'   \code{$extractFeatures}
#'
#'   \code{$predictClusterLabels}
#'
#'   \code{$targetDistribution}
#'
#'   \code{$compile}
#'
#'   \code{$fit}
#'
#' @author Tustison NJ
#'
#' @name DeepEmbeddedClusteringModel
NULL

#' @export
DeepEmbeddedClusteringModel <- R6::R6Class( "DeepEmbeddedClusteringModel",

  # inherit = ,

  public = list(

    numberOfUnitsPerLayer = NULL,

    numberOfClusters = 10,

    alpha = 1.0,

    initializer = 'glorot_uniform',

    initialize = function( numberOfUnitsPerLayer,
      numberOfClusters, alpha = 1.0, initializer = 'glorot_uniform' )
      {
      self$numberOfUnitsPerLayer <- numberOfUnitsPerLayer
      self$numberOfClusters <- numberOfClusters
      self$alpha <- alpha
      self$initializer <- initializer

      ae <- createAutoencoderModel( self$numberOfUnitsPerLayer, self$initializer )

      self$autoencoder <- ae$AutoencoderModel
      self$encoder <- ae$EncoderModel

      clusteringLayer <- self$encoder %>%
        layer_clustering( self$numberOfClusters, name = "clustering" )

      self$model <- keras_model( inputs = self$encoder$input, outputs = clusteringLayer )
      },

    pretrain = function( x, optimizer = 'adam', epochs = 200, batchSize = 256 )
      {
      self$autoencoder$compile( optimizer = optimizer, loss = 'mse' )
      self$autoencoder$fit( x, x, batch_size = batchSize, epochs = epochs )
      },

    loadWeights = function( weights )
      {
      self$model$load_weights( weights )
      },

    extractFeatures = function( x )
      {
      self$encoder$predict( x, verbose = 0 )
      },

    predictClusterLabels = function( x )
      {
      clusterProbabilities <- self$model$predict( x, verbose = 0 )
      return( max.col( clusterProbabilities ) )
      },

    targetDistribution = function( q )
      {
      weight <- q^2 / colSums( q )
      p <- t( t( weight ) / rowSums( weight ) )
      return( p )
      },

    compile = function( optimizer = 'sgd', loss = 'kld' )
      {
      self$model$compile( optimizer = optimizer, loss = loss )
      },

    fit = function( x, maxNumberOfIterations = 2e4, batchSize = 256, tolerance = 1e-3, updateInterval = 140 )
      {

      # Initialize clusters using k-means

      km <- stats::kmeans( self$encoder$predict( x, verbose = 0 ),
        centers = self$numberOfClusters, nstart = 20 )
      currentPrediction <- fitted( km )
      previousPrediction <- currentPrediction

      self$model$get_layer( name = 'clustering' )$set_weights( as.array( km$centers ) )

      # Deep clustering

      loss <- 0
      index <- 0
      indexArray <- 1:( dim( x )[1] )

      for( i in seq_len( maxNumberOfIterations ) )
        {
        if( i %% updateInterval == 0 )
          {
          q <- self$model$predict( x, verbose = 0 )
          p <- self$targetDistribution( q )

          # Met stopping criterion

          currentPrediction <- max.col( q )
          deltaLabel <- sum( currentPrediction != previousPrediction ) / length( currentPrediction )
          previousPrediction <- currentPrediction

          if( i > 0 && deltaLabel < tolerance )
            {
            break
            }
          }

        batchIndices <- indexArray[( index * batchSize + 1 ):min( ( index + 1 ) * batchSize + 1, dim( x )[1] )]
        loss <- self$model$train_on_batch( x = x[batchIndices], y = p[batchIndices] )
        if( ( index + 1 ) * batchSize + 1 <= dim( x )[1] )
          {
          index <- index + 1
          } else {
          index <- 0
          }
        }
      return( prediction )
      }
    )
  )
