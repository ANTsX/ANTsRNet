
#' Clustering layer for Deep Embedded Clustering
#'
#' @docType class
#'
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfClusters}{number of clusters.}
#'  \item{initialClusterWeights}{}
#'  \item{alpha}{parameter}
#'  \item{alpha}{name}
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
#' @examples
#' model = ClusteringLayer$new(numberOfClusters = 2)
#' \dontrun{
#' model$build(c(20, 20))
#' }
#' @name ClusteringLayer
NULL

#' @export
ClusteringLayer <- R6::R6Class( "ClusteringLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    numberOfClusters = 10L,

    initialClusterWeights = NULL,

    alpha = 1.0,

    name = '',

    initialize = function( numberOfClusters,
      initialClusterWeights = NULL, alpha = 1.0, name = '' )
      {
      self$numberOfClusters <- as.integer( numberOfClusters )
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
        shape = list( self$numberOfClusters, input_shape[[2]] ),
        initializer = keras::initializer_glorot_uniform(),
        # initializer = 'glorot_uniform',
        name = 'clusters' )

      if( ! is.null( self$initialClusterWeights ) )
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
        K$expand_dims( inputs, axis = 1L ) - self$clusters ), axis = 2L ) / self$alpha ) )
      q <- q^( ( self$alpha + 1.0 ) / 2.0 )
      q <- K$transpose( K$transpose( q ) / K$sum( q, axis = 1L ) )
      return( q )
      },

    compute_output_shape = function( input_shape )
      {
      return( list( input_shape[[1]], self$numberOfClusters ) )
      }
  )
)

layer_clustering <- function( object,
  numberOfClusters, initialClusterWeights = NULL,
  alpha = 1.0, name = '' )
{
  create_layer( ClusteringLayer, object,
      list( numberOfClusters = numberOfClusters,
            initialClusterWeights = initialClusterWeights,
            alpha = alpha, name = name )
      )
}

#' Deep embedded clustering (DEC) model class
#'
#' @docType class
#'  
#' \url{https://github.com/XifengGuo/DEC-keras}
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
#' @examples
#' \dontrun{
#'
#' library( keras )
#' library( ANTsRNet )
#'
#' fmnist <- dataset_fashion_mnist()
#'
#' numberOfTrainingData <- length( fmnist$train$y )
#' numberOfTestingData <- length( fmnist$test$y )
#'
#' numberOfPixels <- prod( dim( fmnist$test$x[1,,] ) )
#'
#' fmnist$train$xreshaped <- array_reshape( fmnist$train$x,
#'   dim = c( numberOfTrainingData, numberOfPixels ), order = "C" )
#' fmnist$test$xreshaped <- array_reshape( fmnist$train$x,
#'   dim = c( numberOfTrainingData, numberOfPixels ), order = "C" )
#'
#' x <- rbind( fmnist$test$xreshaped, fmnist$train$xreshaped ) / 255
#' y <- c( fmnist$test$y, fmnist$train$y )
#'
#' numberOfClusters <- length( unique( fmnist$train$y ) )
#'
#' initializer <- initializer_variance_scaling(
#'   scale = 1/3, mode = 'fan_in', distribution = 'uniform' )
#' pretrainOptimizer <- optimizer_sgd( lr = 1.0, momentum = 0.9 )
#'
#' decModel <- DeepEmbeddedClusteringModel$new(
#'    numberOfUnitsPerLayer = c( numberOfPixels, 500, 500, 2000, 10 ),
#'    numberOfClusters = numberOfClusters, initializer = initializer )
#'
#' modelWeightsFile <- "decAutoencoderModelWeights.h5"
#' if( ! file.exists( modelWeightsFile ) )
#'   {
#'   decModel$pretrain( x = x, optimizer = optimizer_sgd( lr = 1.0, momentum = 0.9 ),
#'     epochs = 300L, batchSize = 256L )
#'   save_model_weights_hdf5( decModel$autoencoder, modelWeightsFile )
#'   } else {
#'   load_model_weights_hdf5( decModel$autoencoder, modelWeightsFile )
#'   }
#'
#' decModel$compile( optimizer = optimizer_sgd( lr = 1.0, momentum = 0.9 ), loss = 'kld' )
#'
#' yPredicted <- decModel$fit( x, maxNumberOfIterations = 2e4, batchSize = 256,
#'   tolerance = 1e-3, updateInterval = 10 )
#' }
#'
#' @name DeepEmbeddedClusteringModel
NULL

#' @export
DeepEmbeddedClusteringModel <- R6::R6Class( "DeepEmbeddedClusteringModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    numberOfUnitsPerLayer = NULL,

    numberOfClusters = 10,

    alpha = 1.0,

    initializer = 'glorot_uniform',

    convolutional = FALSE,

    inputImageSize = NULL,

    initialize = function( numberOfUnitsPerLayer,
      numberOfClusters, alpha = 1.0, initializer = 'glorot_uniform',
      convolutional = FALSE, inputImageSize = NULL )
      {

      self$numberOfUnitsPerLayer <- as.integer( numberOfUnitsPerLayer )
      self$numberOfClusters <- as.integer( numberOfClusters )
      self$alpha <- alpha
      self$initializer <- initializer
      self$convolutional <- convolutional
      self$inputImageSize <- as.integer( inputImageSize )

      if( self$convolutional == TRUE )
        {
        if( is.null( self$inputImageSize ) )
          {
          stop( "Need to specify the input image size for CNN." )
          }
        if( length( self$inputImageSize ) == 3 )  # 2-D
          {
          ae <- createConvolutionalAutoencoderModel2D(
            inputImageSize = self$inputImageSize,
            numberOfFiltersPerLayer = self$numberOfUnitsPerLayer )
          } else {
          ae <- createConvolutionalAutoencoderModel3D(
            inputImageSize = self$inputImageSize,
            numberOfFiltersPerLayer = self$numberOfUnitsPerLayer )
          }

        self$autoencoder <- ae$convolutionalAutoencoderModel
        self$encoder <- ae$convolutionalEncoderModel

        } else {
        ae <- createAutoencoderModel( self$numberOfUnitsPerLayer,
          initializer = self$initializer )
        self$autoencoder <- ae$autoencoderModel
        self$encoder <- ae$encoderModel
        }

      clusteringLayer <- self$encoder$output %>%
        layer_clustering( self$numberOfClusters, name = "clustering" )

      if( self$convolutional == TRUE )
        {
        self$model <- keras_model( inputs = self$encoder$input, outputs = list( clusteringLayer, self$autoencoder$output ) )
        } else {
        self$model <- keras_model( inputs = self$encoder$input, outputs = clusteringLayer )
        }
      },

    pretrain = function( x, optimizer = 'adam', epochs = 200L, batchSize = 256L )
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

    compile = function( optimizer = 'sgd', loss = 'kld', lossWeights = NULL )
      {
      self$model$compile( optimizer = optimizer, loss = loss, loss_weights = lossWeights )
      },

    fit = function( x, maxNumberOfIterations = 2e4, batchSize = 256, tolerance = 1e-3, updateInterval = 140 )
      {

      # Initialize clusters using k-means

      km <- stats::kmeans( self$encoder$predict( x, verbose = 0 ),
        centers = self$numberOfClusters, nstart = 20 )
      currentPrediction <- fitted( km )
      previousPrediction <- currentPrediction

      self$model$get_layer( name = 'clustering' )$set_weights( list( km$centers ) )

      # Deep clustering

      loss <- 100000000
      index <- 0
      indexArray <- 1:( dim( x )[1] )

      for( i in seq_len( maxNumberOfIterations ) )
        {
        if( i %% updateInterval == 1 )
          {
          if( self$convolutional == TRUE )
            {
            q <- self$model$predict( x, verbose = 0 )[[1]]
            } else {
            q <- self$model$predict( x, verbose = 0 )
            }
          p <- self$targetDistribution( q )

          # Met stopping criterion

          currentPrediction <- max.col( q )
          deltaLabel <- sum( currentPrediction != previousPrediction ) / length( currentPrediction )
          previousPrediction <- currentPrediction

          cat( "Iteration", i, ": ( out of", maxNumberOfIterations,
            "): loss = [", unlist( loss ), "], deltaLabel =", deltaLabel, "\n", sep = ' ' )

          if( i > 1 && deltaLabel < tolerance )
            {
            break
            }
          }

        batchIndices <- indexArray[( index * batchSize + 1 ):min( ( index + 1 ) * batchSize, dim( x )[1] )]

        if( self$convolutional == TRUE )
          {
          if( length( self$inputImageSize ) == 3 )  # 2-D
            {
            loss <- self$model$train_on_batch( x = x[batchIndices,,,, drop = FALSE], 
                                               y = list( p[batchIndices,], x[batchIndices,,,, drop = FALSE] ) )
            } else {
            loss <- self$model$train_on_batch( x = x[batchIndices,,,,, drop = FALSE], 
                                               y = list( p[batchIndices,], x[batchIndices,,,,, drop = FALSE] ) )
            }
          } else {
          loss <- self$model$train_on_batch( x = x[batchIndices,], y = p[batchIndices,] )
          }

        if( ( index + 1 ) * batchSize + 1 <= dim( x )[1] )
          {
          index <- index + 1
          } else {
          index <- 0
          }
        }
      return( currentPrediction )
      }
    )
  )
