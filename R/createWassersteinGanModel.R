#' Wasserstein GAN model
#'
#' Wasserstein generative adverserial network from the paper:
#'
#'   https://arxiv.org/abs/1701.07875
#'
#' and ported from the Keras (python) implementation:
#'
#'   https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
#'
#' @docType class
#'
#' @section Usage:
#'
#' @section Arguments:
#' \describe{
#'  \item{inputImageSize}{}
#'  \item{latentDimension}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} {instantiates a new class and builds the
#'       generator and discriminator.}
#'   \code{$buildGenerator}{build generator.}
#'   \code{$buildGenerator}{build discriminator.}
#'
#' @author Tustison NJ
#'
#' @examples
#' \dontrun{
#'
#' library( keras )
#' library( ANTsRNet )
#'
#' keras::backend()$clear_session()
#'
#' # Let's use the mnist data set.
#'
#' mnist <- dataset_mnist()
#'
#' numberOfTrainingData <- length( mnist$train$y )
#'
#' inputImageSize <- c( dim( mnist$train$x[1,,] ), 1 )
#'
#' x <- array( data = mnist$train$x / 255, dim = c( numberOfTrainingData, inputImageSize ) )
#' y <- mnist$train$y
#'
#' numberOfClusters <- length( unique( mnist$train$y ) )
#'
#' # Instantiate the DCEC model
#'
#' ganModel <- WassersteinGanModel$new(
#'    inputImageSize = inputImageSize,
#'    latentDimension = 100 )
#'
#' ganModel$train( x, numberOfEpochs = 100 )
#' }
#'
#' @name WassersteinGanModel
NULL

#' @export
WassersteinGanModel <- R6::R6Class( "WassersteinGanModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    inputImageSize = c( 28, 28, 1 ),

    latentDimension = 100,

    numberOfCriticIterations = 5,

    clipValue = 0.01,

    initialize = function( inputImageSize, latentDimension = 100,
      numberOfCriticIterations = 5, clipValue = 0.01 )
      {
      self$inputImageSize <- inputImageSize
      self$latentDimension <- latentDimension
      self$numberOfCriticIterations <- numberOfCriticIterations
      self$clipValue <- clipValue

      optimizer <- optimizer_rmsprop( lr = 0.00005 )

      self$critic <- self$buildCritic()

      self$critic$compile( loss = self$wassersteinLoss,
        optimizer = optimizer, metrics = list( 'acc' ) )
      self$critic$trainable <- FALSE

      self$generator <- self$buildGenerator()

      z <- layer_input( shape = c( self$latentDimension ) )
      image <- self$generator( z )

      validity <- self$critic( image )

      self$combinedModel <- keras_model( inputs = z, outputs = validity )
      self$combinedModel$compile( loss = self$wassersteinLoss,
        optimizer = optimizer, metrics = list( 'acc' ) )
      },

    wassersteinLoss = function( y_true, y_pred )
      {
      K <- keras::backend()
      return( K$mean( y_true * y_pred ) )
      },

    buildGenerator = function()
      {
      model <- keras_model_sequential()

      for( i in seq_len( 3 ) )
        {
        numberOfUnits <- 2 ^ ( 8 + i - 1 )

        if( i == 1 )
          {
          model <- model %>% layer_dense(
            input_shape = self$latentDimension, units = numberOfUnits )
          } else {
          model <- model %>% layer_dense( units = numberOfUnits )
          }

        model <- model %>% layer_dense( units = numberOfUnits )
        model <- model %>% layer_activation_leaky_relu( alpha = 0.2 )
        model <- model %>% layer_batch_normalization( momentum = 0.8 )
        }

      model <- model %>% layer_dense(
        units = prod( self$inputImageSize ), activation = 'tanh' )
      model <- model %>% layer_reshape( target_shape = self$inputImageSize )

      noise <- layer_input( shape = c( self$latentDimension ) )
      image <- model( noise )

      generator <- keras_model( inputs = noise, outputs = image )

      return( generator )
      },

    buildDiscriminator = function()
      {
      model <- keras_model_sequential()

      model <- model %>% layer_flatten( input_shape = self$inputImageSize )
      model <- model %>% layer_dense( units = 512 )
      model <- model %>% layer_activation_leaky_relu( alpha = 0.2 )
      model <- model %>% layer_dense( units = 256 )
      model <- model %>% layer_activation_leaky_relu( alpha = 0.2 )
      model <- model %>% layer_dense( units = 1, activation = 'sigmoid' )

      image <- layer_input( shape = c( self$inputImageSize ) )

      validity <- model( image )

      discriminator <- keras_model( inputs = image, outputs = validity )

      return( discriminator )
      },

    train = function( X_train, numberOfEpochs, batchSize = 128,
      sampleInterval = NA, sampleFilePrefix = 'sample' )
      {
      valid <- array( data = 1, dim = c( batchSize, 1 ) )
      fake <- array( data = 0, dim = c( batchSize, 1 ) )

      for( epoch in seq_len( numberOfEpochs ) )
        {
        # train discriminator

        indices <- sample.int( dim( X_train )[1], batchSize )
        X_valid_batch <- X_train[indices,,,, drop = FALSE]

        noise <- array( data = rnorm( n = batchSize * self$latentDimension,
          mean = 0, sd = 1 ), dim = c( batchSize, self$latentDimension ) )
        X_fake_batch <- self$generator$predict( noise )

        dLossReal <- self$discriminator$train_on_batch( X_valid_batch, valid )
        dLossFake <- self$discriminator$train_on_batch( X_fake_batch, fake )
        dLoss <- list( 0.5 * ( dLossReal[[1]] + dLossFake[[1]] ),
                       0.5 * ( dLossReal[[2]] + dLossFake[[2]] ) )

        # train generator

        noise <- array( data = rnorm( n = batchSize * self$latentDimension,
          mean = 0, sd = 1 ), dim = c( batchSize, self$latentDimension ) )
        gLoss <- self$combinedModel$train_on_batch( noise, valid )

        cat( "Epoch ", epoch, ": [Discriminator loss: ", dLoss[[1]],
             " acc: ", dLoss[[2]], "] ", "[Generator loss: ", gLoss, "]\n",
             sep = '' )

        if( ! is.na( sampleInterval ) )
          {
          if( ( ( epoch - 1 ) %% sampleInterval ) == 0 )
            {
            noise <- array( data = rnorm( n = 1 * self$latentDimension,
                                          mean = 0, sd = 1 ),
                            dim = c( 1, self$latentDimension ) )
            X_generated <- ganModel$generator$predict( noise )

            # Convert to [0,255] to write as jpg using ANTsR

            X_generated <- 255 * ( X_generated - min( X_generated ) ) /
              ( max( X_generated ) - min( X_generated ) )
            X_generated <- drop( X_generated )
            X_generated[] <- as.integer( X_generated )

            imageFileName <- paste0( sampleFilePrefix, "_iteration" , epoch, ".jpg" )
            cat( "   --> writing sample image: ", imageFileName, "\n" )
            antsImageWrite( as.antsImage( X_generated, pixeltype = "unsigned char" ),
              imageFileName )
            }
          }
        }
      }
    )
  )
