#' Vanilla GAN model
#'
#' Original generative adverserial network from the paper:
#'
#'   https://arxiv.org/abs/1406.2661
#'
#' and ported from the Keras (python) implementation:
#'
#'   https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
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
#' ganModel <- VanillaGanModel$new(
#'    inputImageSize = inputImageSize,
#'    latentDimension = 100 )
#'
#' ganModel$train( x, numberOfEpochs = 100 )
#' }
#'
#' @name VanillaGanModel
NULL

#' @export
VanillaGanModel <- R6::R6Class( "VanillaGanModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    inputImageSize = c( 28, 28, 1 ),

    dimensionality = 2,

    latentDimension = 100,

    initialize = function( inputImageSize, latentDimension = 100 )
      {
      self$inputImageSize <- inputImageSize
      self$latentDimension <- latentDimension

      self$dimensionality <- NA
      if( length( self$inputImageSize ) == 3 )
        {
        self$dimensionality <- 2
        } else if( length( self$inputImageSize ) == 4 ) {
        self$dimensionality <- 3
        } else {
        stop( "Incorrect size for inputImageSize.\n" )
        }

      optimizer <- optimizer_adam( lr = 0.0002, beta_1 = 0.5 )

      self$discriminator <- self$buildDiscriminator()

      self$discriminator$compile( loss = 'binary_crossentropy',
        optimizer = optimizer, metrics = list( 'acc' ) )
      self$discriminator$trainable <- FALSE

      self$generator <- self$buildGenerator()

      z <- layer_input( shape = c( self$latentDimension ) )
      image <- self$generator( z )

      validity <- self$discriminator( image )

      self$combinedModel <- keras_model( inputs = z, outputs = validity )
      self$combinedModel$compile( loss = 'binary_crossentropy',
        optimizer = optimizer )
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

        X_valid_batch <- NULL
        if( self$dimensionality == 2 )
          {
          X_valid_batch <- X_train[indices,,,, drop = FALSE]
          } else {
          X_valid_batch <- X_train[indices,,,,, drop = FALSE]
          }

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

        if( self$dimensionality == 2 )
          {
          if( ! is.na( sampleInterval ) )
            {
            if( ( ( epoch - 1 ) %% sampleInterval ) == 0 )
              {
              # Do a 5x5 grid

              predictedBatchSize <- 5 * 5
              noise <- array( data = rnorm( n = predictedBatchSize * self$latentDimension,
                                            mean = 0, sd = 1 ),
                              dim = c( predictedBatchSize, self$latentDimension ) )
              X_generated <- self$generator$predict( noise )

              # Convert to [0,255] to write as jpg using ANTsR

              X_generated <- 255 * ( X_generated - min( X_generated ) ) /
                ( max( X_generated ) - min( X_generated ) )
              X_generated <- drop( X_generated )
              X_generated[] <- as.integer( X_generated )

              X_tiled <- array( data = 0,
                dim = c( 5 * dim( X_generated )[2], 5 * dim( X_generated )[3] ) )
              for( i in 1:5 )
                {
                indices_i <- ( ( i - 1 ) * dim( X_generated )[2] + 1 ):( i * dim( X_generated )[2] )
                for( j in 1:5 )
                  {
                  indices_j <- ( ( j - 1 ) * dim( X_generated )[3] + 1 ):( j * dim( X_generated )[3] )

                  X_tiled[indices_i, indices_j] <- X_generated[( i - 1 ) * 5 + j,,]
                  }
                }

              sampleDir <- dirname( sampleFilePrefix )
              if( ! dir.exists( sampleDir ) )
                {
                dir.create( sampleDir, showWarnings = TRUE, recursive = TRUE )
                }

              imageFileName <- paste0( sampleFilePrefix, "_iteration" , epoch, ".jpg" )
              cat( "   --> writing sample image: ", imageFileName, "\n" )
              antsImageWrite( as.antsImage( t( X_tiled ), pixeltype = "unsigned char" ),
                imageFileName )
              }
            }
          }
        }
      }
    )
  )
