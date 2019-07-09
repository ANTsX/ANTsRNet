#' Vanilla GAN model interpretation
#'
#' Original generative adverserial network (GAN) model from the
#' paper:
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

    latentDimension = 100,

    initialize = function(  inputImageSize, latentDimension )
      {
      self$inputImageSize <- inputImageSize
      self$latentDimension <- latentDimension

      self$discriminator <- self$buildDiscriminator()
      self$discriminator$compile( loss = 'binary_crossentropy',
        optimizer = optimizer_adam( lr = 0.0001 ), metrics = list( 'acc' ) )
      self$discriminator$trainable <- FALSE

      self$generator <- self$buildGenerator()

      z <- layer_input( shape = c( self$latentDimension ) )
      image <- self$generator( z )

      validity <- self$discriminator( image )

      self$combinedModel <- keras_model( inputs = z, outputs = validity )
      self$combinedModel$compile( loss = 'binary_crossentropy',
      optimizer = optimizer_adam( lr = 0.0001 ) )
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

      model <- model %>% layer_dense( units = prod( self$inputImageSize ) )
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
      model <- model %>% layer_dense( units = 1, activation = 'sigmoid' )

      image <- layer_input( shape = c( self$inputImageSize ) )

      validity <- model( image )

      discriminator <- keras_model( inputs = image, outputs = validity )

      return( discriminator )
      },

    train = function( X_train, numberOfEpochs, batchSize = 128 )
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
        dLoss <- list( 0.5 * ( dLossReal[[1]] + dLossFake[[1]] ), 0.5 * ( dLossReal[[2]] + dLossFake[[2]] ) )

        # train generator

        noise <- array( data = rnorm( n = batchSize * self$latentDimension,
          mean = 0, sd = 1 ), dim = c( batchSize, self$latentDimension ) )
        gLoss <- self$combinedModel$train_on_batch( noise, valid )

        cat( "Epoch ", epoch, ": [Discriminator loss: ", dLoss[[1]], " acc: ", dLoss[[2]], "] ",
          "[Generator loss: ", gLoss, "]\n", sep = '' )
        }
      }
    )
  )
