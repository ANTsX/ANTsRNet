#' Improved Wasserstein GAN model
#'
#' Improved Wasserstein generative adverserial network (with
#' gradient penalty) from the paper:
#'
#'   https://arxiv.org/abs/1704.00028
#'
#' and ported from the Keras (python) implementation:
#'
#'   https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
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
#'       generator and critic.}
#'   \code{$buildGenerator}{build generator.}
#'   \code{$buildGenerator}{build critic.}
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
#' # Instantiate the WGAN model
#'
#' ganModel <- ImprovedWassersteinGanModel$new(
#'    inputImageSize = inputImageSize,
#'    latentDimension = 100 )
#'
#' ganModel$train( x, numberOfEpochs = 100 )
#' }
#'
#' @name ImprovedWassersteinGanModel
NULL

#' @export
ImprovedWassersteinGanModel <- R6::R6Class( "ImprovedWassersteinGanModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    inputImageSize = c( 28, 28, 1 ),

    dimensionality = 2,

    latentDimension = 100,

    numberOfCriticIterations = 5,

    initialize = function( inputImageSize, latentDimension = 100,
      numberOfCriticIterations = 5, clipValue = 0.01 )
      {
      self$inputImageSize <- inputImageSize
      self$latentDimension <- latentDimension
      self$numberOfCriticIterations <- numberOfCriticIterations

      self$dimensionality <- NA
      if( length( self$inputImageSize ) == 3 )
        {
        self$dimensionality <- 2
        } else if( length( self$inputImageSize ) == 4 ) {
        self$dimensionality <- 3
        } else {
        stop( "Incorrect size for inputImageSize.\n" )
        }

      optimizer <- optimizer_rmsprop( lr = 0.00005 )

      self$generator <- self$buildGenerator()
      self$critic <- self$buildCritic()

      self$generator$trainable <- FALSE

      realImage <- layer_input( shape = self$inputImageSize )

      criticNoise <- layer_input( shape = c( self$latentDimension ) )
      fakeImage <- self$generator( criticNoise )

      fakeValidity <- self$critic( fakeImage )
      realValidity <- self$critic( realImage )

      interpolatedImage <- list( fakeImage, realImage ) %>%
        layer_lambda(
          f = function( X )
            {
            K <- keras::backend()
            inputShape <- K$int_shape( X[[1]] )
            batchSize <- inputShape[[1]]
            if( self$dimensionality == 2 )
              {
              alpha <- K$random_uniform( c( batchSize, 1L, 1L, 1L ) )
              } else {
              alpha <- K$random_uniform( c( batchSize, 1L, 1L, 1L, 1L ) )
              }
            return( alpha * X[[1]] + ( 1.0 - alpha ) * X[[2]] )
            }
          )

      interpolatedValidity <- self$critic( interpolatedImage )

      partialGradientPenaltyLoss <- custom_metric( "partialGradientPenaltyLoss",
        function( y_true, y_pred )
          {
          self$gradientPenaltyLoss( y_true, y_pred,
            averagedSamples = interpolatedImage )
          }
        )

      # Construct the critic model

      self$criticModel <- keras_model( inputs = list( realImage, criticNoise ),
        outputs = list( realValidity, fakeValidity, interpolatedValidity ) )
      self$criticModel$compile( loss = list(
        self$wassersteinLoss, self$wassersteinLoss, partialGradientPenaltyLoss ),
        optimizer = optimizer, loss_weights = list( 1, 1, 10 ) )

      # Freeze the critic's layers for the generator model
      self$critic$trainable <- FALSE
      self$generator$trainable <- TRUE

      noise <- layer_input( shape = c( self$latentDimension ) )
      image <- self$generator( noise )
      validity <- self$critic( image )

      self$generatorModel <- keras_model( inputs = noise, outputs = validity )
      self$generatorModel$compile( loss = self$wassersteinLoss,
        optimizer = optimizer )
      },

    wassersteinLoss = function( y_true, y_pred )
      {
      # https://github.com/keras-team/keras-contrib/issues/280

      K <- keras::backend()
      return( K$mean( y_true * y_pred ) )
      },

    gradientPenaltyLoss = function( y_true, y_pred, averagedSamples )
      {
      K <- keras::backend()
      gradients <- K$gradients( y_pred, averagedSamples )[1]
      gradientsSquared <- K$square( gradients )
      gradientsSquaredSum <- K$sum( gradientsSquared,
        axis = seq( 2, length( gradientsSquared$shape ) ) )
      gradientsL2Norm <- K$sqrt( gradientsSquaredSum )
      gradientPenalty <- K$square( 1.0 - gradientsL2Norm )
      return( K$mean( gradientPenalty ) )
      },

    buildGenerator = function( numberOfFiltersPerLayer = c( 128, 64 ),
      kernelSize = 4 )
      {
      model <- keras_model_sequential()

      # To build the generator, we create the reverse encoder model
      # and simply build the reverse model

      encoder <- NA
      if( self$dimensionality == 2 )
        {
        aeModel <- createConvolutionalAutoencoderModel2D(
          inputImageSize = self$inputImageSize,
          numberOfFiltersPerLayer =
            c( rev( numberOfFiltersPerLayer ), self$latentDimension ),
          convolutionKernelSize = c( 5, 5 ),
          deconvolutionKernelSize = c( 5, 5 ) )
        encoder <- aeModel$ConvolutionalEncoderModel
        } else {
        aeModel <- createConvolutionalAutoencoderModel3D(
          inputImageSize = self$inputImageSize,
            numberOfFiltersPerLayer =
              c( rev( numberOfFiltersPerLayer ), self$latentDimension ),
          convolutionKernelSize = c( 5, 5, 5 ),
          deconvolutionKernelSize = c( 5, 5, 5 ) )
        encoder <- aeModel$ConvolutionalEncoderModel
        }

      encoderLayers <- encoder$layers

      penultimateLayer <- encoderLayers[[length( encoderLayers ) - 1]]

      model <- model %>% layer_dense( units = penultimateLayer$output_shape[[2]],
        input_shape = c( self$latentDimension ), activation = "relu" )
      convLayer <- encoderLayers[[length( encoderLayers ) - 2]]
      resampledSize <- convLayer$output_shape
      model <- model %>% layer_reshape( unlist( resampledSize ) )

      count <- 1
      for( i in seq( from = length( encoderLayers ) - 2, to = 2, by = -1 ) )
        {
        convLayer <- encoderLayers[[i]]
        resampledSize <- unlist( convLayer$output_shape )[1:self$dimensionality]

        if( self$dimensionality == 2 )
          {
          model <- model %>% layer_resample_tensor_2d( shape = resampledSize,
            interpolationType = 'linear' )
          model <- model %>% layer_conv_2d(
            filters = numberOfFiltersPerLayer[count], kernel_size = kernelSize,
            padding = 'same' )
          } else {
          model <- model %>% layer_resample_tensor_3d( shape = resampledSize,
            interpolationType = 'linear' )
          model <- model %>% layer_conv_3d(
            filters = numberOfFiltersPerLayer[count], kernel_size = kernelSize )
          }
        model <- model %>% layer_batch_normalization( momentum = 0.8 )
        model <- model %>% layer_activation( "relu" )
        count <- count + 1
        }

      numberOfChannels <- tail( self$inputImageSize, 1 )
      if( self$dimensionality == 2 )
        {
        model <- model %>% layer_resample_tensor_2d(
          shape = as.integer( self$inputImageSize[1:self$dimensionality] ),
          interpolationType = 'linear' )
        model <- model %>% layer_conv_2d( filters = numberOfChannels,
          kernel_size = kernelSize, padding = 'same' )
        } else {
        model <- model %>% layer_resample_tensor_3d(
          shape = as.integer( self$inputImageSize[1:self$dimensionality] ),
          interpolationType = 'linear' )
        model <- model %>% layer_conv_3d( filters = numberOfChannels,
          kernel_size = kernelSize, padding = 'same' )
        }
      model <- model %>% layer_activation( "tanh" )

      noise <- layer_input( shape = c( self$latentDimension ) )
      image <- model( noise )

      generator <- keras_model( inputs = noise, outputs = image )

      return( generator )
      },

    buildCritic = function( numberOfFiltersPerLayer = c( 16, 32, 64, 128 ),
       kernelSize = 3, dropoutRate = 0.25 )
      {
      model <- keras_model_sequential()

      for( i in seq_len( length( numberOfFiltersPerLayer ) ) )
        {
        strides = 2
        if( i == length( numberOfFiltersPerLayer ) )
          {
          strides = 1
          }
        if( self$dimensionality == 2 )
          {
          model <- model %>% layer_conv_2d( input_shape = self$inputImageSize,
            filters = numberOfFiltersPerLayer[i], kernel_size = kernelSize,
            strides = strides, padding = 'same' )
          } else {
          model <- model %>% layer_conv_3d( input_shape = self$inputImageSize,
            filters = numberOfFiltersPerLayer[i], kernel_size = kernelSize,
            strides = strides, padding = 'same' )
          }
        if( i > 1 )
          {
          model <- model %>% layer_batch_normalization( momentum = 0.8 )
          }
        model <- model %>% layer_activation_leaky_relu( alpha = 0.2 )
        model <- model %>% layer_dropout( rate = dropoutRate )
        }

      model <- model %>% layer_flatten()
      model <- model %>% layer_dense( units = 1 )

      image <- layer_input( shape = c( self$inputImageSize ) )

      validity <- model( image )

      critic <- keras_model( inputs = image, outputs = validity )

      return( critic )
      },

    train = function( X_train, numberOfEpochs, batchSize = 128,
      sampleInterval = NA, sampleFilePrefix = 'sample' )
      {
      valid <- array( data = -1, dim = c( batchSize, 1 ) )
      fake <- array( data = 1, dim = c( batchSize, 1 ) )
      dummy <- array( data = 0, dim = c( batchSize, 1 ) )

      for( epoch in seq_len( numberOfEpochs ) )
        {

        # Train critic

        for( c in seq_len( self$numberOfCriticIterations ) )
          {
          indices <- sample.int( dim( X_train )[1], batchSize )
          X_valid_batch <- X_train[indices,,,, drop = FALSE]

          noise <- array( data = rnorm( n = batchSize * self$latentDimension,
            mean = 0, sd = 1 ), dim = c( batchSize, self$latentDimension ) )

          dLoss <- self$criticModel$train_on_batch(
            list( X_valid_batch, noise ), list( valid, fake, dummy ) )
          }

        # Train generator

        gLoss <- self$generatorModel$train_on_batch( noise, valid )

        cat( "Epoch ", epoch, ": [Critic loss: ", dLoss[[1]],
             "] [Generator loss: ", gLoss, "]\n",
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
              X_generated <- ganModel$generator$predict( noise )

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
