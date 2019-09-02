#' Cycle GAN model
#'
#' Deep conviolutional generative adverserial network from the paper:
#'
#'   https://arxiv.org/pdf/1703.10593
#'
#' and ported from the Keras (python) implementation:
#'
#'   https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py
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
#' ganModel <- CycleGanModel$new(
#'    inputImageSize = inputImageSize )
#' }
#'
#' @name CycleGanModel
NULL

#' @export
CycleGanModel <- R6::R6Class( "CycleGanModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    dimensionality = 2,

    inputImageSize = c( 128, 128, 3 ),

    numberOfChannels = 3,

    lambdaCycleLossWeight = 10.0,

    lambdaIdentityLossWeight = 1.0,

    numberOfFiltersAtBaseLayer = c( 32, 64 ),

    initialize = function( inputImageSize,
      lambdaCycleLossWeight = 10.0, lambdaIdentityLossWeight = 1.0,
      numberOfFiltersAtBaseLayer = c( 32, 64 ) )
      {
      self$inputImageSize <- inputImageSize
      self$numberOfChannels <- tail( self$inputImageSize, 1 )

      self$discriminatorPatchSize <- NULL

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

      # Build discriminators for domains A and B

      self$discriminatorA <- self$buildDiscriminator()
      self$discriminatorA$compile( loss = 'mse',
        optimizer = optimizer, metrics = list( 'acc' ) )
      self$discriminatorA$trainable <- FALSE

      self$discriminatorB <- self$buildDiscriminator()
      self$discriminatorB$compile( loss = 'mse',
        optimizer = optimizer, metrics = list( 'acc' ) )
      self$discriminatorB$trainable <- FALSE

      # Build U-net like generators

      self$generatorAtoB <- self$buildGenerator()
      self$generatorBtoA <- self$buildGenerator()

      # Images

      imageA <- layer_input( shape = self$inputImageSize )
      imageB <- layer_input( shape = self$inputImageSize )

      fakeImageB <- self$generatorAtoB( imageA )
      fakeImageA <- self$generatorBtoA( imageB )

      reconstructedImageA <- self$generatorBtoA( fakeImageB )
      reconstructedImageB <- self$generatorAtoB( fakeImageA )

      identityImageA <- self$generatorBtoA( imageA )
      identityImageB <- self$generatorAtoB( imageB )

      # Check the images

      validityA <- self$discriminatorA( fakeImageA )
      validityB <- self$discriminatorB( fakeImageB )

      # Combined model

      self$combinedModel <- keras_model( inputs = list( imageA, imageB ),
        outputs = list( validityA, validityB, reconstructedImageA,
          reconstructedImageB, identityImageA, identityImageB  ) )
      self$combinedModel$compile( loss = list( 'mse', 'mse', 'mae', 'mae',
        'mae', 'mae' ), loss_weights = c( 1.0, 1.0,
          self$lambdaCycleLossWeight, self$lambdaCycleLossWeight,
          self$lambdaIdentityLossWeight, self$lambdaIdentityLossWeight ),
        optimizer = optimizer )
      },

    buildGenerator = function()
      {
      buildEncodingLayer <- function( input, numberOfFilters, kernelSize = 4 )
        {
        encoder <- input
        if( self$dimensionality == 2 )
          {
          encoder <- encoder %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          } else {
          encoder <- encoder %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          }
        encoder <- encoder %>% layer_activation_leaky_relu( alpha = 0.2 )
        encoder <- encoder %>% layer_instance_normalization()
        return( encoder )
        }

      buildDecodingLayer <- function( input, skipInput, numberOfFilters,
        kernelSize = 4, dropoutRate = 0 )
        {
        decoder <- input
        if( self$dimensionality == 2 )
          {
          decoder <- decoder %>% layer_upsampling_2d( size = 2 )
          decoder <- decoder %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = 1, padding = 'same',
            activation = 'relu' )
          } else {
          decoder <- decoder %>% layer_upsampling_3d( size = 2 )
          decoder <- decoder %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = 1, padding = 'same',
            activation = 'relu' )
          }
        if( dropoutRate > 0.0 )
          {
          decoder <- decoder %>% layer_dropout( rate = dropoutRate )
          }
        decoder <- decoder %>% layer_instance_normalization()
        decoder <- list( decoder, skipInput ) %>% layer_concatenate()
        return( decoder )
        }

       input <- layer_input( shape = self$inputImageSize )

       encodingLayers <- list()

       encodingLayers[[1]] <- buildEncodingLayer( input,
         self$numberOfFiltersAtBaseLayer[1], kernelSize = 4 )
       encodingLayers[[2]] <- buildEncodingLayer( encodingLayers[[1]],
         self$numberOfFiltersAtBaseLayer[1] * 2, kernelSize = 4 )
       encodingLayers[[3]] <- buildEncodingLayer( encodingLayers[[2]],
         self$numberOfFiltersAtBaseLayer[1] * 4, kernelSize = 4 )
       encodingLayers[[4]] <- buildEncodingLayer( encodingLayers[[3]],
         self$numberOfFiltersAtBaseLayer[1] * 8, kernelSize = 4 )

      decodingLayers <- list()
      decodingLayers[[1]] <- buildDecodingLayer( encodingLayers[[4]],
        encodingLayers[[3]], self$numberOfFiltersAtBaseLayer[1] * 4 )
      decodingLayers[[2]] <- buildDecodingLayer( decodingLayers[[1]],
        encodingLayers[[2]], self$numberOfFiltersAtBaseLayer[1] * 2 )
      decodingLayers[[3]] <- buildDecodingLayer( decodingLayers[[2]],
        encodingLayers[[1]], self$numberOfFiltersAtBaseLayer[1] )

      if( self$dimensionality == 2 )
        {
        decodingLayers[[4]] <- decodingLayers[[3]] %>%
          layer_upsampling_2d( size = 2 )
        decodingLayers[[4]] <- decodingLayers[[4]] %>%
          layer_conv_2d( self$numberOfChannels,
           kernel_size = 4, strides = 1, padding = 'same',
          activation = 'tanh' )
        } else {
        decodingLayers[[4]] <- decodingLayers[[4]] %>%
          layer_upsampling_3d( size = 2 )
        decodingLayers[[4]] <- decodingLayers[[4]] %>%
          layer_conv_3d( self$numberOfChannels,
           kernel_size = 4, strides = 1, padding = 'same',
          activation = 'tanh' )
        }

      model <- keras_model( inputs = input, outputs = decodingLayers[[4]] )

      return( model )
      },

    buildDiscriminator = function()
      {
      buildLayer <- function( input, numberOfFilters, kernelSize = 4,
        normalization = TRUE )
        {
        layer <- input
        if( self$dimensionality == 2 )
          {
          layer <- layer %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          } else {
          layer <- layer %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          }
        layer <- layer %>% layer_activation_leaky_relu( alpha = 0.2 )
        if( normalization == TRUE )
          {
          layer <- layer %>% layer_instance_normalization()
          }
        return( layer )
        }

      image <- layer_input( shape = c( self$inputImageSize ) )

      layers <- list()
      layers[[1]] <- image %>% buildLayer( self$numberOfFiltersAtBaseLayer[2],
        normalization = FALSE )
      layers[[2]] <- layers[[1]] %>%
        buildLayer( self$numberOfFiltersAtBaseLayer[2] * 2 )
      layers[[3]] <- layers[[2]] %>%
        buildLayer( self$numberOfFiltersAtBaseLayer[2] * 4 )
      layers[[4]] <- layers[[3]] %>%
        buildLayer( self$numberOfFiltersAtBaseLayer[2] * 8 )

      validity <- NA
      if( self$dimensionality == 2 )
        {
        validity <- layers[[4]] %>%
          layer_conv_2d( 1,  kernel_size = 4, strides = 1, padding = 'same')
        } else {
        validity <- layers[[4]] %>%
          layer_conv_3d( 1,  kernel_size = 4, strides = 1, padding = 'same')
        }

      if( is.null( self$discriminatorPatchSize ) )
        {
        K <- keras::backend()
        self$discriminatorPatchSize <- unlist( K$int_shape( validity ) )
        }

      discriminator <- keras_model( inputs = image, outputs = validity )

      return( discriminator )
      },

    train = function( X_trainA, X_trainB, numberOfEpochs, batchSize = 128,
      sampleInterval = NA, sampleFilePrefix = 'sample' )
      {
      valid <- array( data = 1, dim = c( batchSize, self$discriminatorPatchSize ) )
      fake <- array( data = 0, dim = c( batchSize, self$discriminatorPatchSize ) )

      for( epoch in seq_len( numberOfEpochs ) )
        {
        indicesA <- sample.int( dim( X_trainA )[1], batchSize )
        imagesA <- X_trainA[indicesA,,,, drop = FALSE]

        indicesB <- sample.int( dim( X_trainB )[1], batchSize )
        imagesB <- X_trainB[indicesB,,,, drop = FALSE]

        # train discriminator

        fakeImagesB <- self$generatorAtoB$predict( imagesA )
        fakeImagesA <- self$generatorBtoA$predict( imagesB )

        dALossReal <- self$discriminatorA$train_on_batch( imagesA, valid )
        dALossFake <- self$discriminatorA$train_on_batch( fakeImagesA, fake )

        dBLossReal <- self$discriminatorB$train_on_batch( imagesB, valid )
        dBLossFake <- self$discriminatorB$train_on_batch( fakeImagesB, fake )

        dLoss <- list()
        for( i in seq_len( length( dALossReal ) ) )
          {
          dLoss[[i]] <- 0.25 * ( dALossReal[[i]] + dALossFake[[i]] +
            dBLossReal[[i]] + dBLossFake[[i]] )
          }

        # train generator

        gLoss <- self$combinedModel$train_on_batch( list( imagesA, imagesB ),
          list( valid, valid, imagesA, imagesB, imagesA, imagesB ) )

        cat( "Length ", length( gLoss ), "\n" )

        cat( "Epoch ", epoch, ": [Discriminator loss: ", dLoss[[1]],
             " acc: ", dLoss[[2]], "] ", "[Generator loss: ", gLoss[1], ", ",
             mean( gLoss[2:4] ), ", ", mean( gLoss[4:6] ),
             mean( gLoss[6:7] ), "]\n",
             sep = '' )

        # if( self$dimensionality == 2 )
        #   {
        #   if( ! is.na( sampleInterval ) )
        #     {
        #     if( ( ( epoch - 1 ) %% sampleInterval ) == 0 )
        #       {
        #       # Do a 5x5 grid

        #       predictedBatchSize <- 5 * 5
        #       noise <- array( data = rnorm( n = predictedBatchSize * self$latentDimension,
        #                                     mean = 0, sd = 1 ),
        #                       dim = c( predictedBatchSize, self$latentDimension ) )
        #       X_generated <- ganModel$generator$predict( noise )

        #       # Convert to [0,255] to write as jpg using ANTsR

        #       X_generated <- 255 * ( X_generated - min( X_generated ) ) /
        #         ( max( X_generated ) - min( X_generated ) )
        #       X_generated <- drop( X_generated )
        #       X_generated[] <- as.integer( X_generated )

        #       X_tiled <- array( data = 0,
        #         dim = c( 5 * dim( X_generated )[2], 5 * dim( X_generated )[3] ) )
        #       for( i in 1:5 )
        #         {
        #         indices_i <- ( ( i - 1 ) * dim( X_generated )[2] + 1 ):( i * dim( X_generated )[2] )
        #         for( j in 1:5 )
        #           {
        #           indices_j <- ( ( j - 1 ) * dim( X_generated )[3] + 1 ):( j * dim( X_generated )[3] )

        #           X_tiled[indices_i, indices_j] <- X_generated[( i - 1 ) * 5 + j,,]
        #           }
        #         }

        #       sampleDir <- dirname( sampleFilePrefix )
        #       if( ! dir.exists( sampleDir ) )
        #         {
        #         dir.create( sampleDir, showWarnings = TRUE, recursive = TRUE )
        #         }

        #       imageFileName <- paste0( sampleFilePrefix, "_iteration" , epoch, ".jpg" )
        #       cat( "   --> writing sample image: ", imageFileName, "\n" )
        #       antsImageWrite( as.antsImage( t( X_tiled ), pixeltype = "unsigned char" ),
        #         imageFileName )
        #       }
        #     }
        #   }
        }
      }
    )
  )
