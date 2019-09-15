#' Super resolution GAN model
#'
#' Super resolution generative adverserial network from the paper:
#'
#'   https://arxiv.org/abs/1609.04802
#'
#' and ported from the Keras (python) implementation:
#'
#'   https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py
#'
#' @docType class
#'
#' @section Usage:
#'
#' @section Arguments:
#' \describe{
#'  \item{inputImageSize}{}
#'  \item{numberOfResidualBlocks}{}
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
#' ganModel <- SuperResolutionGanModel$new(
#'    inputImageSize = inputImageSize )
#' }
#'
#' @name SuperResolutionGanModel
NULL

#' @export
SuperResolutionGanModel <- R6::R6Class( "SuperResolutionGanModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    dimensionality = 2,

    inputImageSize = c( 128, 128, 3 ),

    numberOfChannels = 3,

    numberOfResidualBlocks = 16,

    numberOfFiltersAtBaseLayer = c( 64, 64 ),

    scaleFactor = 2,

    useImageNetWeights = TRUE,

    initialize = function( inputImageSize,
      numberOfResidualBlocks = 16, numberOfFiltersAtBaseLayer = c( 64, 64 ),
      scaleFactor = 2, useImageNetWeights = TRUE )
      {
      self$inputImageSize <- inputImageSize
      self$numberOfChannels <- tail( self$inputImageSize, 1 )
      self$numberOfResidualBlocks <- numberOfResidualBlocks
      self$numberOfFiltersAtBaseLayer <- numberOfFiltersAtBaseLayer
      self$useImageNetWeights <- useImageNetWeights

      self$scaleFactor <- scaleFactor
      if( ! scaleFactor %in% c( 1, 2, 4, 8 ) )
        {
        stop( "Error:  scale factor must be one of 1, 2, 4, or 8." )
        }

      self$dimensionality <- NA
      if( length( self$inputImageSize ) == 3 )
        {
        self$dimensionality <- 2
        } else if( length( self$inputImageSize ) == 4 ) {
        self$dimensionality <- 3
        if( self$useImageNetWeights == TRUE )
          {
          self$useImageNetWeights <- FALSE
          warning( "Warning: imageNet weights are unavailable for 3D." )
          }
        } else {
        stop( "Incorrect size for inputImageSize.\n" )
        }

      optimizer <- optimizer_adam( lr = 0.0002, beta_1 = 0.5 )

      # Build discriminator

      self$discriminator <- self$buildDiscriminator(
        numberOfFiltersAtBaseLayer = self$numberOfFiltersAtBaseLayer[1] )
      self$discriminator$compile( loss = 'mse',
        optimizer = optimizer, metrics = list( 'acc' ) )

      # Build generator

      self$generator <- self$buildGenerator()

      # Images

      highResolutionShape <- c( as.integer( self$scaleFactor ) *
        self$inputImageSize[1:self$dimensionality], self$numberOfChannels )

      imageHighResolution <- layer_input( shape = highResholutionShape )
      imageLowResolution <- layer_input( shape = self$inputImageSize )

      fakeImageHighResolution <- self$generator$predict( imageLowResolution )

      # Vgg

      vggTmp <- NULL
      if( self$dimensionality == 2 )
        {
        vggTmp <- createVggModel2D( highResolutionShape, style = '19' )
        if( self$useImageNetWeights == TRUE )
          {
          kerasVgg <- application_vgg19( input_shape = highResolutionShape,
            weights = 'imagenet' )
          vggTmp$load_weights( kerasVgg$get_weights() )
          }
        } else {
        vggTmp <- createVggModel3D( highResolutionShape, style = '19' )
        }
      self$vggModel <- keras_model( inputs = vggTmp$input,
        outputs = vggTmp$layers[[20]]$output )

      self$discriminatorPatchSize <- c(
        unlist( vggTmp$layers[[20]]$output_shape )[1:self$dimensionality], 1 )

      # Discriminator

      self$discriminator$trainable <- FALSE

      validity <- self$discriminator( fakeImageHighResolution )

      # Combined model

      if( self$useImageNetWeights )
        {
        fakeFeatures <- self$vggModel( fakeImageHighResolution )
        self$combinedModel = keras_model( inputs = list( imageHighResolution, imageLowResolution ),
                                          outputs = list( validity, fakeFeatures ) )
        self$combinedModel$compile( loss = list( 'binary_crossentropy', 'mse' ),
          loss_weights = list( 1e-3, 1 ), optimizer = optimizer )
        } else {
        self$combinedModel = keras_model( inputs = list( imageHighResolution, imageLowResolution ),
                                          outputs = validity )
        self$combinedModel$compile( loss = list( 'binary_crossentropy' ),
          optimizer = optimizer )
        }
      },

    buildGenerator = function( numberOfFilters = 64 )
      {
      buildResidualBlock <- function( input, numberOfFilters, kernelSize = 3 )
        {
        shortcut <- input

        if( self$dimensionality == 2 )
          {
          model <- model %>% layer_conv_2d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          } else {
          model <- model %>% layer_conv_3d( filters = numberOfFiltersIn,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          }
        model <- model %>% layer_activation_relu()
        model <- model %>% layer_batch_normalization( momentum = 0.8 )
        if( self$dimensionality == 2 )
          {
          model <- model %>% layer_conv_2d( filters = numberOfFiltersIn,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          } else {
          model <- model %>% layer_conv_3d( filters = numberOfFiltersIn,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          }
        model <- model %>% layer_batch_normalization( momentum = 0.8 )
        model <- list( model, shortcut ) %>% layer_add()
        return( model )
        }

      buildDeconvolutionLayer <- function( input, numberOfFilters = 256 )
        {
        model <- input
        if( self$dimensionality == 2 )
          {
          model <- model %>% layer_upsampling_2d( size = 2 )
          model <- model %>% layer_conv_2d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          } else {
          model <- model %>% layer_upsampling_3d( size = 2 )
          model <- model %>% layer_conv_3d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          }
        model <- model %>% layer_activation_relu()
        return( model )
        }

      image <- layer_input( shape = inputImageSize )

      preResidual <- image
      if( self$dimensionality == 2 )
        {
        preResidual <- preResidual %>% layer_conv_2d( filters = numberOfFilters,
          kernel_size = 9, strides = 1, padding = 'same' )
        } else {
        preResidual <- preResidual %>% layer_conv_3d( filters = numberOfFilters,
          kernel_size = 9, strides = 1, padding = 'same' )
        }
      preResidual <- preResidual %>% layer_activation_relu()

      residuals <- preResidual %>% buildResidualBlock(
        numberOfFilters = self$numberOfFiltersAtBaseLayer[1] )
      for( i in seq_len( self$numberOfResidualBlocks - 1 ) )
        {
        residuals <- residuals %>% buildResidualBlock(
          numberOfFilters = self$numberOfFiltersAtBaseLayer[1] )
        }

      postResidual <- residuals
      if( self$dimensionality == 2 )
        {
        postResidual <- postResidual %>% layer_conv_2d( filters = numberOfFilters,
          kernel_size = 3, strides = 1, padding = 'same' )
        } else {
        postResidual <- postResidual %>% layer_conv_3d( filters = numberOfFilters,
          kernel_size = 3, strides = 1, padding = 'same' )
        }
      postResidual <- postResidual %>% layer_batch_normalization( momentum = 0.8 )
      model <- list( postResidual, preResidual ) %>% layer_add()

      # upsampling

      if( self$scalingFactor >= 2 )
        {
        model <- buildDeconvolutionLayer( model )
        }
      if( self$scalingFactor >= 4 )
        {
        model <- buildDeconvolutionLayer( model )
        }
      if( self$scalingFactor == 8 )
        {
        model <- buildDeconvolutionLayer( model )
        }

      if( self$dimensionality == 2 )
        {
        model <- model %>% layer_conv_2d( filters = self$numberOfChannels,
          kernel_size = 9, strides = 1, padding = 'same',
          activation = 'tanh' )
        } else {
        postResidual <- model %>% layer_conv_3d( filters = self$numberOfChannels,
          kernel_size = 9, strides = 1, padding = 'same',
          activation = 'tanh' )
        }

      generator <- keras_model( inputs = image, outputs = model )

      return( generator )
      },

    buildDiscriminator = function()
      {
      buildLayer <- function( input, numberOfFilters, strides = 1,
        kernelSize = 3, normalization = TRUE )
        {
        layer <- input
        if( self$dimensionality == 2 )
          {
          layer <- layer %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = strides, padding = 'same' )
          } else {
          layer <- layer %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = strides, padding = 'same' )
          }
        layer <- layer %>% layer_activation_leaky_relu( alpha = 0.2 )
        if( normalization == TRUE )
          {
          layer <- layer %>% layer_batch_normalization( momentum = 0.8 )
          }
        return( layer )
        }

      image <- layer_input( shape = c( self$inputImageSize ) )

      model <- image %>% buildLayer( self$numberOfFiltersAtBaseLayer[2],
        normalization = FALSE )
      model <- image %>% buildLayer( self$numberOfFiltersAtBaseLayer[2],
        strides = 2 )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2] * 2 )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2] * 2,
        strides = 2 )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2] * 4 )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2] * 4,
        strides = 2 )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2] * 8 )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2] * 8,
        strides = 2 )

      model <- model %>%
        layer_dense( units = self$numberOfFiltersAtBaseLayer[2] * 16 )
      model <- model %>% layer_activation_leaky_relu( alpha = 0.2 )
      validity <- model %>% layer_dense( units = 1, activation = 'sigmoid' )

      discriminator <- keras_model( inputs = image, outputs = validity )

      return( discriminator )
      },

    train = function( X_trainLowResolution, X_trainHighResolution, numberOfEpochs,
      batchSize = 128, sampleInterval = NA, sampleFilePrefix = 'sample' )
      {

      valid <- NULL
      fake <- NULL
      if( self$useImageNetWeights )
        {
        valid <- array( data = 1, dim = c( batchSize, self$discriminatorPatchSize ) )
        fake <- array( data = 0, dim = c( batchSize, self$discriminatorPatchSize ) )
        } else {
        valid <- array( data = 1, dim = c( batchSize, 1 ) )
        fake <- array( data = 0, dim = c( batchSize, 1 ) )
        }

      for( epoch in seq_len( numberOfEpochs ) )
        {
        indices <- sample.int( dim( X_trainLowResolution )[1], batchSize )

        imagesLowResolution <- NULL
        imagesHighResolution <- NULL
        if( self$dimensionality == 2 )
          {
          imagesLowResolution <- X_trainLowResolution[indices,,,, drop = FALSE]
          imagesHighResolution <- X_trainHighResolution[indices,,,, drop = FALSE]
          } else {
          imagesLowResolution <- X_trainLowResolution[indices,,,,, drop = FALSE]
          imagesHighResolution <- X_trainHighResolution[indices,,,,, drop = FALSE]
          }

        # train discriminator

        fakeImagesHighResolution <- self$generator$predict( imagesLowResolution )

        dLossReal <- self$discriminator$train_on_batch( imagesHighResolution, valid )
        dLossFake <- self$discriminator$train_on_batch( fakeImagesHighResolution, fake )

        dLoss <- list()
        for( i in seq_len( length( dLossReal ) ) )
          {
          dLoss[[i]] <- 0.5 * ( dLossReal[[i]] + dLossFake[[i]] )
          }

        # train generator

        gLoss <- NULL
        if( self$useImageNetWeights == TRUE )
          {
          imageFeatures = self$vggModel$predict( imagesHighResolution )
          gLoss <- self$combinedModel$train_on_batch(
            list( imagesLowResolution, imagesHighResolution ),
            list( valid, imageFeatures ) )
          } else {
          gLoss <- self$combinedModel$train_on_batch(
            list( imagesLowResolution, imagesHighResolution ), valid )
          }

        cat( "Epoch ", epoch, ": [Discriminator loss: ", dLoss[[1]], "] ",
          "[Generator loss: ", gLoss[[1]], "]\n", sep = '' )

        if( self$dimensionality == 2 )
          {
          if( ! is.na( sampleInterval ) )
            {
            if( ( ( epoch - 1 ) %% sampleInterval ) == 0 )
              {
              # Do a 2x3 grid
              #
              # low res image | high res image | original high res image
              # low res image | high res image | original high res image

              X <- list()

              indices <- sample.int( dim( X_trainLowResolution )[1], 2 )

              imageLowResolution <- X_trainLowResolution[indices[1],,,, drop = FALSE]
              imageHighResolution <- X_trainHighResolution[indices[1],,,, drop = FALSE]

              X[[1]] <- imageLowResolution
              X[[2]] <- self$generator$predict( imageLowResolution )
              X[[3]] <- imageHighResolution

              imageLowResolution <- X_trainLowResolution[indices[2],,,, drop = FALSE]
              imageHighResolution <- X_trainHighResolution[indices[2],,,, drop = FALSE]

              X[[4]] <- imageLowResolution
              X[[5]] <- self$generator$predict( imageLowResolution )
              X[[6]] <- imageHighResolution

              for( i in seq_len( length( X ) ) )
                {
                X[[i]] <- ( X[[i]] - min( X[[i]] ) ) /
                  ( max( X[[i]] ) - min( X[[i]] ) )
                X[[i]] <- drop( X[[i]] )
                }
              XrowA <- image_append(
                         c( image_read( X[[1]] ),
                            image_read( X[[2]] ),
                            image_read( X[[3]] ) ) )
              XrowB <- image_append(
                         c( image_read( X[[4]] ),
                            image_read( X[[5]] ),
                            image_read( X[[6]] ) ) )
              XAB <- image_append( c( XrowA, XrowB ), stack = TRUE )

              sampleDir <- dirname( sampleFilePrefix )
              if( ! dir.exists( sampleDir ) )
                {
                dir.create( sampleDir, showWarnings = TRUE, recursive = TRUE )
                }

              imageFileName <- paste0( sampleFilePrefix, "_iteration" , epoch, ".jpg" )
              cat( "   --> writing sample image: ", imageFileName, "\n" )
              image_write( XAB, path = imageFileName, format = "jpg")
              }
            }
          }
        }
      }
    )
  )
