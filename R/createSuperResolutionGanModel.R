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
#'
#' @section Arguments:
#' \describe{
#'  \item{lowResolutionImageSize}{}
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
#'    lowResolutionImageSize = c( 112, 112, 3 ) )
#' testthat::expect_error({
#'  ganModel <- SuperResolutionGanModel$new(
#'    lowResolutionImageSize = c( 64, 65, 3 ) )
#'  })
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

    lowResolutionImageSize = c( 64, 64, 3 ),

    highResolutionImageSize = c( 256, 256, 3 ),

    numberOfChannels = 3,

    numberOfResidualBlocks = 16,

    numberOfFiltersAtBaseLayer = c( 64, 64 ),

    scaleFactor = 2,

    useImageNetWeights = TRUE,

    initialize = function( lowResolutionImageSize,
      scaleFactor = 2, useImageNetWeights = TRUE,
      numberOfResidualBlocks = 16, numberOfFiltersAtBaseLayer = c( 64, 64 ) )
      {
      self$lowResolutionImageSize <- lowResolutionImageSize
      self$numberOfChannels <- tail( self$lowResolutionImageSize, 1 )
      self$numberOfResidualBlocks <- numberOfResidualBlocks
      self$numberOfFiltersAtBaseLayer <- numberOfFiltersAtBaseLayer
      self$useImageNetWeights <- useImageNetWeights

      self$scaleFactor <- scaleFactor
      if( ! scaleFactor %in% c( 1, 2, 4, 8 ) )
        {
        stop( "Error:  scale factor must be one of 1, 2, 4, or 8." )
        }

      self$dimensionality <- NA
      if( length( self$lowResolutionImageSize ) == 3 )
        {
        self$dimensionality <- 2
        } else if( length( self$lowResolutionImageSize ) == 4 ) {
        self$dimensionality <- 3
        if( self$useImageNetWeights == TRUE )
          {
          self$useImageNetWeights <- FALSE
          warning( "Warning: imageNet weights are unavailable for 3D." )
          }
        } else {
        stop( "Incorrect size for lowResolutionImageSize.\n" )
        }

      optimizer <- optimizer_adam( lr = 0.0002, beta_1 = 0.5 )

      # Images

      self$highResolutionImageSize <- c( as.integer( self$scaleFactor ) *
        self$lowResolutionImageSize[1:self$dimensionality], self$numberOfChannels )

      highResolutionImage <- layer_input( shape = self$highResolutionImageSize )

      lowResolutionImage <- layer_input( shape = self$lowResolutionImageSize )

      # Build generator

      self$generator <- self$buildGenerator()

      fakeHighResolutionImage <- self$generator( lowResolutionImage )

      # Build discriminator

      self$discriminator <- self$buildDiscriminator()
      self$discriminator$compile( loss = 'mse',
        optimizer = optimizer, metrics = list( 'acc' ) )

      # Vgg

      self$vggModel <- self$buildTruncatedVggModel()
      self$vggModel$trainable <- FALSE
      self$vggModel$compile( loss = 'mse', optimizer = optimizer,
        metrics = list( 'accuracy') )

      if( self$dimensionality == 2 )
        {
        self$discriminatorPatchSize <- c( 16, 16, 1 )
        } else {
        self$discriminatorPatchSize <- c( 16, 16, 16, 1 )
        }
      # unlist( self$vggModel$output_shape )[1:self$dimensionality], 1 )

      # Discriminator

      self$discriminator$trainable <- FALSE

      validity <- self$discriminator( fakeHighResolutionImage )

      # Combined model

      if( self$useImageNetWeights == TRUE )
        {
        fakeFeatures <- self$vggModel( fakeHighResolutionImage )
        self$combinedModel = keras_model( inputs = list( lowResolutionImage, highResolutionImage ),
                                          outputs = list( validity, fakeFeatures ) )
        self$combinedModel$compile( loss = list( 'binary_crossentropy', 'mse' ),
          loss_weights = list( 1e-3, 1 ), optimizer = optimizer )
        } else {
        self$combinedModel = keras_model( inputs = list( lowResolutionImage, highResolutionImage ),
                                          outputs = validity )
        self$combinedModel$compile( loss = list( 'binary_crossentropy' ),
          optimizer = optimizer )
        }
      },

    buildTruncatedVggModel = function()
      {
      vggTmp <- NULL
      if( self$dimensionality == 2 )
        {
        if( self$useImageNetWeights == TRUE )
          {
          vggTmp <- createVggModel2D( c( 224, 224, 3 ), style = '19' )
          kerasVgg <- application_vgg19( weights = "imagenet" )
          vggTmp$set_weights( kerasVgg$get_weights() )
          } else {
          vggTmp <- createVggModel2D( self$highResolutionImageSize, style = '19' )
          }

        } else {
        vggTmp <- createVggModel3D( self$highResolutionImageSize, style = '19' )
        }

      vggTmp$outputs = list( vggTmp$layers[[10]]$output )

      highResolutionImage <- layer_input( self$highResolutionImageSize )
      highResolutionImageFeatures <- vggTmp( highResolutionImage )

      vggModel <- keras_model( inputs = highResolutionImage,
        outputs = highResolutionImageFeatures )

      return( vggModel )
      },

    buildGenerator = function( numberOfFilters = 64 )
      {
      buildResidualBlock <- function( input, numberOfFilters, kernelSize = 3 )
        {
        shortcut <- input

        if( self$dimensionality == 2 )
          {
          input <- input %>% layer_conv_2d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          } else {
          input <- input %>% layer_conv_3d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          }
        input <- input %>% layer_activation_relu()
        input <- input %>% layer_batch_normalization( momentum = 0.8 )
        if( self$dimensionality == 2 )
          {
          input <- input %>% layer_conv_2d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          } else {
          input <- input %>% layer_conv_3d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = 1, padding = 'same' )
          }
        input <- input %>% layer_batch_normalization( momentum = 0.8 )
        input <- list( input, shortcut ) %>% layer_add()
        return( input )
        }

      buildDeconvolutionLayer <- function( input, numberOfFilters = 256, kernelSize = 3 )
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

      image <- layer_input( shape = self$lowResolutionImageSize )

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

      if( self$scaleFactor >= 2 )
        {
        model <- buildDeconvolutionLayer( model )
        }
      if( self$scaleFactor >= 4 )
        {
        model <- buildDeconvolutionLayer( model )
        }
      if( self$scaleFactor == 8 )
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

      image <- layer_input( shape = self$highResolutionImageSize )

      model <- image %>% buildLayer( self$numberOfFiltersAtBaseLayer[2],
        normalization = FALSE )
      model <- model %>% buildLayer( self$numberOfFiltersAtBaseLayer[2],
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

      valid <- array( data = 1, dim = c( batchSize, self$discriminatorPatchSize ) )
      fake <- array( data = 0, dim = c( batchSize, self$discriminatorPatchSize ) )

      for( epoch in seq_len( numberOfEpochs ) )
        {
        indices <- sample.int( dim( X_trainLowResolution )[1], batchSize )

        lowResolutionImages <- NULL
        highResolutionImages <- NULL
        if( self$dimensionality == 2 )
          {
          lowResolutionImages <- X_trainLowResolution[indices,,,, drop = FALSE]
          highResolutionImages <- X_trainHighResolution[indices,,,, drop = FALSE]
          } else {
          lowResolutionImages <- X_trainLowResolution[indices,,,,, drop = FALSE]
          highResolutionImages <- X_trainHighResolution[indices,,,,, drop = FALSE]
          }

        # train discriminator

        fakeHighResolutionImages <- self$generator$predict( lowResolutionImages )

        dLossReal <- self$discriminator$train_on_batch( highResolutionImages, valid )
        dLossFake <- self$discriminator$train_on_batch( fakeHighResolutionImages, fake )

        dLoss <- list()
        for( i in seq_len( length( dLossReal ) ) )
          {
          dLoss[[i]] <- 0.5 * ( dLossReal[[i]] + dLossFake[[i]] )
          }

        # train generator

        gLoss <- NULL
        if( self$useImageNetWeights == TRUE )
          {
          imageFeatures = self$vggModel$predict( highResolutionImages )
          gLoss <- self$combinedModel$train_on_batch(
            list( lowResolutionImages, highResolutionImages ),
            list( valid, imageFeatures ) )
          } else {
          gLoss <- self$combinedModel$train_on_batch(
            list( lowResolutionImages, highResolutionImages ), valid )
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

              lowResolutionImage <- X_trainLowResolution[indices[1],,,, drop = FALSE]
              highResolutionImage <- X_trainHighResolution[indices[1],,,, drop = FALSE]

              X[[1]] <- lowResolutionImage
              X[[2]] <- self$generator$predict( lowResolutionImage )
              X[[3]] <- highResolutionImage

              lowResolutionImage <- X_trainLowResolution[indices[2],,,, drop = FALSE]
              highResolutionImage <- X_trainHighResolution[indices[2],,,, drop = FALSE]

              X[[4]] <- lowResolutionImage
              X[[5]] <- self$generator$predict( lowResolutionImage )
              X[[6]] <- highResolutionImage

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
