#' In-painting with contextual attention
#'
#' Original generative adverserial network (GAN) model from the
#' paper:
#'
#'   https://arxiv.org/abs/1801.07892
#'
#' and ported from the (TensorFlow) implementation:
#'
#'   https://github.com/JiahuiYu/generative_inpainting
#'
#' @docType class
#'
#'
#' @section Arguments:
#' \describe{
#'  \item{inputImageSize}{}
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
#' x = InpaintingDeepFillModel$new(c( 28, 28, 1 ))
#' \dontrun{
#' x$buildNetwork()
#' }
#'
#' @name InpaintingDeepFillModel
NULL

#' @export
InpaintingDeepFillModel <- R6::R6Class(
  "InpaintingDeepFillModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    dimensionality = 2,

    inputImageSize = c( 28, 28, 1 ),

    batchSize = 32,

    numberOfFiltersBaseLayer = 32L,

    initialize = function( inputImageSize, batchSize = 32,
                           numberOfFiltersBaseLayer = 32L )
    {
      self$inputImageSize <- inputImageSize
      if( length( inputImageSize ) == 3 )
      {
        self$dimensionality <- 2
      } else if( length( inputImageSize ) == 4 ) {
        self$dimensionality <- 3
      } else {
        stop( "Error:  incorrect input image size specification." )
      }
      self$numberOfFiltersBaseLayer <- numberOfFiltersBaseLayer
    },

    generativeConvolutionLayer = function(
      model, numberOfFilters = 4L,
      kernelSize = 3L, stride = 1L,
      dilationRate = 1L, activation = 'elu',
      trainable = TRUE, name = '' )
    {
      output <- NULL
      if( self$dimensionality == 2 )
      {
        output <- model %>%
          layer_conv_2d( filters = numberOfFilters,
                         kernel_size = kernelSize, strides = stride,
                         dilation_rate = dilationRate, activation = activation,
                         padding = 'same', trainable = trainable, name = name )
      } else {
        output <- model %>%
          layer_conv_3d( filters = numberOfFilters,
                         kernel_size = kernelSize, strides = stride,
                         dilation_rate = dilationRate, activation = activation,
                         padding = 'same', trainable = trainable, name = name )
      }

      return( output )
    },

    discriminativeConvolutionLayer = function( model, numberOfFilters,
                                               kernelSize = 5L, stride = 2L, dilationRate = 1L,
                                               activation = 'leaky_relu', trainable = TRUE, name = '' )
    {
      output <- NULL
      if( self$dimensionality == 2 )
      {
        output <- model %>%
          layer_conv_2d( filters = numberOfFilters,
                         kernel_size = kernelSize, strides = stride,
                         dilation_rate = dilationRate, activation = activation,
                         padding = 'same', trainable = trainable, name = name )
      } else {
        output <- model %>%
          layer_conv_3d( filters = numberOfFilters,
                         kernel_size = kernelSize, strides = stride,
                         dilation_rate = dilationRate, activation = activation,
                         padding = 'same', trainable = trainable, name = name )
      }
      return( output )
    },

    generativeDeconvolutionLayer = function( model, numberOfFilters = 4L,
                                             trainable = TRUE, name = '' )
    {
      K <- keras::backend()

      shape <- K$int_shape( model )
      shape <- unlist( shape )

      newSize <- as.integer( 2.0 * shape[2:( self$dimensionality + 1 )] )
      resizedModel <- NULL
      if( self$dimensionality == 2 )
      {
        resizedModel <- layer_resample_tensor_2d( model, newSize )
      } else {
        resizedModel <- layer_resample_tensor_3d( model, newSize )
      }
      output <- self$generativeConvolutionLayer(
        resizedModel,
        numberOfFilters = numberOfFilters, kernelSize = 3, stride = 1,
        trainable = trainable )

      return( output )
    },

    buildNetwork = function( trainable = TRUE )
    {
      K <- keras::backend()

      imageInput <- layer_input( batch_shape =
                                   c( self$batchSize, self$inputImageSize ) )
      maskInput <- layer_input( batch_shape =
                                  c( self$batchSize, self$inputImageSize[1:self$dimensionality], 1 ) )

      output <- imageInput

      ones <- NULL
      if( self$dimensionality == 2 )
      {
        ones <- output %>% layer_lambda( f = function( X )
        { K$ones_like( X )[,,,1, drop = FALSE] } )
      } else {
        # ones <- K$ones_like( output )[,,,,1, drop = FALSE]
        ones <- output %>% layer_lambda( f = function( X )
        { K$ones_like( X )[,,,1, drop = FALSE] } )
      }

      maskedOnes <- list( ones, maskInput ) %>% layer_lambda(
        f = function( inputs )
        { return( inputs[[1]] * inputs[[2]] ) } )
      output <- layer_concatenate( list( output, ones, maskedOnes ),
                                   axis = as.integer( self$dimensionality + 1 ) )

      # Stage 1

      output <- self$generativeConvolutionLayer(
        output,
        self$numberOfFiltersBaseLayer, 5L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        2 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        4 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      outputShape <- unlist(
        K$int_shape( output ) )[2:( self$dimensionality + 1 )]

      resampledMaskInput <- NULL
      if( self$dimensionality == 2 )
      {
        resampledMaskInput <- layer_resample_tensor_2d(
          maskInput,
          shape = outputShape, interpolationType = 'nearestNeighbor' )
      } else {
        resampledMaskInput <- layer_resample_tensor_3d(
          maskInput,
          shape = outputShape, interpolationType = 'nearestNeighbor' )
      }

      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 2L )
      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 4L )
      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 8L )
      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 16L )

      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      output <- self$generativeDeconvolutionLayer(
        output, 2 * self$numberOfFiltersBaseLayer )
      output <- self$generativeConvolutionLayer(
        output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L )
      output <- self$generativeDeconvolutionLayer(
        output, self$numberOfFiltersBaseLayer )

      output <- self$generativeConvolutionLayer(
        output, as.integer( self$numberOfFiltersBaseLayer / 2 ), 3L, 1L )
      output <- self$generativeConvolutionLayer(
        output, 3L, 3L, 1L, activation = NULL )

      output <- output %>% layer_lambda( function( X )
      { return( tensorflow::tf$clip_by_value( X, -1.0, 1.0 ) ) } )

      modelStage1 <- keras_model( inputs = list( imageInput, maskInput ),
                                  outputs = output )

      # Stage 2

      maskedOnes <- list( ones, maskInput ) %>%
        layer_lambda( f = function( inputs )
        { return( inputs[[1]] * inputs[[2]] ) } )

      output <- list( output, maskInput, imageInput ) %>%
        layer_lambda( f = function( inputs )
        { return( inputs[[1]] * inputs[[2]] + inputs[[3]] *
                    ( 1.0 - inputs[[2]] ) )
        } )

      # Conv branch

      outputNow <- layer_concatenate(
        list( output, ones, maskedOnes ),
        axis = as.integer( self$dimensionality + 1 ) )
      output <- self$generativeConvolutionLayer( outputNow,
                                                 self$numberOfFiltersBaseLayer, 5, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 2 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 2L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 4L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 8L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 16L )

      outputHallu <- output

      # Attention branch

      output <- self$generativeConvolutionLayer( outputNow,
                                                 self$numberOfFiltersBaseLayer, 5, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output,
                                                 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L,
        activation = 'relu' )

      contextualInputList <- list( output, output, resampledMaskInput )
      if( self$dimensionality == 2 )
      {
        output <- layer_contextual_attention_2d(
          contextualInputList, 3L, 1L, dilationRate = 2L )
      } else {
        output <- layer_contextual_attention_3d(
          contextualInputList, 3L, 1L, dilationRate = 2L )
      }

      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output,
        4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      output <- layer_concatenate( list( outputHallu, output ),
                                   axis = as.integer( self$dimensionality + 1 ) )

      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer(
        output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeDeconvolutionLayer(
        output, 2 * self$numberOfFiltersBaseLayer )
      output <- self$generativeConvolutionLayer(
        output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeDeconvolutionLayer(
        output,   self$numberOfFiltersBaseLayer )
      output <- self$generativeConvolutionLayer(
        output, as.integer( 0.5 * self$numberOfFiltersBaseLayer ), 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 3L, 3L, 1L, 1L )

      output <- output %>% layer_lambda( function( X )
      { return( tensorflow::tf$clip_by_value( X, -1.0, 1.0 ) ) } )

      modelStage2 <- keras_model( inputs = list( imageInput, maskInput ),
                                  outputs = output )

      return( list( modelStage1 = modelStage1, modelStage2 = modelStage2 ) )
    },

    buildLocalWganGpDiscriminator = function( model, reuse = FALSE,
                                              trainable = TRUE )
    {
      with( tensorflow::tf$variable_scope( 'localDiscriminator', reuse = reuse ) )
      {
        numberOfFilters <- 64
        numberOfLayers <- 4

        for( i in seq_len( numberOfLayers ) )
        {
          localNumberOfFilters = numberOfFilters * 2 ^ ( i - 1 )
          model <- discriminativeConvolutionLayer( model,
                                                   localNumberOfFilters, trainable = trainable )
        }
        model <- model %>% layer_flatten()
      }
    },

    buildGlobalWganGpDiscriminator = function( model,
                                               reuse = FALSE, trainable = TRUE )
    {
      with( tensorflow::tf$variable_scope( 'globalDiscriminator', reuse = reuse ) )
      {
        numberOfFilters <- 64
        localNumberOfFilters <- numberOfFilters * c( 1, 2, 4, 4 )
        numberOfLayers <- length( localNumberOfFilters )

        for( i in seq_len( numberOfLayers ) )
        {
          model <- discriminativeConvolutionLayer(
            model,
            localNumberOfFilters[i], trainable = trainable )
        }
        model <- model %>% layer_flatten()
      }
    },

    buildCombinedWganGpDiscriminator = function(
      localModel,
      globalModel, reuse = FALSE, trainable = TRUE )
    {
      with( tensorflow::tf$variable_scope( 'discriminator', reuse = reuse ) )
      {
        localDiscriminator <- self$buildLocalWganGpDiscriminator(
          localModel, reuse = reuse, trainable = trainable )
        globalDiscriminator <- self$buildGlobalWganGpDiscriminator(
          globalModel, reuse = reuse, trainable = trainable )

        localDiscriminator <- localDiscriminator %>% layer_dense( 1 )
        globalDiscriminator <- globalDiscriminator %>% layer_dense( 1 )

        return( list( localDiscriminator, globalDiscriminator ) )
      }
    },

    randomWeightedImage = function( X, Y )
    {
      K <- keras::backend()
      inputShape <- K$int_shape( X )
      batchSize <- inputShape[[1]]
      if( self$dimensionality == 2 )
      {
        alpha <- K$random_uniform( c( batchSize, 1L, 1L, 1L ) )
      } else {
        alpha <- K$random_uniform( c( batchSize, 1L, 1L, 1L, 1L ) )
      }
      return( ( 1.0 - alpha ) * X + alpha * Y )
    },

    wassersteinLoss = function( X, Y )
    {
      K <- keras::backend()
      dLoss <- tensorflow::tf$math$reduce_mean( Y - X )
      gLoss <- -tensorflow::tf$math$reduce_mean( Y )
      return( gLoss, dLoss )
    },

    gradientPenaltyLoss = function( X, Y, mask = NA, norm = 1.0 )
    {
      K <- keras::backend()
      gradients <- tensorflow::tf$gradients( Y, X )[0]
      if( is.na( mask ) )
      {
        mask <- tensorflow::tf$ones_like( gradients )
      }
      slopes <- tensorflow::tf$sqrt( tensorflow::tf$reduce_sum( tensorflow::tf$square( gradients ) * mask,
                                                                axis = 1:( self$dimensionality + 1 ) ) )
      return( tensorflow::tf$math$reduce_mean( tensorflow::tf$square( slopes - norm ) ) )
    },

    train = function( X_train, numberOfEpochs = 200, maskRegionSize = NA,
                      limitTrainingToCoarseNetwork = FALSE )
    {
      K <- keras::backend()

      network <- self$buildNetwork()

      modelStage1 <- network$modelStage1
      modelStage2 <- network$modelStage2

      # Build spatial discount mask

      gamma <- 0.9
      discountMask <- array( data = 1, dim = maskRegionSize )
      if( self$dimensionality == 2 )
      {
        for( i in seq_len( maskRegionSize[1] ) )
        {
          for( j in seq_len( maskRegionSize[2] ) )
          {
            discountMask <- max( gamma ^ min( i, maskRegionSize[1] - i ),
                                 gamma ^ min( j, maskRegionSize[2] - j ) )
          }
        }
      } else {
        for( i in seq_len( maskRegionSize[1] ) )
        {
          for( j in seq_len( maskRegionSize[2] ) )
          {
            for( k in seq_len( maskRegionSize[3] ) )
            {
              discountMask <- max( gamma ^ min( i, maskRegionSize[1] - i ),
                                   gamma ^ min( j, maskRegionSize[2] - j ),
                                   gamma ^ min( k, maskRegionSize[3] - k ) )
            }
          }
        }
      }
      discountMask <- array( data = discountMask,
                             dim = c( 1, dim( discountMask ), 1 ) )

      # Perform batchwise training

      for( i in seq_len( numberOfEpochs ) )
      {
        # Sample batch from full data set

        batchIndices <- sample.int( dim( X_train )[1], self$batchSize )

        # Build graph with losses

        X_batch <- X_train
        if( self$dimensionality == 2 )
        {
          X_batch <- X_train[batchIndices,,,, drop = FALSE]
        } else {
          X_batch <- X_train[batchIndices,,,,, drop = FALSE]
        }

        # Generate random mask

        beginCorner <- rep( 0, self$dimensionality )
        for( i in seq_len( self$dimensionality ) )
        {
          beginCorner[d] <- runif( 1, min = 1,
                                   max = self$inputImageSize[d] - maskRegionSize[d] + 1 )
        }
        randomMask <- drop( array( data = 0,
                                   dim = head( self$inputImageSize, self$dimensionality ) ) )

        endCorner <- beginCorner + maskRegionSize
        if( self$dimensionality == 2 )
        {
          randomMask[beginCorner[1]:endCorner[1],
                     beginCorner[2]:endCorner[2]] <- 1
        } else {
          randomMask[beginCorner[1]:endCorner[1],
                     beginCorner[2]:endCorner[2],
                     beginCorner[3]:endCorner[3]] <- 1
        }

        X_mask <- array( data = randomMask, dim = c( 1, dim( randomMask ), 1 ) )

        # mask the batch data

        X_masked <- apply( X_batch, self$dimensionality + 1,
                           function( x ) x * ( 1.0 - X_mask ) )
        X_predictedStage1 <- modelStage1$predict( X_masked )
        X_predictedStage2 <- modelStage2$predict( X_masked )
        X_complete <- X_predicted * X_mask + X_masked * ( 1.0 - X_mask )

        # Get local mask region patches

        if( self$dimensionality == 2 )
        {
          X_maskLocalPatch <- X_mask[,beginCorner[1]:endCorner[1],
                                     beginCorner[2]:endCorner[2],]
          X_batchLocalPatch <- X_batch[,beginCorner[1]:endCorner[1],
                                       beginCorner[2]:endCorner[2],]
          X_maskedLocalPatch <- X_masked[,beginCorner[1]:endCorner[1],
                                         beginCorner[2]:endCorner[2],]
          X_predictedStage1LocalPatch <-
            X_predictedStage1[,beginCorner[1]:endCorner[1],
                              beginCorner[2]:endCorner[2],]
          X_predictedStage2LocalPatch <-
            X_predictedStage2[,beginCorner[1]:endCorner[1],
                              beginCorner[2]:endCorner[2],]
          X_completeLocalPatch <-
            X_complete[,beginCorner[1]:endCorner[1],
                       beginCorner[2]:endCorner[2],]
        } else {
          X_maskLocalPatch <- X_mask[,beginCorner[1]:endCorner[1],
                                     beginCorner[2]:endCorner[2],
                                     beginCorner[3]:endCorner[3],]
          X_batchLocalPatch <- X_batch[,beginCorner[1]:endCorner[1],
                                       beginCorner[2]:endCorner[2],
                                       beginCorner[3]:endCorner[3],]
          X_maskedLocalPatch <- X_masked[,beginCorner[1]:endCorner[1],
                                         beginCorner[2]:endCorner[2],
                                         beginCorner[3]:endCorner[3],]
          X_predictedStage1LocalPatch <-
            X_predictedStage1[,beginCorner[1]:endCorner[1],
                              beginCorner[2]:endCorner[2],
                              beginCorner[3]:endCorner[3],]
          X_predictedStage2LocalPatch <-
            X_predictedStage2[,beginCorner[1]:endCorner[1],
                              beginCorner[2]:endCorner[2],
                              beginCorner[3]:endCorner[3],]
          X_completeLocalPatch <- X_complete[,beginCorner[1]:endCorner[1],
                                             beginCorner[2]:endCorner[2],
                                             beginCorner[3]:endCorner[3],]
        }





        # Calculate losses

        losses <- list()
        l1Alpha <- 1.2

        losses$l1Loss <- l1Alpha * mean( abs( X_batchLocalPatch -
                                                X_predictedStage1LocalPatch ) * discountMask )
        if( ! limitTrainingToCoarseNetwork )
        {
          losses$l1Loss <- losses$l1Loss + mean( abs( X_batchLocalPatch -
                                                        X_predictedStage2LocalPatch ) * discountMask )
        }
        losses$aeLoss <- l1Alpha * mean( abs( X_batch - X_predictedStage1 ) *
                                           ( 1.0 - X_mask ) )
        if( ! limitTrainingToCoarseNetwork )
        {
          losses$aeLoss <- losses$aeLoss + mean(
            abs( X_batch - X_predictedStage1 ) * ( 1.0 - X_mask ) )
        }
        losses$aeLoss <- losses$aeLoss / mean( 1.0 - X_mask )

        # Add global and local Wasserstein GAN losses

        modelBatch <- layer_input( batch_shape = dim( X_batch ) )
        modelComplete <- layer_input( batch_shape = dim( X_complete ) )
        modelPositiveNegative <- list( modelBatch, modelComplete ) %>%
          layer_concatenate( axis = 0 )
        mask <- layer_input( shape = dim( X_mask ) )

        modelBatchLocalPatch <-
          layer_input( batch_shape = dim( X_batchLocalPatch ) )
        modelCompleteLocalPatch <-
          layer_input( batch_shape = dim( X_completeLocalPatch ) )
        modelPositiveNegativeLocalPatch <-
          list( modelBatchLocalPatch, modelCompleteLocalPatch ) %>%
          layer_concatenate( axis = 0 )

        wganGpModel <- self$buildCombinedWganGpDiscriminator(
          modelPositiveNegativeLocalPatch, modelPositiveNegative )
        localComponents <- tensorflow::tf$split( wganGpModel$localDiscriminator, 2 )
        globalComponents <- tensorflow::tf$split( wganGpModel$globalDiscriminator, 2 )

        # WGAN Loss

        lossLocal <- self$wassersteinLoss(
          localComponents[[1]], localComponents[[2]] )
        lossGlobal <- self$wassersteinLoss(
          globalComponents[[1]], globalComponents[[2]] )

        losses$dLoss <- lossLocal$dLoss + lossGlobal$dLoss
        losses$gLoss <- lossLocal$gLoss + lossGlobal$gLoss

        # Gradient penalty loss

        interpolatedLocal <-
          randomWeightedImage( modelBatchLocalPatch, modelCompleteLocalPatch )
        interpolatedGlobal <- randomWeightedImage( modelBatch, modelComplete )

        wganGpInterpolatedModel <- self$buildCombinedWganGpDiscriminator(
          interpolatedLocal, interpolatedGlobal )

        # Apply penalty

        wganGpLambda <- 10.0
        wganGpAlpha <- 0.001

        localPenalty <- self$gradientPenaltyLoss(
          interpolatedLocal,
          wgandGpInterpolatedModel$localDiscriminator, mask = localPatchMask )
        globalPenalty <- self$gradientPenaltyLoss(
          interpolatedGlobal,
          wgandGpInterpolatedModel$globalDiscriminator, mask = mask )

        losses$wganGpLosses <- wganGpLambda * ( localPenalty + globalPenalty )
        losses$dLoss <- losses$dLoss + losses$wganGpLosses

        if( limitTrainingToCoarseNetwork == TRUE )
        {
          losses$gLoss <- 0
        } else {
          losses$gLoss <- wganGpAlpha * losses$gLoss
        }
        losses$gLoss <- l1Alpha * losses$l1Loss
      }
    }
  )
)

#' Contextual attention layer (2-D)
#'
#' Contextual attention layer for generative image inpainting described in
#'
#' Jiahui Yu, et al., Generative Image Inpainting with Contextual Attention,
#'      CVPR 2018.
#'
#' available here:
#'
#'         \code{https://arxiv.org/abs/1801.07892}
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{layer <- ContextualAttentionLayer2D$new( scale )
#'
#' layer$call( x, mask = NULL )
#' layer$build( input_shape )
#' layer$compute_output_shape( input_shape )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{layer}{A \code{process} object.}
#'  \item{scale}{feature scale.  Default = 20}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$build}
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return output tensor with the same shape as the input.
#'
#' @examples
#' x = ContextualAttentionLayer2D$new()
#' x$build()
#'
#' @name ContextualAttentionLayer2D
NULL

#' @export
ContextualAttentionLayer2D <- R6::R6Class(
  "ContextualAttentionLayer2D",

  inherit = KerasLayer,

  public = list(

    kernelSize = 3L,

    stride = 1L,

    dilationRate = 1L,

    fusionKernelSize = 3L,

    initialize = function( kernelSize = 3L, stride = 1L,
                           dilationRate = 1L, fusionKernelSize = 3L )
    {
      self$kernelSize = kernelSize
      self$stride = stride
      self$dilationRate = dilationRate
      self$fusionKernelSize = fusionKernelSize
      self
    },

    compute_output_shape = function( input_shape )
    {
      return( input_shape[[1]] )
    },

    call = function( inputs, mask = NULL )
    {
      # inputs should consist of the foreground tensor, background tensor,
      # and, optionally, the mask
      if( length( inputs ) < 2 )
      {
        errorMessage <- paste0( "inputs should consist of the foreground ",
                                "tensor, background tensor, and, optionally, the mask." )
        stop( errorMessage )
      }

      foregroundTensor <- inputs[[1]]
      backgroundTensor <- inputs[[2]]
      mask <- NULL
      if( length( inputs ) > 2 )
      {
        mask <- inputs[[3]]
      }

      K <- keras::backend

      # Get tensor shapes

      foregroundShape <- foregroundTensor$get_shape()$as_list()
      backgroundShape <- backgroundTensor$get_shape()$as_list()

      maskShape <- NULL
      if( ! is.null( mask ) )
      {
        maskShape <- mask$get_shape()$as_list()
      }

      # Extract patches from background and reshape to be
      #  c( batchSize, backgroundKernelSize, backgroundKernelSize, channelSize,
      #     height*width )

      backgroundKernelSize <- as.integer( 2 * self$dilationRate )
      stridexRate <- as.integer( self$stride * self$dilationRate )

      backgroundPatches <- tensorflow::tf$image$extract_patches(
        backgroundTensor,
        sizes = c( 1, backgroundKernelSize, backgroundKernelSize, 1 ),
        strides = c( 1, stridexRate, stridexRate, 1 ),
        rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      backgroundPatches <- tensorflow::tf$reshape(
        backgroundPatches,
        c( tensorflow::tf$shape( backgroundTensor )[1], -1L,
           backgroundKernelSize, backgroundKernelSize,
           tensorflow::tf$shape( backgroundTensor )[4] ) )
      backgroundPatches <- tensorflow::tf$transpose( backgroundPatches,
                                                     c( 0L, 2L, 3L, 4L, 1L ) )

      # Resample foreground, background, and mask

      newForegroundShape <- as.integer(
        foregroundShape[2:3] / self$dilationRate )
      resampledForegroundTensor <- layer_resample_tensor_2d( foregroundTensor,
                                                             shape = newForegroundShape,
                                                             interpolationType = 'nearestNeighbor' )

      newBackgroundShape <-
        as.integer( backgroundShape[2:3] / self$dilationRate )
      resampledBackgroundTensor <- layer_resample_tensor_2d( backgroundTensor,
                                                             shape = newBackgroundShape,
                                                             interpolationType = 'nearestNeighbor' )

      newMaskShape <- maskShape
      if( ! is.null( mask ) )
      {
        newMaskShape <- as.integer( maskShape[2:3] / self$dilationRate )
        mask <- layer_resample_tensor_2d( mask,
                                          shape = newMaskShape, interpolationType = 'nearestNeighbor' )
      }

      # Create resampled background patches

      resampledBackgroundPatches <- tensorflow::tf$image$extract_patches(
        resampledBackgroundTensor,
        sizes = c( 1, self$kernelSize, self$kernelSize, 1 ),
        strides = c( 1, self$stride, self$stride, 1 ),
        rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      resampledBackgroundPatches <- tensorflow::tf$reshape(
        resampledBackgroundPatches,
        c( tensorflow::tf$shape( resampledBackgroundTensor )[1], -1L,
           self$kernelSize, self$kernelSize,
           tensorflow::tf$shape( resampledBackgroundTensor )[4] ) )
      resampledBackgroundPatches <- tensorflow::tf$transpose(
        resampledBackgroundPatches, c( 0L, 2L, 3L, 4L, 1L ) )

      # Process mask

      if( is.null( mask ) )
      {
        maskShape <- c( 1L, newBackgroundShape, 1L )
        mask = tensorflow::tf$zeros( maskShape )
      }

      maskPatches <- tensorflow::tf$image$extract_patches(
        mask,
        sizes = c( 1, self$kernelSize, self$kernelSize, 1 ),
        strides = c( 1, self$stride, self$stride, 1 ),
        rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      maskPatches <- tensorflow::tf$reshape( maskPatches,
                                             c( 1L, -1L, self$kernelSize, self$kernelSize, 1L ) )
      maskPatches <- tensorflow::tf$transpose( maskPatches, c( 0L, 2L, 3L, 4L, 1L ) )

      maskPatches <- maskPatches[1,,,,]
      maskData <- tensorflow::tf$cast(
        tensorflow::tf$equal( tensorflow::tf$math$reduce_mean( maskPatches,
                                                               axis = c( 0L, 1L, 2L ), keepdims = TRUE ), 0.0 ),
        tensorflow::tf$float32 )

      # Split into groups

      resampledForegroundGroups <- tensorflow::tf$split(
        resampledForegroundTensor,
        resampledForegroundTensor$get_shape()$as_list()[1], axis = 0L )
      backgroundGroups <- tensorflow::tf$split( backgroundPatches,
                                                backgroundTensor$get_shape()$as_list()[1], axis = 0L )
      resampledBackgroundGroups <- tensorflow::tf$split(
        resampledBackgroundPatches,
        resampledBackgroundTensor$get_shape()$as_list()[1],
        axis = 0L )

      numberOfIterations <- min( c( length( resampledForegroundGroups ) ),
                                 c( length( backgroundGroups ) ),
                                 c( length( resampledBackgroundGroups ) ) )

      outputGroups <- list()
      for( i in seq_len( numberOfIterations ) )
      {
        rg <- resampledBackgroundGroups[[i]][1,,,,]
        rgNorm <- rg / tensorflow::tf$maximum( tensorflow::tf$sqrt( tensorflow::tf$reduce_sum(
          tensorflow::tf$square( rg ), axis = c( 0L, 1L, 2L ) ) ), 1e-4 )
        fg <- resampledForegroundGroups[[i]]

        output <- tensorflow::tf$nn$conv2d( fg, rgNorm, strides = c( 1, 1, 1, 1 ),
                                            padding = 'SAME' )

        # fusion to encourage large patches

        if( self$fusionKernelSize > 0L )
        {
          fusionKernel <- tensorflow::tf$reshape( tensorflow::tf$eye( self$fusionKernelSize ),
                                                  c( self$fusionKernelSize, self$fusionKernelSize, 1L, 1L ) )

          output <- tf$reshape( output, c( 1L,
                                           as.integer( newForegroundShape[1] * newForegroundShape[2] ),
                                           as.integer( newBackgroundShape[1] * newBackgroundShape[2] ), 1L ) )
          output <- tf$nn$conv2d( output, fusionKernel,
                                  strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
          output <- tf$reshape( output, c( 1L, newForegroundShape[1],
                                           newForegroundShape[2], newBackgroundShape[1],
                                           newBackgroundShape[2] ) )

          output <- tf$transpose( output, c( 0L, 2L, 1L, 4L, 3L ) )
          output <- tf$reshape( output,
                                c( 1L, newForegroundShape[1] * newForegroundShape[2],
                                   newBackgroundShape[1] * newBackgroundShape[2], 1L ) )
          output <- tf$nn$conv2d( output, fusionKernel,
                                  strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
          output <- tf$reshape( output, c( 1L, newForegroundShape[2],
                                           newForegroundShape[1], newBackgroundShape[2],
                                           newBackgroundShape[1] ) )

          output <- tf$transpose( output, c( 0L, 2L, 1L, 4L, 3L ) )
        }

        output <- tf$reshape( output, c( 1L, newForegroundShape[1],
                                         newForegroundShape[2],
                                         newBackgroundShape[1] * newBackgroundShape[2] ) )

        # softmax to match

        output <- output * maskData
        output <- tf$nn$softmax( output * 10.0, axis = 3L )
        output <- output * maskData

        bg <- backgroundGroups[[i]][1,,,,]

        output <- tensorflow::tf$nn$conv2d_transpose(
          output, bg,
          tensorflow::tf$concat( list( list( 1L ), foregroundShape[2:4] ), axis = 0L ),
          strides = c( 1, self$dilationRate, self$dilationRate, 1 ) ) / 4.0
        outputGroups[[i]] <- output
      }

      output <- tensorflow::tf$concat( outputGroups, axis = 0L )
      output$set_shape( foregroundShape )

      return( output )
    }
  )
)


#' Contextual attention layer (2-D and 3-D)
#'
#' Contextual attention layer for generative image inpainting described in
#'
#' Jiahui Yu, et al., Generative Image Inpainting with Contextual Attention,
#'      CVPR 2018.
#'
#' available here:
#'
#'         \code{https://arxiv.org/abs/1801.07892}
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param kernelSize integer specifying convolution size
#' @param stride integer for specifyingstride length for sampling the tensor
#' @param dilationRate ingeger specifying dilation
#' @param fusionKernelSize Enhance saliency of large patches
#' @param name The name of the layer
#' @param trainable Whether the layer weights will be updated during training.
#'
#' @return a keras layer tensor
#' @examples
#' layer_contextual_attention_2d()
#' layer_contextual_attention_3d()
#' keras::keras_model_sequential() %>%
#'     layer_contextual_attention_2d(fusionKernelSize = 2)
#' keras::keras_model_sequential() %>%
#'     layer_contextual_attention_3d()
#' @export
layer_contextual_attention_2d <- function(
  object,
  kernelSize = 3L, stride = 1L, dilationRate = 1L,
  fusionKernelSize = 0L,
  name = NULL, trainable = FALSE ) {
  create_layer( ContextualAttentionLayer2D, object,
                list( kernelSize = kernelSize, stride = stride,
                      dilationRate = dilationRate,
                      fusionKernelSize = fusionKernelSize,
                      name = name, trainable = trainable )
  )
}

#' @rdname layer_contextual_attention_2d
#' @export
layer_contextual_attention_3d <- function(
  object,
  kernelSize = 3L, stride = 1L, dilationRate = 1L,
  fusionKernelSize = 0L,
  name = NULL, trainable = FALSE ) {
  create_layer( ContextualAttentionLayer3D, object,
                list( kernelSize = kernelSize, stride = stride,
                      dilationRate = dilationRate, fusionKernelSize = fusionKernelSize,
                      name = name, trainable = trainable )
  )
}


#' Contextual attention layer (3-D)
#'
#' Contextual attention layer for generative image inpainting described in
#'
#' Jiahui Yu, et al., Generative Image Inpainting with Contextual Attention,
#'      CVPR 2018.
#'
#' available here:
#'
#'         \code{https://arxiv.org/abs/1801.07892}
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{layer <- ContextualAttentionLayer3D$new( scale )
#'
#' layer$call( x, mask = NULL )
#' layer$build( input_shape )
#' layer$compute_output_shape( input_shape )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{layer}{A \code{process} object.}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$build}
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return output tensor with the same shape as the input.
#'
#' @examples
#' x = ContextualAttentionLayer3D$new()
#' x$build()
#' @name ContextualAttentionLayer3D
NULL

#' @export
ContextualAttentionLayer3D <- R6::R6Class(
  "ContextualAttentionLayer3D",

  inherit = KerasLayer,

  public = list(

    kernelSize = 3L,

    stride = 1L,

    dilationRate = 1L,

    fusionKernelSize = 3L,

    initialize = function( kernelSize = 3L, stride = 1L,
                           dilationRate = 1L, fusionKernelSize = 3L )
    {
      self$kernelSize = kernelSize
      self$stride = stride
      self$dilationRate = dilationRate
      self$fusionKernelSize = fusionKernelSize
      self
    },

    compute_output_shape = function( input_shape )
    {
      return( input_shape[[1]] )
    },

    call = function( inputs, mask = NULL )
    {
      # inputs should consist of the foreground tensor, background tensor,
      # and, optionally, the mask
      if( length( inputs ) < 2 )
      {
        errorMessage <- paste0( "inputs should consist of the foreground ",
                                "tensor, background tensor, and, optionally, the mask." )
        stop( errorMessage )
      }

      foregroundTensor <- inputs[[1]]
      backgroundTensor <- inputs[[2]]
      mask <- NULL
      if( length( inputs ) > 2 )
      {
        mask <- inputs[[3]]
      }

      K <- keras::backend

      # Get tensor shapes

      foregroundShape <- foregroundTensor$get_shape()$as_list()
      backgroundShape <- backgroundTensor$get_shape()$as_list()

      maskShape <- NULL
      if( ! is.null( mask ) )
      {
        maskShape <- mask$get_shape()$as_list()
      }

      # Extract patches from background and reshape to be
      #  c( batchSize, backgroundKernelSize, backgroundKernelSize,
      #     backgroundKernelSize, channelSize, height*width*depth )

      backgroundKernelSize <- as.integer( 2 * self$dilationRate )
      stridexRate <- as.integer( self$stride * self$dilationRate )

      backgroundPatches <- tensorflow::tf$extract_volume_patches(
        backgroundTensor,
        ksizes = c( 1, backgroundKernelSize, backgroundKernelSize,
                    backgroundKernelSize, 1 ),
        strides = c( 1, stridexRate, stridexRate, stridexRate, 1 ),
        rates = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )
      backgroundPatches <- tensorflow::tf$reshape(
        backgroundPatches,
        c( tensorflow::tf$shape( backgroundTensor )[1], -1L,
           backgroundKernelSize, backgroundKernelSize, backgroundKernelSize,
           tensorflow::tf$shape( backgroundTensor )[5] ) )
      backgroundPatches <- tensorflow::tf$transpose( backgroundPatches,
                                                     c( 0L, 2L, 3L, 4L, 5L, 1L ) )

      # Resample foreground, background, and mask

      newForegroundShape <-
        as.integer( foregroundShape[2:4] / self$dilationRate )
      resampledForegroundTensor <- layer_resample_tensor_3d(
        foregroundTensor,
        shape = newForegroundShape,
        interpolationType = 'nearestNeighbor' )

      newBackgroundShape <-
        as.integer( backgroundShape[2:4] / self$dilationRate )
      resampledBackgroundTensor <- layer_resample_tensor_3d(
        backgroundTensor,
        shape = newBackgroundShape,
        interpolationType = 'nearestNeighbor' )

      newMaskShape <- maskShape
      if( ! is.null( mask ) )
      {
        newMaskShape <- as.integer( maskShape[2:4] / self$dilationRate )
        mask <- layer_resample_tensor_3d( mask,
                                          shape = newMaskShape,
                                          interpolationType = 'nearestNeighbor' )
      }

      # Create resampled background patches

      resampledBackgroundPatches <- tensorflow::tf$extract_volume_patches(
        resampledBackgroundTensor, ksizes =
          c( 1, self$kernelSize, self$kernelSize, self$kernelSize, 1 ),
        strides = c( 1, self$stride, self$stride, self$stride, 1 ),
        rates = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )
      resampledBackgroundPatches <- tensorflow::tf$reshape(
        resampledBackgroundPatches,
        c( tensorflow::tf$shape( resampledBackgroundTensor )[1], -1L,
           self$kernelSize, self$kernelSize, self$kernelSize,
           tensorflow::tf$shape( resampledBackgroundTensor )[5] ) )
      resampledBackgroundPatches <- tensorflow::tf$transpose(
        resampledBackgroundPatches, c( 0L, 2L, 3L, 4L, 5L, 1L ) )

      # Process mask

      if( is.null( mask ) )
      {
        maskShape <- c( 1L, newBackgroundShape, 1L )
        mask = tensorflow::tf$zeros( maskShape )
      }

      maskPatches <- tensorflow::tf$extract_volume_patches(
        mask,
        ksizes = c( 1, self$kernelSize, self$kernelSize, self$kernelSize, 1 ),
        strides = c( 1, self$stride, self$stride, self$stride, 1 ),
        rates = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )
      maskPatches <- tensorflow::tf$reshape( maskPatches,
                                             c( 1L, -1L, self$kernelSize, self$kernelSize,
                                                self$kernelSize, 1L ) )
      maskPatches <- tensorflow::tf$transpose( maskPatches,
                                               c( 0L, 2L, 3L, 4L, 5L, 1L ) )

      maskPatches <- maskPatches[1,,,,,]
      maskData <- tensorflow::tf$cast( tensorflow::tf$equal(
        tensorflow::tf$math$reduce_mean( maskPatches,
                                         axis = c( 0L, 1L, 2L, 3L ), keepdims = TRUE ), 0.0 ),
        tensorflow::tf$float32 )

      # Split into groups

      resampledForegroundGroups <- tensorflow::tf$split(
        resampledForegroundTensor,
        resampledForegroundTensor$get_shape()$as_list()[1], axis = 0L )
      backgroundGroups <- tensorflow::tf$split( backgroundPatches,
                                                backgroundTensor$get_shape()$as_list()[1], axis = 0L )
      resampledBackgroundGroups <- tensorflow::tf$split(
        resampledBackgroundPatches,
        resampledBackgroundTensor$get_shape()$as_list()[1], axis = 0L )

      numberOfIterations <- min( c( length( resampledForegroundGroups ) ),
                                 c( length( backgroundGroups ) ),
                                 c( length( resampledBackgroundGroups ) ) )

      outputGroups <- list()
      for( i in seq_len( numberOfIterations ) )
      {
        rg <- resampledBackgroundGroups[[i]][1,,,,,]
        rgNorm <- rg / tensorflow::tf$maximum( tensorflow::tf$sqrt( tensorflow::tf$reduce_sum(
          tensorflow::tf$square( rg ), axis = c( 0L, 1L, 2L, 3L ) ) ), 1e-4 )
        fg <- resampledForegroundGroups[[i]]

        output <- tensorflow::tf$nn$conv2d( fg, rgNorm, strides = c( 1, 1, 1, 1 ),
                                            padding = 'SAME' )

        # fusion to encourage large patches

        if( self$fusionKernelSize > 0L )
        {
          fusionKernel <- tensorflow::tf$reshape( tensorflow::tf$eye( self$fusionKernelSize ),
                                                  c( self$fusionKernelSize, self$fusionKernelSize, 1L, 1L ) )

          output <- tf$reshape( output, c( 1L,
                                           as.integer( newForegroundShape[1] * newForegroundShape[2] *
                                                         newForegroundShape[3] ),
                                           as.integer( newBackgroundShape[1] * newBackgroundShape[2] *
                                                         newBackgroundShape[3] ),
                                           1L ) )
          output <- tf$nn$conv2d( output, fusionKernel,
                                  strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
          output <- tf$reshape( output, c( 1L, newForegroundShape[1],
                                           newForegroundShape[2], newForegroundShape[3],
                                           newBackgroundShape[1], newBackgroundShape[2],
                                           newBackgroundShape[3] ) )

          output <- tf$transpose( output, c( 0L, 3L, 1L, 2L, 6L, 4L, 5L ) )
          output <- tf$reshape( output, c( 1L, newForegroundShape[1] *
                                             newForegroundShape[2] * newForegroundShape[3],
                                           newBackgroundShape[1] * newBackgroundShape[2] *
                                             newBackgroundShape[3], 1L ) )
          output <- tf$nn$conv2d( output, fusionKernel,
                                  strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
          output <- tf$reshape( output, c( 1L, newForegroundShape[3],
                                           newForegroundShape[1], newForegroundShape[2],
                                           newBackgroundShape[3], newBackgroundShape[1],
                                           newBackgroundShape[2] ) )

          output <- tf$transpose( output, c( 0L, 3L, 1L, 2L, 6L, 4L, 5L ) )
          output <- tf$reshape( output, c( 1L, newForegroundShape[1] *
                                             newForegroundShape[2] * newForegroundShape[3],
                                           newBackgroundShape[1] * newBackgroundShape[2] *
                                             newBackgroundShape[3], 1L ) )
          output <- tf$nn$conv2d( output, fusionKernel,
                                  strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
          output <- tf$reshape( output, c( 1L, newForegroundShape[2],
                                           newForegroundShape[3], newForegroundShape[1],
                                           newBackgroundShape[2], newBackgroundShape[3],
                                           newBackgroundShape[1] ) )

          output <- tf$transpose( output, c( 0L, 3L, 1L, 2L, 6L, 4L, 5L ) )
        }

        output <- tf$reshape( output, c( 1L, newForegroundShape[1],
                                         newForegroundShape[2], newForegroundShape[3],
                                         newBackgroundShape[1] * newBackgroundShape[2] *
                                           newBackgroundShape[3] ) )

        # softmax to match

        output <- output * maskData
        output <- tf$nn$softmax( output * 10.0, axis = 4L )
        output <- output * maskData

        bg <- backgroundGroups[[i]][1,,,,,]

        output <- tensorflow::tf$nn$conv2d_transpose(
          output, bg,
          tensorflow::tf$concat( list( list( 1L ), foregroundShape[2:5] ),
                                 axis = 0L ),
          strides = c( 1, self$dilationRate, self$dilationRate,
                       self$dilationRate, 1 ) ) / 4.0
        outputGroups[[i]] <- output
      }

      output <- tensorflow::tf$concat( outputGroups, axis = 0L )
      output$set_shape( foregroundShape )

      return( output )
    }
  )
)


