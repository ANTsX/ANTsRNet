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
#' @section Usage:
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
#' \dontrun{
#'
#' }
#'
#' @name inpaintingDeepFillModel
NULL

#' @export
InpaintingDeepFillModel <- R6::R6Class( "InpaintingDeepFillModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    dimensionality = 2,

    inputImageSize = c( 28, 28, 1 ),

    batchSize = 32,

    numberOfFiltersBaseLayer = 32L,

    tf = tensorflow::tf,

    initialize = function( inputImageSize, batchSize = 32, numberOfFiltersBaseLayer = 32L )
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

    generativeConvolutionLayer = function( model, numberOfFilters = 4L,
      kernelSize = 3L, stride = 1L, dilationRate = 1L, activation = 'elu',
      trainable = TRUE, name = '' )
      {
      output <- NULL
      if( self$dimensionality == 2 )
        {
        output <- model %>% layer_conv_2d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride,
            dilation_rate = dilationRate, activation = activation, padding = 'same',
            trainable = trainable, name = name )
        } else {
        output <- model %>% layer_conv_3d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride,
            dilation_rate = dilationRate, activation = activation, padding = 'same',
            trainable = trainable, name = name )
        }

      return( output )
      },

    discriminativeConvolutionLayer = function( model, numberOfFilters,
      kernelSize = 5L, stride = 2L, dilationRate = 1L, activation = 'leaky_relu',
      trainable = TRUE, name = '' )
      {
      output <- NULL
      if( self$dimensionality == 2 )
        {
        output <- model %>% layer_conv_2d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride, dilation_rate = dilationRate,
            activation = activation, padding = 'same', trainable = trainable, name = name )
        } else {
        output <- model %>% layer_conv_3d( filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride,
            dilation_rate = dilationRate, activation = activation, padding = 'same',
            trainable = trainable, name = name )
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
      resizedModel <- resampleTensor( model, newSize )
      output <- self$generativeConvolutionLayer( resizedModel,
        numberOfFilters = numberOfFilters, kernelSize = 3, stride = 1,
        trainable = trainable )

      return( output )
      },

    contextualAttentionLayer = function( foregroundTensor, backgroundTensor,
      mask, kernelSize = 3, stride = 1, dilationRate = 1, doFusion = TRUE )
      {
      if( self$dimensionality == 2 )
        {
        output <- layer_lambda( f = self$contextualAttentionLayer2D,
          arguments = list( backgroundTensor, mask, kernelSize, stride, dilationRate, doFusion ) )

        # output <- self$contextualAttentionLayer2D( foregroundTensor, backgroundTensor,
        #                   mask, kernelSize, stride, dilationRate )
        } else {
        output <- layer_lambda( f = self$contextualAttentionLayer3D,
          arguments = list( backgroundTensor, mask, kernelSize, stride, dilationRate, doFusion ) )

        # output <- self$contextualAttentionLayer3D( foregroundTensor, backgroundTensor,
        #                   mask, kernelSize, stride, dilationRate )
        }
      return( output )
      },

    contextualAttentionLayer2D = function( foregroundTensor, backgroundTensor,
      mask = NULL, kernelSize = 3L, stride = 1L, dilationRate = 1L, doFusion = TRUE )
      {
      K <- keras::backend()

      # Get tensor shapes

      foregroundShape <- unlist( K$int_shape( foregroundTensor ) )
      backgroundShape <- unlist( K$int_shape( backgroundTensor ) )

      maskShape <- NULL
      if( ! is.null( mask ) )
        {
        maskShape <- unlist( K$int_shape( mask ) )
        }

      # Extract patches from background and reshape to be
      #  c( batchSize, backgroundKernelSize, backgroundKernelSize, channelSize, height*width )

      backgroundKernelSize <- as.integer( 2 * dilationRate )
      stridexRate <- as.integer( stride * dilationRate )

      backgroundPatches <- self$tf$extract_image_patches( backgroundTensor,
                                   ksizes = c( 1, backgroundKernelSize, backgroundKernelSize, 1 ),
                                   strides = c( 1, stridexRate, stridexRate, 1 ),
                                   rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      backgroundPatches <- self$tf$reshape( backgroundPatches,
                                   c( self$tf$shape( backgroundTensor )[1], -1L,
                                      backgroundKernelSize, backgroundKernelSize,
                                      self$tf$shape( backgroundTensor )[4] ) )
      backgroundPatches <- self$tf$transpose( backgroundPatches, c( 0L, 2L, 3L, 4L, 1L ) )

      # Resample foreground, background, and mask

      newForegroundShape <- as.integer( foregroundShape[2:( self$dimensionality + 1 )] /
                                        dilationRate )
      resampledForegroundTensor <- resampleTensor( foregroundTensor, shape = newForegroundShape,
                                          interpolationType = 'nearestNeighbor' )

      newBackgroundShape <- as.integer( backgroundShape[2:( self$dimensionality + 1 )] /
                                        dilationRate )
      resampledBackgroundTensor <- resampleTensor( backgroundTensor, shape = newBackgroundShape,
                                          interpolationType = 'nearestNeighbor' )

      newMaskShape <- maskShape
      if( ! is.null( mask ) )
        {
        newMaskShape <- as.integer( maskShape[2:( self$dimensionality + 1 )] / dilationRate )
        mask <- resampleTensor( mask, shape = newMaskShape, interpolationType = 'nearestNeighbor' )
        }

      # Create resampled background patches
      resampledBackgroundPatches <- self$tf$extract_image_patches( resampledBackgroundTensor,
                                          ksizes = c( 1, kernelSize, kernelSize, 1 ),
                                          strides = c( 1, stride, stride, 1 ),
                                          rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      resampledBackgroundPatches <- self$tf$reshape( resampledBackgroundPatches,
                                       c( self$tf$shape( resampledBackgroundTensor )[1], -1L, kernelSize, kernelSize,
                                          self$tf$shape( resampledBackgroundTensor )[4] ) )
      resampledBackgroundPatches <- self$tf$transpose( resampledBackgroundPatches, c( 0L, 2L, 3L, 4L, 1L ) )

      # Process mask

      if( is.null( mask ) )
        {
        maskShape <- c( 1L, newBackgroundShape, 1L )
        mask = self$tf$zeros( maskShape )
        }

      maskPatches <- self$tf$extract_image_patches( mask,
                                   ksizes = c( 1, kernelSize, kernelSize, 1 ),
                                   strides = c( 1, stride, stride, 1 ),
                                   rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      maskPatches <- self$tf$reshape( maskPatches,
                                   c( 1L, -1L, kernelSize, kernelSize, 1L ) )
      maskPatches <- self$tf$transpose( maskPatches, c( 0L, 2L, 3L, 4L, 1L ) )

      maskPatches <- maskPatches[1,,,,]
      maskData <- self$tf$cast( self$tf$equal( self$tf$reduce_mean( maskPatches,
                  axis = c( 0L, 1L, 2L ), keep_dims = TRUE ), 0.0 ), self$tf$float32 )

      # Split into groups

      resampledForegroundGroups <- self$tf$split( resampledForegroundTensor,
                              K$int_shape( resampledForegroundTensor )[[1]], axis = 0L )
      backgroundGroups <- self$tf$split( backgroundPatches,
                              K$int_shape( backgroundTensor )[[1]], axis = 0L )
      resampledBackgroundGroups <- self$tf$split( resampledBackgroundPatches,
                              K$int_shape( resampledBackgroundTensor )[[1]], axis = 0L )

      numberOfIterations <- min( c( length( resampledForegroundGroups ) ),
                                 c( length( backgroundGroups ) ),
                                 c( length( resampledBackgroundGroups ) ) )

      fusionWeight <- self$tf$reshape( self$tf$eye( 3L ), c( 3L, 3L, 1L, 1L ) )

      yGroups <- list()
      # offsets <- list()
      for( i in seq_len( numberOfIterations ) )
        {
        rg <- resampledBackgroundGroups[[i]][1,,,,]
        rgNorm <- rg / self$tf$maximum( self$tf$sqrt( self$tf$reduce_sum(
                         self$tf$square( rg ), axis = c( 0L, 1L, 2L ) ) ), 1e-4 )
        fg <- resampledForegroundGroups[[i]]

        y <- self$tf$nn$conv2d( fg, rgNorm, strides = c( 1, 1, 1, 1 ), padding = 'SAME' )

        # fusion to encourage large patches

        if( doFusion == TRUE )
          {
          y <- tf$reshape( y, c( 1L,
                                 as.integer( newForegroundShape[1] * newForegroundShape[2] ),
                                 as.integer( newBackgroundShape[1] * newBackgroundShape[2] ),
                                 1L ) )
          y <- tf$nn$conv2d( y, fusionWeight, strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
          y <- tf$reshape( y, c( 1L, newForegroundShape[1], newForegroundShape[2],
                                 newBackgroundShape[1], newBackgroundShape[2] ) )
          y <- tf$transpose( y, c( 0L, 2L, 1L, 4L, 3L ) )

          y <- tf$reshape( y, c( 1L, newForegroundShape[1] * newForegroundShape[2],
                                 newBackgroundShape[1] * newBackgroundShape[2], 1L ) )
          y <- tf$nn$conv2d( y, fusionWeight, strides = c( 1, 1, 1, 1 ), padding = 'SAME' )

          y <- tf$reshape( y, c( 1L, newForegroundShape[1], newForegroundShape[2],
                                 newBackgroundShape[1], newBackgroundShape[2] ) )
          y <- tf$transpose( y, c( 0L, 2L, 1L, 4L, 3L ) )
          }

        y <- tf$reshape( y, c( 1L, newForegroundShape[1], newForegroundShape[2],
                               newBackgroundShape[1] * newBackgroundShape[2] ) )

        # softmax to match

        y <- y * maskData
        y <- tf$nn$softmax( y * 10.0, axis = 3L )
        y <- y * maskData

        # offset <- tf$argmax( y, axis = 3L, output_type = tf$int32 )
        # offset <- tf$stack( c( tf$cast(
        #    offset / tf$constant( newForegroundShape[2] ), dtype = tf$int32 ),
        #      offset %% newForegroundShape[2] ), axis = -1L )

        bg <- backgroundGroups[[i]][1,,,,]

        y <- self$tf$nn$conv2d_transpose( y, bg,
                self$tf$concat( list( list( 1L ),
                    foregroundShape[2:( self$dimensionality + 2 )] ), axis = 0L ),
                strides = c( 1, dilationRate, dilationRate, 1 ) ) / 4.0
        yGroups[[i]] <- y
        # offsets[[i]] <- offset
        }

      y <- self$tf$concat( yGroups, axis = 0L )
      y$set_shape( foregroundShape )

      # # calculate offsets

      # offsets <- tf$concat( offsets, axis = 0L )
      # offsets$set_shape( c( backgroundShape[1], newBackgroundShape, self$dimensionality ) )
      # dx = tf$tile( tf$reshape( tf$range( newBackgroundShape[1] ),
      #                           c( 1L, newBackgroundShape[1], 1L, 1L ) ),
      #                           c( backgroundShape[1], 1L, newBackgroundShape[2], 1L ) )
      # dy = tf$tile( tf$reshape( tf$range( newBackgroundShape[2] ),
      #                           c( 1L, 1L, newBackgroundShape[2], 1L ) ),
      #                           c( backgroundShape[1], newBackgroundShape[1], 1L, 1L ) )
      # offsets <- offsets - tf$concat( list( dx, dy ), axis = 3L )

      return( y )
      },

    buildNetwork = function( trainable = TRUE )
      {
      K <- keras::backend()

      imageInput <- layer_input( batch_shape = c( self$batchSize, self$inputImageSize ) )
      maskInput <- layer_input( batch_shape = c( 1, self$inputImageSize[1:self$dimensionality], 1 ) )

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

      maskedOnes <- NULL
      if( ! is.null( mask ) )
        {
        maskedOnes <- list( ones, maskInput ) %>% layer_lambda( f = function( inputs )
          { return( inputs[[1]] * inputs[[2]] ) } )
        } else {
        maskedOnes <- maskInput %>% layer_lambda( f = function( X ){ return( X + 0 ) } )
        }

      cat( "HERE A\n" )

      output <- layer_concatenate( list( output, ones, maskedOnes ),
                 axis = as.integer( self$dimensionality + 1 ) )
      cat( "HERE B\n" )

      # Stage 1

      cat( "HERE 0\n" )

      output <- self$generativeConvolutionLayer( output,     self$numberOfFiltersBaseLayer, 5L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      outputShape <- unlist( K$int_shape( output ) )[2:( self$dimensionality + 1 )]

      cat( "HERE 1\n" )
      resampledMaskInput <- maskInput %>% layer_lambda( f = resampleTensor( X ),
        arguments = list( shape = outputShape, interpolationType = 'nearestNeighbor' ) )
      cat( "HERE 2\n" )

      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 2L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 4L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 8L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 16L )

      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      output <- self$generativeDeconvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L )
      output <- self$generativeDeconvolutionLayer( output, self$numberOfFiltersBaseLayer )

      output <- self$generativeConvolutionLayer( output, as.integer( self$numberOfFiltersBaseLayer / 2 ), 3L, 1L )
      output <- self$generativeConvolutionLayer( output, 3L, 3L, 1L, activation = NULL )

      cat( "HERE 3\n" )
      output <- output %>% layer_lambda( function( X )
        { return( self$tf$clip_by_value( X, -1.0, 1.0 ) ) } )
      cat( "HERE 4\n" )

      modelStage1 <- keras_model( inputs = list( imageInput, maskInput ), outputs = output )

      # Stage 2

      cat( "HERE 4.5\n" )
      output <- output * mask + inputs * ( 1.0 - mask )
      cat( "HERE 5\n" )
      output$set_shape( inputs$get_shape()$as_list() )
      cat( "HERE 6\n" )

      # Conv branch

      outputNow <- layer_concatenate( list( output, ones, ones * mask ),
                                     axis = as.integer( self$dimensionality + 1 ) )
      output <- self$generativeConvolutionLayer( outputNow,  self$numberOfFiltersBaseLayer, 5, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,     self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      cat( "HERE 7\n" )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 2L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 4L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 8L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 16L )

      cat( "HERE 8\n" )

      outputHallu <- output

      # Attention branch

      output <- self$generativeConvolutionLayer( outputNow,  self$numberOfFiltersBaseLayer, 5, 1L, 1L )
      output <- self$generativeConvolutionLayer( output,     self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 2L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L,
                                           activation = 'relu' )

      output <- self$contextualAttentionLayer( output, output, resampledMaskInput, 3L, 1L, dilationRate = 2L )

      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )

      output <- layer_concatenate( list( outputHallu, output ),
                                  axis = as.integer( self$dimensionality + 1 ) )

      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 4 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeDeconvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer )
      output <- self$generativeConvolutionLayer( output, 2 * self$numberOfFiltersBaseLayer, 3L, 1L, 1L )
      output <- self$generativeDeconvolutionLayer( output,   self$numberOfFiltersBaseLayer )
      output <- self$generativeConvolutionLayer( output, as.integer( 0.5 * self$numberOfFiltersBaseLayer ), 3L, 1L, 1L )
      output <- self$generativeConvolutionLayer( output, 3L, 3L, 1L, 1L )

      output <- output %>% layer_lambda( function( X )
        { X <- self$tf$clip_by_value( X, -1.0, 1.0 ) } )
      modelStage2 <- keras_model( inputs = list( imageInput, maskInput ), outputs = output )


      return( list( modelStage1 = modelStage1, modelStage2 = modelStage2 ) )
      },

    buildLocalDiscriminator = function( model, reuse = FALSE, trainable = TRUE )
      {
      with( self$tf$variable_scope( 'localDiscriminator', reuse = reuse ) )
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

    buildGlobalDiscriminator = function( model, reuse = FALSE, trainable = TRUE )
      {
      with( self$tf$variable_scope( 'globalDiscriminator', reuse = reuse ) )
        {
        numberOfFilters <- 64
        numberOfLayers <- numberOfLayers

        localNumberOfFilters <- numberOfFilters * c( 1, 2, 4, 4 )
        for( i in seq_len( numberOfLayers ) )
          {
          model <- discriminativeConvolutionLayer( model,
            localNumberOfFilters[i], trainable = trainable )
          }
        model <- model %>% layer_flatten()
        }
      },

    buildCombinedDiscriminator = function( localBatch, globalBatch, reuse = FALSE, trainable = TRUE )
      {
      with( self$tf$variable_scope( 'discriminator', reuse = reuse ) )
        {
        localDiscriminator <- self$buildLocalDiscriminator( localBatch, reuse = reuse, trainable = trainable )
        globalDiscriminator <- self$buildGlobalDiscriminator( globalBatch, reuse = reuse, trainable = trainable )

        localDiscriminator <- localDiscriminator %>% layer_dense( 1 )
        globalDiscriminator <- globalDiscriminator %>% layer_dense( 1 )

        return( list( localDiscriminator, globalDiscriminator ) )
        }
      },

    train = function( X_train, numberOfEpochs = 200, batchSize = 32 )
      {
      cat( "HERE\n" )
      }

    )
  )


