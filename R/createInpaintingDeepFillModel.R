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

    tf = tensorflow::tf,

    initialize = function( dimensionality, inputImageSize, latentDimension, mask, numberOfFiltersBaseLayer = 32 )
      {
      self$dimensionality <- dimensionality
      self$inputImageSize <- inputImageSize
      self$latentDimension <- latentDimension
      self$mask <- mask
      self$numberOfFiltersBaseLayer <- numberOfFiltersBaseLayer
      },

    generativeConvolutionLayer = function( model, numberOfFilters = 4,
      kernelSize = 3, stride = 1, dilationRate = 1, activation = 'elu',
      trainable = TRUE, name = '' )
      {
      output <- NULL
      if( self$dimensionality == 2 )
        {
        output <- model %>% layer_conv_2d( x, number_of_filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride,
            dilation_rate = dilationRate, activation = activation, padding = 'same',
            trainable = trainable, name = name )
        } else {
        output <- model %>% layer_conv_3d( x, number_of_filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride,
            dilation_rate = dilationRate, activation = activation, padding = 'same',
            trainable = trainable, name = name )
        }

      return( output )
      },

    discriminativeConvolutionLayer = function( model, numberOfFilters,
      kernelSize = 5, stride = 2, dilationRate = 1, activation = 'leaky_relu',
      trainable = TRUE, name = '' )
      {
      output <- NULL
      if( self$dimensionality == 2 )
        {
        output <- model %>% layer_conv_2d( x, number_of_filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride, dilation_rate = dilationRate,
            activation = activation, padding = 'same', trainable = trainable, name = name )
        } else {
        output <- model %>% layer_conv_3d( x, number_of_filters = numberOfFilters,
            kernel_size = kernelSize, strides = stride,
            dilation_rate = dilationRate, activation = activation, padding = 'same',
            trainable = trainable, name = name )
        }
      return( output )
      },

    generativeDeconvolutionLayer = function( model, numberOfFilters = 4,
      trainable = TRUE, name = '' )
      {
      K <- keras::backend()

      shape <- unlist( K$int_shape( model ) )

      newSize <- as.integer( 2.0 * shape[2:( self$dimensionality + 1 )] )
      resizedModel <- resizeTensor( model, newSize )
      output <- generativeConvolutionLayer( resizedModel,
        numberOfFilters = numberOfFilters, kernelSize = 3, stride = 1,
        trainable = trainable )

      return( output )
      },

    contextualAttentionLayer = function( foregroundTensor, backgroundTensor,
      mask, kernelSize = 3, stride = 1, dilationRate = 1 )
      {
      if( dimensionality == 2 )
        {
        output <- contextualAttentionLayer2D( foregroundTensor, backgroundTensor,
                   mask, kernelSize, stride, dilationRate )
        } else {
        output <- contextualAttentionLayer3D( foregroundTensor, backgroundTensor,
                   mask, kernelSize, stride, dilationRate )
        }
      return( output )
      },

    contextualAttentionLayer2D = function( foregroundTensor, backgroundTensor,
      mask = NULL, kernelSize = 3, stride = 1, dilationRate = 1 )
      {
      K <- keras::backend()

      # Get tensor shapes

      foregroundShape <- unlist( K$int_shape( foregroundTensor ) )
      backgroundShape <- unlist( K$int_shape( backgroundTensor ) )
      maskShape <- NULL
      if( ! is.null( maskShape ) )
        {
        maskShape <- unlist( K$int_shape( mask ) )
        }

      if( ! all( foregroundShape == backgroundShape ) )
        {
        stop( "Error in contextual attention layer:  foregroundShape != backGroundShape." )
        }
      if( length( foregroundShape ) != 4 )
        {
        stop( "Error in contextual attention layer:  input tensor must be of rank 4." )
        }

      # Extract patches from background and reshape to be
      #  c( batchSize, kernelSize, kernelSize, channelSize, height*width )

      kernelSize <- as.integer( 2 * dilationRate )
      strideLength <- as.integer( stride * dilationRate )

      backgroundPatches <- self$tf$extract_image_patches( backgroundTensor,
                                   ksizes = c( 1, kernelSize, kernelSize, 1 ),
                                   strides = c( 1, strideLength, strideLength, 1 ),
                                   rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      backgroundPatches <- self$tf$reshape( backgroundPatches,
                                   c( backgroundShape[1], -1, kernelSize, kernelSize, backgroundShape[4] ) )
      backgroundPatches <- self$tf$transpose( backgroundPatches, c( 0, 2, 3, 4, 1 ) )

      # Resample foreground, background, and mask

      newForegroundShape <- as.integer( foregroundShape[2:( self$dimensionality + 1 )] /
                                        dilationRate )
      foregroundTensor <- resampleTensor( foregroundTensor, shape = newForegroundShape,
                                          interpolationType = 'nearestNeighbor' )

      newBackgroundShape <- as.integer( backgroundShape[2:( self$dimensionality + 1 )] /
                                        dilationRate )
      backgroundTensor <- resampleTensor( backgroundTensor, shape = newBackgroundShape,
                                          interpolationType = 'nearestNeighbor' )

      newMaskShape <- maskShape
      if( ! is.null( mask ) )
        {
        newMaskShape <- as.integer( maskShape[2:( self$dimensionality + 1 )] / dilationRate )
        mask <- resampleTensor( mask, shape = newMaskShape, interpolationType = 'nearestNeighbor' )
        }

      # Create resampled background patches

      resampledBackgroundPatches <- self$tf$extract_image_patches( backgroundTensor,
                                          ksizes = c( 1, kernelSize, kernelSize, 1 ),
                                          strides = c( 1, strideLength, strideLength, 1 ),
                                          rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      resampledBackgroundPatches <- self$tf$reshape( resampledBackgroundPatches,
                                           c( newBackgroundShape[1], -1, kernelSize, kernelSize, backgroundShape[4] ) )
      resampledBackgroundPatches <- self$tf$transpose( resampledBackgroundPatches, c( 0, 2, 3, 4, 1 ) )

      # Process mask

      if( is.null( mask ) )
        {
        maskShape <- c( 1, backgroundShape[2:( self$dimensionality + 1 )], 1 )
        mask = self$tf$zeros( maskShape )
        }

      maskPatches <- self$tf$extract_image_patches( mask,
                                   ksizes = c( 1, kernelSize, kernelSize, 1 ),
                                   strides = c( 1, strideLength, strideLength, 1 ),
                                   rates = c( 1, 1, 1, 1 ), padding = 'SAME' )
      maskPatches <- self$tf$reshape( maskPatches,
                                   c( 1, -1, kernelSize, kernelSize, 1 ) )
      maskPatches <- self$tf$transpose( maskPatches, c( 0, 2, 3, 4, 1 ) )

      maskData <- self$tf$cast( self$tf$equal( self$tf$reduce_mean( maskPatches[1,,,],
                  axis = c( 0L, 1L, 2L ), keep_dims = TRUE ), 0.0 ), self$tf$float32 )

      # Split into groups

      foregroundGroups <- self$tf$split( foregroundPatches, newForegroundShape[1], axis = 0L )
      backgroundGroups <- self$tf$split( backgroundPatches, newBackgroundShape[1], axis = 0L )
      resampledBackgroundGroups <- self$tf$split( resampledBackgroundPatches,
                                                  backgroundShape[1], axis = 0L )

      numberOfIterations <- min( c( length( foregroundGroups ) ),
                                 c( length( backgroundGroups ) ),
                                 c( length( resampledBackgroundGroups ) ) )

      fusionWeight <- self$tf$reshape( self$tf$eye( 3 ), c( 3, 3, 1, 1 ) )

      yGroups <- list()
      offsets <- list()
      for( i in seq_len( numberOfIterations ) )
        {
        rg <- resampledBackgroundGroups[[i]][1,,,]
        rgNorm <- rg / self$tf$maximum( self$tf$sqrt( self$tf$reduce_sum(
                         self$tf$square( rg ), axis = c( 0L, 1L, 2L ) ) ), 1e-4 )
        fg <- foregroundGroups[[i]]

        y <- self$tf$nn$conv2d( fg, rgNorm, strides = c( 1, 1, 1, 1 ), padding = 'SAME' )

        # fusion to encourage large patches

        y <- self$tf$reshape( y, c( 1, newForegroundShape[2] * newForegroundShape[3],
                                       newBackgroundShape[2] * newBackgroundShape[3], 1 ) )
        y <- self$tf$nn$conv2d( f, fusionWeight, strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
        y <- self$tf$reshape( y, c( 1, newForegroundShape[2], newForegroundShape[3],
                                       newBackgroundShape[2], newBackgroundShape[3] ) )
        y <- self$tf$transpose( y, c( 0, 2, 1, 4, 3 ) )
        y <- self$tf$reshape( y, c( 1, newForegroundShape[2] * newForegroundShape[3],
                                       newBackgroundShape[2] * newBackgroundShape[3], 1 ) )
        y <- self$tf$nn$conv2d( f, fusionWeight, strides = c( 1, 1, 1, 1 ), padding = 'SAME' )
        y <- self$tf$reshape( y, c( 1, newForegroundShape[2], newForegroundShape[3],
                                       newBackgroundShape[2], newBackgroundShape[3] ) )
        y <- self$tf$transpose( y, c( 0, 2, 1, 4, 3 ) )

        y <- self$tf$reshape( y, c( 1, newForegroundShape[2], newForegroundShape[3],
                                       newBackgroundShape[2] * newBackgroundShape[3] ) )

        # softmax to match

        y <- y * maskData
        y <- self$tf$nn$softmax( yi * 10.0, axis = 3L )
        y <- y * maskData

        offset <- self$tf$argmax( y, axis = 3L, output_type = self$tf$int32 )
        offset <- self$tf$stack( c( as.integer( offset / newForegroundShape[3] ),
                                    offset %% newForegroundShape[3] ), axis = -1L )

        bg <- backgroundGroups[[[i]]][1,,,]

        y <- self$tf$nn$conv2d_transpose( y, bg,
                self$tf$concat( list( list( 1 ), foregroundShape[2:( dimensionality + 2 )] ), axis = 0L ),
                strides = c( 1, dilationRate, dilationRate, 1 ) ) / 4.0
        yGroups[[i]] <- y
        offsets[[i]] <- offset
        }

      y <- self$tf$concat( y, axis = 0L )
      y$set_shape( foregroundShape )

      # calculate offsets

      offsets <- self$tf$concat( offsets, axis = 0L )
      offsets$set_shape( c( newBackgroundShape[1:( dimensionality + 1 )], dimensionality ) )

      height = self$tf$tile( self$tf$reshape( self$tf$range( newBackgroundShape[2] ),
                                              c( 1L, newBackgroundShape[2], 1L, 1L ) ),
                             c( newBackgroundShape[1], 1L, newBackgroundShape[3], 1L ) )
      width = self$tf$tile( self$tf$reshape( self$tf$range( newBackgroundShape[3] ),
                                              c( 1L, 1L, newBackgroundShape[3], 1L ) ),
                             c( newBackgroundShape[1], newBackgroundShape[2], 1L, 1L ) )
      offsets <- offsets - self$tf$concat( list( height, width ), axis = 3L )

      return( list( y, offsets ) )
      }

    contextualAttentionLayer3D = function( foregroundTensor, backgroundTensor,
      mask, kernelSize = 3, stride = 1, dilationRate = 1 )
      {
      K <- keras::backend()

      # Get tensor shapes

      foregroundShape <- unlist( K$int_shape( foregroundTensor ) )
      backgroundShape <- unlist( K$int_shape( backgroundTensor ) )

      if( ! all( foregroundShape == backgroundShape ) )
        {
        stop( "Error in contextual attention layer:  foregroundShape != backGroundShape." )
        }
      if( length( foregroundShape ) != 5 )
        {
        stop( "Error in contextual attention layer:  input tensors must be of rank 5." )
        }

      # Extract patches from background and reshape to be
      #  c( batchSize, kernelSize, kernelSize, channelSize, height*width*depth )

      kernelSize <- as.integer( 2 * dilationRate )
      strideLength <- as.integer( stride * dilationRate )

      backgroundPatches <- self$tf$extract_volume_patches( backgroundTensor,
                                   ksizes = c( 1, kernelSize, kernelSize, kernelSize, 1 ),
                                   strides = c( 1, strideLength, strideLength, strideLength, 1 ),
                                   padding = 'SAME' )
      backgroundPatches <- self$tf$reshape( backgroundPatches,
                                   c( backgroundShape[1], -1, kernelSize, kernelSize, kernelSize, backgroundShape[5] ) )
      backgroundPatches <- self$tf$transpose( backgroundPatches, c( 0, 2, 3, 4, 5, 1 ) )

      # Resample foreground, background, and mask

      newForegroundShape <- as.integer( foregroundShape[2:( self$dimensionality + 1 )] /
                                        dilationRate )
      foregroundTensor <- resampleTensor( foregroundTensor, shape = newForegroundShape,
                                          interpolationType = 'nearestNeighbor' )

      newBackgroundShape <- as.integer( backgroundShape[2:( self$dimensionality + 1 )] /
                                        dilationRate )
      backgroundTensor <- resampleTensor( backgroundTensor, shape = newBackgroundShape,
                                          interpolationType = 'nearestNeighbor' )

      newMaskShape <- maskShape
      if( ! is.null( mask ) )
        {
        newMaskShape <- as.integer( maskShape[2:( self$dimensionality + 1 )] / dilationRate )
        mask <- resampleTensor( mask, shape = newMaskShape, interpolationType = 'nearestNeighbor' )
        }

      # Create resampled background patches

      resampledBackgroundPatches <- self$tf$extract_volume_patches( backgroundTensor,
                                          ksizes = c( 1, kernelSize, kernelSize, kernelSize, 1 ),
                                          strides = c( 1, strideLength, strideLength, strideLength, 1 ),
                                          padding = 'SAME' )
      resampledBackgroundPatches <- self$tf$reshape( resampledBackgroundPatches,
                                           c( newBackgroundShape[1], -1, kernelSize, kernelSize, kernelSize, backgroundShape[5] ) )
      resampledBackgroundPatches <- self$tf$transpose( resampledBackgroundPatches, c( 0, 2, 3, 4, 5, 1 ) )

      # Process mask

      if( is.null( mask ) )
        {
        maskShape <- c( 1, backgroundShape[2:( self$dimensionality + 1 )], 1 )
        mask = self$tf$zeros( maskShape )
        }

      maskPatches <- self$tf$extract_image_patches( mask,
                                   ksizes = c( 1, kernelSize, kernelSize, kernelSize, 1 ),
                                   strides = c( 1, strideLength, strideLength, strideLength, 1 ),
                                   rates = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )
      maskPatches <- self$tf$reshape( maskPatches,
                                   c( 1, -1, kernelSize, kernelSize, kernelSize, 1 ) )
      maskPatches <- self$tf$transpose( maskPatches, c( 0, 2, 3, 4, 5, 1 ) )

      maskData <- self$tf$cast( self$tf$equal( self$tf$reduce_mean( maskPatches[1,,,,],
                  axis = c( 0L, 1L, 2L, 3L ), keep_dims = TRUE ), 0.0 ), self$tf$float32 )

      # Split into groups

      foregroundGroups <- self$tf$split( foregroundPatches, newForegroundShape[1], axis = 0L )
      backgroundGroups <- self$tf$split( backgroundPatches, newBackgroundShape[1], axis = 0L )
      resampledBackgroundGroups <- self$tf$split( resampledBackgroundPatches,
                                                  backgroundShape[1], axis = 0L )

      numberOfIterations <- min( c( length( foregroundGroups ) ),
                                 c( length( backgroundGroups ) ),
                                 c( length( resampledBackgroundGroups ) ) )

      fusionWeight <- self$tf$reshape( self$tf$eye( 3 ), c( 3, 3, 3, 1, 1 ) )

      yGroups <- list()
      offsets <- list()
      for( i in seq_len( numberOfIterations ) )
        {
        rg <- resampledBackgroundGroups[[i]][1,,,,]
        rgNorm <- rg / self$tf$maximum( self$tf$sqrt( self$tf$reduce_sum(
                         self$tf$square( rg ), axis = c( 0L, 1L, 2L, 3L ) ) ), 1e-4 )
        fg <- foregroundGroups[[i]]

        y <- self$tf$nn$conv3d( fg, rgNorm, strides = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )

        # fusion to encourage large patches

        y <- self$tf$reshape( y, c( 1, newForegroundShape[2] * newForegroundShape[3] * newForegroundShape[4],
                                       newBackgroundShape[2] * newBackgroundShape[3] * newBackgroundShape[4], 1 ) )
        y <- self$tf$nn$conv3d( f, fusionWeight, strides = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )
        y <- self$tf$reshape( y, c( 1, newForegroundShape[2], newForegroundShape[3], newForegroundShape[4],
                                       newBackgroundShape[2], newBackgroundShape[3], newBackgroundShape[4] ) )
        y <- self$tf$transpose( y, c( 0, 2, 1, 3, 5, 4, 6 ) )
        y <- self$tf$reshape( y, c( 1, newForegroundShape[2] * newForegroundShape[3] * newForegroundShape[4],
                                       newBackgroundShape[2] * newBackgroundShape[3] * newBackgroundShape[4], 1 ) )
        y <- self$tf$nn$conv3d( f, fusionWeight, strides = c( 1, 1, 1, 1, 1 ), padding = 'SAME' )
        y <- self$tf$reshape( y, c( 1, newForegroundShape[2], newForegroundShape[3], newForegroundShape[4]
                                       newBackgroundShape[2], newBackgroundShape[3], newBackgroundShape[4] ) )
        y <- self$tf$transpose( y, c( 0, 2, 1, 3, 5, 4, 6 ) )

        y <- self$tf$reshape( y, c( 1, newForegroundShape[2], newForegroundShape[3], newForegroundShape[4],
                                       newBackgroundShape[2] * newBackgroundShape[3] * newBackgroundShape[4] ) )

        # softmax to match

        y <- y * maskData
        y <- self$tf$nn$softmax( yi * 10.0, axis = 4L )
        y <- y * maskData

        offset <- self$tf$argmax( y, axis = 4L, output_type = self$tf$int32 )
        offset <- self$tf$stack( c( as.integer( offset / newForegroundShape[4] ),
                                    offset %% newForegroundShape[4] ), axis = -1L )

        bg <- backgroundGroups[[[i]]][1,,,,]

        y <- self$tf$nn$conv3d_transpose( y, bg,
                self$tf$concat( list( list( 1 ), foregroundShape[2:( dimensionality + 2 )] ), axis = 0L ),
                strides = c( 1, dilationRate, dilationRate, dilationRate ) ) / 4.0
        yGroups[[i]] <- y
        offsets[[i]] <- offset
        }

      y <- self$tf$concat( y, axis = 0L )
      y$set_shape( foregroundShape )

      # calculate offsets

      offsets <- self$tf$concat( offsets, axis = 0L )
      offsets$set_shape( c( newBackgroundShape[1:( dimensionality + 1 )], dimensionality ) )

      height = self$tf$tile( self$tf$reshape( self$tf$range( newBackgroundShape[2] ),
                                              c( 1L, newBackgroundShape[2], 1L, 1L, 1L ) ),
                             c( newBackgroundShape[1], 1L, newBackgroundShape[3], newBackgroundShape[4], 1L ) )
      width = self$tf$tile( self$tf$reshape( self$tf$range( newBackgroundShape[3] ),
                                              c( 1L, 1L, newBackgroundShape[3], 1L, 1L ) ),
                             c( newBackgroundShape[1], newBackgroundShape[2], 1L, newBackgroundShape[4], 1L ) )
      depth = self$tf$tile( self$tf$reshape( self$tf$range( newBackgroundShape[4] ),
                                              c( 1L, 1L, 1L, newBackgroundShape[4], 1L ) ),
                             c( newBackgroundShape[1], newBackgroundShape[2], newBackgroundShape[3], 1L, 1L ) )
      offsets <- offsets - self$tf$concat( list( height, width, depth ), axis = 4L )

      return( list( y, offsets ) )
      }

    buildNetwork = function()
      {
      K <- keras::backend()

      inputs <- layer_input( shape = inputImageSize )

      model <- inputs
      if( self$dimensionality == 2 )
        {
        ones <- K$ones_like( model )[,,,1, drop = FALSE]
        } else {
        ones <- K$ones_like( model )[,,,,1, drop = FALSE]
        }

      model <- layer_concatenate( list( model, ones, ones * mask ),
                 axis = as.integer( dimensionality + 1 ) )

      # Stage 1

      model <- generativeConvolutionLayer( model,     self$numberOfFiltersBaseLayer, 5, 1, 1 )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 2, 1 )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 2, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )

      resampledMask <- resizeTensorLike( mask, model, 'nearestNeighbor' )

      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 2 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 4 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 8 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 16 )

      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )

      model <- generativeDeconvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 1 )
      model <- generativeDeconvolutionLayer( model, self$numberOfFiltersBaseLayer )

      model <- generativeConvolutionLayer( model, as.integer( self$numberOfFiltersBaseLayer / 2 ), 3, 1 )
      model <- generativeConvolutionLayer( model, 3, 3, 1, activation = NULL )

      model <- self$tf$clip_by_value( model, -1.0, 1.0 )
      modelStage1 <- model

      # Stage 2

      model <- model * mask + inputs * ( 1.0 - mask )
      model$set_shape( inputs$get_shape()$as_list() )

      # Conv branch

      modelNow <- layer_concatenate( list( model, ones, ones * mask ),
                                     axis = as.integer( dimensionality + 1 ) )
      model <- generativeConvolutionLayer( modelNow,  self$numberOfFiltersBaseLayer, 5, 1, 1 )
      model <- generativeConvolutionLayer( model,     self$numberOfFiltersBaseLayer, 3, 2, 1 )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 2, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )

      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 2 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 4 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 8 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 16 )

      modelHallu <- model

      # Attention branch

      model <- generativeConvolutionLayer( modelNow,  self$numberOfFiltersBaseLayer, 5, 1, 1 )
      model <- generativeConvolutionLayer( model,     self$numberOfFiltersBaseLayer, 3, 2, 1 )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 2, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1,
                                           activation = 'relu' )

      attention <- contextualAttentionLayer( model, model, resampledMask, 3, 1, rate = 2 )
      model <- attention$model
      offsets <- attention$offsets

      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )

      model <- layer_concatenate( list( modelHallu, model ),
                                  axis = as.integer( dimensionality + 1 ) )

      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 4 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeDeconvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer )
      model <- generativeConvolutionLayer( model, 2 * self$numberOfFiltersBaseLayer, 3, 1, 1 )
      model <- generativeDeconvolutionLayer( model,   self$numberOfFiltersBaseLayer )
      model <- generativeConvolutionLayer( model, as.integer( 0.5 * self$numberOfFiltersBaseLayer ), 3, 1, 1 )
      model <- generativeConvolutionLayer( model, 3, 3, 1, 1 )

      model <- self$tf$clip_by_value( model, -1.0, 1.0 )
      modelStage2 <- model

      return( list( modelStage1, modelStage2, offsets ) )
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

    compile = function()

    )
  )


