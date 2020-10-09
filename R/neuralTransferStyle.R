#' Neural transfer style
#'
#' The popular neural style transfer described here:
#'
#'     https://arxiv.org/abs/1508.06576 and https://arxiv.org/abs/1605.04603
#'
#' and taken from François Chollet's implementation
#'
#'     https://keras.io/examples/generative/neural_style_transfer/
#'
#' and titu1994's modifications:
#'
#'     https://github.com/titu1994/Neural-Style-Transfer
#'
#' in order to possibly modify and experiment with medical images.
#'
#' @param contentImage ANTs image (1 or 3-component).  Content (or base) image.
#' @param styleImages ANTsImage or list of ANTsImages as the style (or reference)
#' image.
#' @param initialCombinationImage ANTsImage (1 or 3-component).  Starting point
#' for the optimization.  Allows one to start from the output from a previous
#' run.  Otherwise, start from the content image. Note that the original paper
#' starts with a noise image.
#' @param numberOfIterations Number of gradient steps taken during optimization.
#' @param learningRate Parameter for Adam optimization.
#' @param totalVariationWeight A penalty on the regularization term to keep the
#' features of the output image locally coherent.
#' @param contentWeight Weight of the content layers in the optimization function.
#' @param styleImageWeights float or vector of floats.  Weights of the style term
#' in the optimization function for each style image.  Can either specify a
#' single scalar to be used for all the images or one for each image.  The
#' style term computes the sum of the L2 norm between the Gram matrices of the
#' different layers (using ImageNet-trained VGG) of the style and content images.
#' @param contentLayerNames vector of strings. Names of VGG layers from which
#' to compute the content loss.
#' @param styleLayerNames vector of strings. Names of VGG layers from which to
#' compute the style loss.  If "all", the layers used are c('block1_conv1',
#' 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
#' 'block3_conv3', 'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3',
#' 'block4_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4').
#' This is a proposed improvement from https://arxiv.org/abs/1605.04603.  In the
#' original implementation, the layers used are: c('block1_conv1', 'block2_conv1',
#' block3_conv1', 'block4_conv1', 'block5_conv1').
#' @param contentMask an ANTsImage mask to specify the region for content consideration.
#' @param styleMasks ANTsImage masks to specify the region for style consideration.
#' @param useShiftedActivations boolean to determine whether or not to use shifted
#' activations in calculating the Gram matrix (improvement mentioned in
#' https://arxiv.org/abs/1605.04603).
#' @param useChainedInference boolean corresponding to another proposed improvement
#' from https://arxiv.org/abs/1605.04603.
#' @param verbose boolean to print progress to the screen.
#' @param outputPrefix If specified, outputs a png image to disk at each iteration.
#' @return ANTs 3-component image.
#' @author Tustison, NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#'
#' }
#' @export
neuralStyleTransfer <- function(contentImage, styleImages,
  initialCombinationImage = NULL, numberOfIterations = 10,
  learningRate = 1.0, totalVariationWeight = 8.5e-5, contentWeight = 0.025,
  styleImageWeights = 1.0, contentLayerNames = c( 'block5_conv2' ),
  styleLayerNames = "all", contentMask = NULL, styleMasks = NULL,
  useShiftedActivations = TRUE, useChainedInference = TRUE,
  verbose = FALSE, outputPrefix = NULL )
{

  K <- keras::backend()
  tf <- tensorflow::tf

  preprocessAntsImage <- function( image, doScaleAndCenter = TRUE )
    {
    imageArray <- array( data = 0, dim = c( 1, dim( image ), 3 ) )
    if( image@components == 1 )
      {
      imageArray[1,,,1] <- as.array( image )
      imageArray[1,,,2] <- as.array( image )
      imageArray[1,,,3] <- as.array( image )
      } else if( image@components == 3 ) {
      imageChannels <- splitChannels( image )
      imageArray[1,,,1] <- as.array( imageChannels[1] )
      imageArray[1,,,2] <- as.array( imageChannels[2] )
      imageArray[1,,,3] <- as.array( imageChannels[3] )
      } else {
      stop( "Unexpected number of components." )
      }

    if( doScaleAndCenter == TRUE )
      {
      for( i in seq.int( 3 ) )
        {
        imageArray[1,,,i] <- ( imageArray[1,,,i] - min( imageArray[1,,,i] ) ) /
          ( max( imageArray[1,,,i] ) - min( imageArray[1,,,i] ) )
        }
      imageArray <- imageArray * 255
      # RGB -> BGR
      imageArray <- imageArray[,,,rev( seq.int( 3 ) ), drop = FALSE]
      imageArray[1,,,1] <- imageArray[1,,,1] - 103.939
      imageArray[1,,,2] <- imageArray[1,,,2] - 116.779
      imageArray[1,,,3] <- imageArray[1,,,3] - 123.68
      }
    return( imageArray )
    }

  postProcessArray <- function( imageArray, referenceImage )
    {
    imageArray <- drop( imageArray )
    imageArray[,,1] <- imageArray[,,1] + 103.939
    imageArray[,,2] <- imageArray[,,2] + 116.779
    imageArray[,,3] <- imageArray[,,3] + 123.68
    # BGR -> RGB
    imageArray <- imageArray[,,,rev( seq.int( 3 ) ), drop = FALSE]
    imageArray[imageArray < 0] <- 0
    imageArray[imageArray > 255] <- 255

    image <- as.antsImage( imageArray, reference = referenceImage,
                          components = TRUE )
    return( image )
    }

  gramMatrix <- function( x, shiftedActivations = FALSE )
    {
    F <- K$batch_flatten( K$permute_dimensions( x, c( 2L, 0L, 1L ) ) )
    if( shiftedActivations )
      {
      F <- F - 1
      }
    gram <- K$dot( F, K$transpose( F ) )
    return( gram )
    }

  processMask <- function( mask, shape )
    {
    maskProcessed <- tf$image$resize( mask, size = c( shape[0], shape[1] ),
      method = tf$image$ResizeMethod$NEAREST_NEIGHBOR )
    maskProcessedTensor <- array( data = maskProcessed, dim = c( dim( mask ), shape[2] ) )
    for( i in range( shape[2] ) )
        maskProcessedTensor[,,i] = maskProcessed[,,0]
    return( maskProcessedTensor )
    }

  styleLoss <- function( styleFeatures, combinationFeatures, imageShape, styleMask = NULL, contentMask = NULL )
    {
    if( ! is.null( contentMask ) )
      {
      maskTensor <- K$variable( processMask( contentMask, combinationFeatures$shape ) )
      combinationFeatures <- combinationFeatures * K$stop_gradient( maskTensor )
      rm( maskTensor )
      }

    if( ! is.null( styleMask ) )
      {
      maskTensor <- K$variable( processMask( styleMask, styleFeatures$shape ) )
      styleFeatures <- styleFeatures * K$stop_gradient( maskTensor )
      if( ! is.null( contentMask ) )
        {
        combinationFeatures <- combinationFeatures * K$stop_gradient( maskTensor )
        }
      rm( maskTensor )
      }

    styleGram <- gramMatrix( styleFeatures, useShiftedActivations )
    contentGram <- gramMatrix( combinationFeatures, useShiftedActivations )
    size <- imageShape[0] * imageShape[1]
    numberOfChannels <- 3
    loss <- tf$reduce_sum( tf$square( styleGram - contentGram ) ) /
      ( 4.0 * numberOfChannels^2 * size^2 )
    return( loss )
    }

  contentLoss <- function( contentFeatures, combinationFeatures )
    {
    loss <- tf$reduce_sum( tf$square( contentFeature - combinationFeatures ) )
    return( loss )
    }

  totalVariationLoss <- function( x )
    {
    shape <- x$shape
    a <- tf$square( x[, 1:( shape[1] - 1 ), 1:( shape[2] - 1 ),] - x[, 2:shape[1], 1:( shape[2] - 1 ),] )
    b <- tf$square( x[, 1:( shape[1] - 1 ), 1:( shape[2] - 1 ),] - x[, 1:( shape[1] - 1 ), 2:shape[2],] )
    loss <- tf$reduce_sum( tf$pow( a + b, 1.25 ) )
    }

  computeTotalLoss <- function( contentArray, styleArrayList, combinationTensor,
                               featureModel, contentLayerNames, styleLayerIndices,
                               imageShape, contentMaskTensor = NULL, styleMaskTensorList = NULL )
    {
    numberOfStyleImages <- length( styleArrayList )

    inputArray <- list()
    inputArrays[[1]] <- contentArray
    for( i in seq.int( numberOfStyleImages ) )
      {
      inputArrays[[i + 1]] <- styleArrayList[[i]]
      }
    inputArrays[[2 + numberOfStyleImages]] <- combinationTensor
    inputTensor <- tf$concat( inputArrays, axis = 0L )

    features <- featureModel( inputTensor )

    totalLoss <- tf$zeros( shape = list() )

    # content loss
    for( i in seq.int( length( contentLayerNames ) ) )
      {
      layerFeatures <- features[[contentLayerNames[i]]]
      contentFeatures <- layerFeatures[1,,,]
      combinationFeatures <- layerFeatures[3,,,]
      totalLoss <- totalLoss + contentLoss( contentFeatures, combinationFeatures ) *
        contentWeight / length( contentLayerNames )
      }

    # style loss
    if( useChainedInference )
      {
      for( i in seq.int( length( styleLayerIndices ) - 1 ) )
        {
        layerFeatures <- features[styleLayerIndices[i]][[1]]
        styleFeatures <- layerFeatures[2:( numberOfStyleImages + 1 ),,,]
        combinationFeatures <- layerFeatures[( numberOfStyleImages + 2 ),,,]
        loss <- list()
        for( j in seq.int( numberOfStyleImages ) )
          {
          if( is.null( styleMaskTensorList ) )
            {
            loss[[j]] <- styleLoss( styleFeatures[j], combinationFeatures, imageShape,
                                    styleMask = NULL, contentMask = contentMaskTensor )
            } else {
            loss[[j]] <- styleLoss( styleFeatures[j], combinationFeatures, imageShape,
                                    styleMask = styleMaskTensorList[[j]], contentMask = contentMaskTensor )
            }
          }

        layerFeatures = features[styleLayerIndices[i+1]][[1]]
        styleFeatures = layerFeatures[2:( numberOfStyleImages + 1 ),,,]
        combinationFeatures = layerFeatures[( numberOfStyleImages + 2 ),,,]
        lossP1 <- list()
        for( j in seq.int( numberOfStyleImages ) )
          {
          if( is.null( styleMaskTensorList ) )
            {
            lossP1[[j]] <- styleLoss( styleLoss( styleFeatures[j,,,,drop = FALSE], combinationFeatures, imageShape,
                                      styleMask = NULL, contentMask = contentMaskTensor ) )
            } else {
            lossP1[[j]] <- styleLoss( styleLoss( styleFeatures[j,,,,drop = FALSE], combinationFeatures, imageShape,
                                      styleMask = styleMaskTensorList[i], contentMask = contentMaskTensor ) )
            }
          }

        for( j in seq.int( numberOfStyleImages ) )
          {
          lossDifference <- loss[j] - lossP1[j]
          totalLoss <- totalLoss + styleImageWeights[j] * lossDifference / ( 2^( length( styleLayerNames ) - ( i + 1 ) ) )
          }
        }
      } else {
      for( i in seq.int( length( styleLayerIndices ) ) )
        {
        layerFeatures <- features[styleLayerIndices[i]][[1]]
        styleFeatures <- layerFeatures[2:( numberOfStyleImages + 1 ),,,]
        combinationFeatures <- layerFeatures[( numberOfStyleImages + 2 ),,,]
        loss <- list()
        for( j in seq.int( numberOfStyleImages ) )
          {
          if( is.null( styleMaskTensorList ) )
            {
            loss[[j]] <- styleLoss( styleFeatures[j], combinationFeatures, imageShape,
                                    styleMask = NULL, contentMask = contentMaskTensor )
            } else {
            loss[[j]] <- styleLoss( styleFeatures[j], combinationFeatures, imageShape,
                                    styleMask = styleMaskTensorList[[j]], contentMask = contentMaskTensor )
            }
          }
        for( j in seq.int( numberOfStyleImages ) )
          {
          totalLoss <- totalLoss + ( loss[[j]] * styleImageWeights[j] / length( styleLayerIndices ) )
          }
        }
      }
    totalLoss <- totalLoss + totalVariationWeight + totalVariationLoss( combinationTensor )
    return( totalLoss )
    }

  computeLossAndGradients <- function( contentArray, styleArrayList, combinationTensor,
                featureModel, contentLayerNames, styleLayerIndices, imageShape,
                contentMaskTensor, styleMaskTensorList )
    {
    with( tf$GradientTape() %as% tape,
      {
      loss <- computeTotalLoss( contentArray, styleArrayList, combinationTensor,
                    featureModel, contentLayerNames, styleLayerIndices, imageShape, contentMaskTensor,
                    styleMaskTensorList )
      } )
    gradients <- tape$gradient( loss, combinationTensor )
    return( list( loss = loss, gradients = gradients) )
    }


  numberOfStyleImages <- 1
  if( is.list( styleImages ) )
    {
    numberOfStyleImages <- length( styleImages )
    }

  styleImageList <- list()
  if( numberOfStyleImages == 1 )
    {
    styleImageList[[1]] <- styleImages
    } else {
    styleImageList <- styleImages
    }

  for( i in seq.int( numberOfStyleImages ) )
    {
    if( styleImageList[[i]]@dimension != 2 )
      {
      stop( "Input style images must be 2-D." )
      }
    if( any( dim( styleImageList[[i]] ) != dim( contentImage ) ) )
      {
      stop( "Input images must have matching dimensions/shapes." )
      }
    }

  numberOfStyleMasks <- 0
  styleMaskTensorList <- NULL
  if( ! is.null( styleMasks ) )
    {
    numberOfStyleMasks <- 1
    if( is.list( styleMasks ) )
      {
      numberOfStyleMasks <- length( styleMasks )
      }

    styleMaskTensorList <- list()
    if( numberOfStyleMasks == 1 )
      {
      styleMaskArray <- as.array( thresholdImage( styleMasks, 0, 0, 0, 1 ) )
      styleMaskTensor <- array( data = styleMaskArray, dim = c( dim( styleMaskArray ), 1 ) )
      styleMaskTensorList[[1]] <- styleMaskTensor
      } else {
      for( i in seq.int( length( styleMasks ) ) )
        {
        styleMaskArray <- as.array( thresholdImage( styleMasks[[i]], 0, 0, 0, 1 ) )
        styleMaskTensor <- array( data = styleMaskArray, dim = c( dim( styleMaskArray ), 1 ) )
        styleMaskTensorList[[i]] <- styleMaskTensor
        }
      }
    }

  if( numberOfStyleMasks > 0 && numberOfStyleImages != numberOfStyleMasks )
    {
    stop( "The number of style images/masks are not the same." )
    }

  if( is.numeric( styleImageWeights ) )
    {
    styleImageWeights <- rep( styleImageWeights, length( styleImageList ) )
    } else {
    if( length( styleImageWeights ) == 1 )
      {
      styleImageWeights <- rep( styleImageWeights[1], length( styleImageList ) )
      } else if( length( styleImageWeights ) != length( styleImageList ) ) {
      stop( "Length of style weights must be 1 or the number of style images." )
      }
    }

  if( contentImage@dimension != 2 )
    {
    stop( "Input content image must be 2-D." )
    }

  contentMaskTensor <- NULL
  if( ! is.null( contentMask ) )
    {
    contentMaskArray <- as.array( thresholdImage( contentMask, 0, 0, 0, 1 ) )
    contentMaskTensor <- array( data = contentMaskArray, dim = c( dim( contentMaskArray ), 1 ) )
    }

  if( styleLayerNames == "all" )
    {
    styleLayerNames <- c( 'block1_conv1', 'block1_conv2', 'block2_conv1',
          'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
          'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3',
          'block4_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3',
          'block5_conv4')
    }

  model <- tf$keras$applications$VGG19( weights = "imagenet", include_top = FALSE )

  styleLayerIndices <- c()
  count <- 1
  for( i in seq.int( length( model$layers ) ) )
    {
    index <- which( model$layers[[i]]$name %in% styleLayerNames )
    if( length( index ) == 0 )
      {
      next
      }
    styleLayerIndices[count] <- i
    count <- count + 1
    }
  if( length( styleLayerIndices ) != length( styleLayerNames ) )
    {
    stop( "Style layer names don't match model." )
    }

  outputsList <- list()
  for( i in seq.int( model$layers ) )
    {
    outputsList[[i]] <- model$layers[[i]]$output
    }

  featureModel <- tf$keras$Model( inputs = model$inputs, outputs = outputsList )

  # Preprocess data
  contentArray <- preprocessAntsImage( contentImage )
  styleArrayList <- list()
  for( i in seq.int( numberOfStyleImages ) )
    {
    styleArrayList <- preprocessAntsImage( styleImageList[[i]] )
    }

  imageShape <- c( dim( contentArray ), 3 )

  combinationTensor <- NULL
  if( is.null( initialCombinationImage ) )
    {
    combinationTensor <- tf$Variable( array( data = contentArray, dim = dim( contentArray ) ) )
    } else {
    initialCombinationTensor <- preprocessAntsImage( initialCombinationImage, doScaleAndCenter = FALSE )
    combinationTensor <- tf$Variable( initialCombinationTensor )
    }

  if( any( imageShape != dim( combinationTensor, 3 ) ) )
    {
    stop( "Initial combination image size does not match content image." )
    }

  optimizer <- tf$optimizers$Adam( learning_rate = learningRate, beta_1 = 0.99, epsilon = 0.1 )

  for( i in seq.int( numberOfIterations ) )
    {
    startTime <- Sys.time()
    lossAndGradients <- computeLossAndGradients( contentArray, styleArrayList,
                          combinationTensor, featureModel, contentLayerNames,
                          styleLayerIndices, imageShape, contentMaskTensor,
                          styleMaskTensorList )
    endTime <- Sys.time()
    if( verbose == TRUE )
      {
      cat( "Iteration ", i, " of ", numberOfIterations, ": total loss = ", lossAndGradients$loss,
           " (elapsed time = ", endTime - startTime, "s)",  sep = "" )
      }
    optimizer$apply_gradients( list( lossAndGradients$gradients, combinationTensor ) )

    if( ! is.null( outputPrefix ) )
      {
      combinationArray <- array( combinationTensor )
      combinationImage <- postProcessArray( combinationArray, contentImage )
      combinationRgb <- combinationImage
      antsImageWrite( combinationRgb, paste0( outputPrefix, "_iteration", i, ".png" ) )
      }
    }

  combinationArray <- array( combinationTensor )
  combinationImage <- postProcessArray( combinationArray, contentImage )
  return( combinationImage )
}