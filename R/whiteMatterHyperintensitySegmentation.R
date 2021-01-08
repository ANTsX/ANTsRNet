#' White matter hyperintensity segmentation
#'
#' Perform WMH segmentation using the winning submission in the MICCAI
#' 2017 challenge by the sysu_media team using FLAIR or T1/FLAIR.  The
#' MICCAI challenge is discussed in
#'
#' \url{https://pubmed.ncbi.nlm.nih.gov/30908194/}
#'
#' with the sysu_media's team entry is discussed in
#'
#'  \url{https://pubmed.ncbi.nlm.nih.gov/30125711/}
#'
#' with the original implementation available here:
#'
#'     \url{https://github.com/hongweilibran/wmh_ibbmTum}
#'
#' @param flair input 3-D FLAIR brain image.
#' @param t1 input 3-D T1-weighted brain image (assumed to be aligned to
#' the flair, if specified).
#' @param doPreprocessing perform n4 bias correction?
#' @param useEnsemble boolean to check whether to use all 3 sets of weights.
#' @param useAxialSlicesOnly if \code{TRUE}, use original implementation which
#' was trained on axial slices.  If \code{FALSE}, use ANTsXNet variant
#' implementation which applies the slice-by-slice models to all 3 dimensions
#' and averages the results.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return WMH segmentation probability image
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "flair.nii.gz" )
#' probabilityMask <-sysuMediaWmhSegmentation( image )
#' }
#' @export
sysuMediaWmhSegmentation <- function( flair, t1 = NULL,
  doPreprocessing = TRUE, useEnsemble = TRUE, useAxialSlicesOnly = TRUE,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( flair@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  ################################
  #
  # Preprocess images
  #
  ################################

  flairPreprocessed <- flair
  if( doPreprocessing == TRUE )
    {
    flairPreprocessing <- preprocessBrainImage( flair,
        truncateIntensity = c( 0.01, 0.99 ),
        doBrainExtraction = FALSE,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    flairPreprocessed <- flairPreprocessing$preprocessedImage
    }

  numberOfChannels <- 1
  if( ! is.null( t1 ) )
    {
    t1Preprocessed <- t1
    if( doPreprocessing == TRUE )
      {
      t1Preprocessing <- preprocessBrainImage( t1,
          truncateIntensity = c( 0.01, 0.99 ),
          doBrainExtraction = FALSE,
          templateTransformType = NULL,
          doBiasCorrection = TRUE,
          doDenoising = FALSE,
          antsxnetCacheDirectory = antsxnetCacheDirectory,
          verbose = verbose )
      t1Preprocessed <- t1Preprocessing$preprocessedImage
      }
    numberOfChannels <- 2
    }

  ################################
  #
  # Estimate mask
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Estimating brain mask.\n" )
    }
  if( ! is.null( t1 ) )
    {
    brainMask <- brainExtraction( t1, modality = "t1" )
    } else {
    brainMask <- brainExtraction( flair, modality = "flair" )
    }

  referenceImage <- makeImage( c( 200, 200, 200 ),
                               voxval = 1,
                               spacing = c( 1, 1, 1 ),
                               origin = c( 0, 0, 0 ),
                               direction = diag( 3 ) )
  centerOfMassReference <- getCenterOfMass( referenceImage )
  centerOfMassImage <- getCenterOfMass( brainMask )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassReference,
        translation = centerOfMassImage - centerOfMassReference )
  flairPreprocessedWarped <- applyAntsrTransformToImage( xfrm, flairPreprocessed, referenceImage )
  brainMaskWarped <- thresholdImage(
        applyAntsrTransformToImage( xfrm, brainMask, referenceImage ), 0.5, 1.1, 1, 0 )
  if( ! is.null( t1 ) )
    {
    t1PreprocessedWarped <- applyAntsrTransformToImage( xfrm, t1Preprocessed, referenceImage )
    }

  ################################
  #
  # Gaussian normalize intensity based on brain mask
  #
  ################################

  meanFlair <- mean( flairPreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
  sdFlair <- sd( flairPreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
  flairPreprocessedWarped <- ( flairPreprocessedWarped - meanFlair ) / sdFlair

  if( numberOfChannels == 2 )
    {
    meanT1 <- mean( t1PreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
    sdT1 <- sd( t1PreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
    t1PreprocessedWarped <- ( t1PreprocessedWarped - meanT1 ) / sdT1
    }

  ################################
  #
  # Build models and load weights
  #
  ################################

  numberOfModels <- 1
  if( useEnsemble == TRUE )
    {
    numberOfModels <- 3
    }

  unetModels <- list()
  for( i in seq.int( numberOfModels ) )
    {
    weightsFileName <- ''
    if( numberOfChannels == 1 )
      {
      if( verbose == TRUE )
        {
        cat( "White matter hyperintensity:  retrieving model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( paste0( "sysuMediaWmhFlairOnlyModel", i - 1 ) )
      } else {
      if( verbose == TRUE )
        {
        cat( "White matter hyperintensity:  downloading model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( paste0( "sysuMediaWmhFlairT1Model", i - 1 ) )
      }

    unetModels[[i]] <- createSysuMediaUnetModel2D( c( 200, 200, numberOfChannels ) )
    unetModels[[i]]$load_weights( weightsFileName )
    }

  ################################
  #
  # Extract slices
  #
  ################################

  dimensionsToPredict <- c( 3 )
  if( useAxialSlicesOnly == FALSE )
    {
    dimensionsToPredict <- 1:3
    }

  batchX <- array( data = 0,
    c( sum( dim( flairPreprocessedWarped )[dimensionsToPredict]), 200, 200, numberOfChannels ) )

  sliceCount <- 1
  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    numberOfSlices <- dim( flairPreprocessedWarped )[dimensionsToPredict[d]]

    if( verbose == TRUE )
      {
      cat( "Extracting slices for dimension", d, "\n" )
      pb <- txtProgressBar( min = 1, max = numberOfSlices, style = 3 )
      }

    for( i in seq.int( numberOfSlices ) )
      {
      if( verbose )
        {
        setTxtProgressBar( pb, i )
        }

      flairSlice <- padOrCropImageToSize( extractSlice( flairPreprocessedWarped, i, dimensionsToPredict[d] ), c( 200, 200 ) )
      batchX[sliceCount,,,1] <- as.array( flairSlice )
      if( numberOfChannels == 2 )
        {
        t1Slice <- padOrCropImageToSize( extractSlice( t1PreprocessedWarped, i, dimensionsToPredict[d] ), c( 200, 200 ) )
        batchX[sliceCount,,,2] <- as.array( t1Slice )
        }
      sliceCount <- sliceCount + 1
      }
    if( verbose == TRUE )
      {
      cat( "\n" )
      }
    }

  ################################
  #
  # Do prediction and then restack into the image
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Prediction.\n" )
    }

  prediction <- predict( unetModels[[1]], batchX, verbose = verbose )
  if( numberOfModels > 1 )
    {
    for( i in seq.int( from = 2, to = numberOfModels ) )
      {
      prediction <- prediction + predict( unetModels[[i]], batchX, verbose = verbose )
      }
    }
  prediction <- prediction / numberOfModels

  permutations <- list()
  permutations[[1]] <- c( 1, 2, 3 )
  permutations[[2]] <- c( 2, 1, 3 )
  permutations[[3]] <- c( 2, 3, 1 )

  predictionImageAverage <- antsImageClone( flairPreprocessedWarped ) * 0

  currentStartSlice <- 1
  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    currentEndSlice <- currentStartSlice - 1 + dim( flairPreprocessedWarped )[dimensionsToPredict[d]]
    whichBatchSlices <- currentStartSlice:currentEndSlice
    predictionPerDimension <- prediction[whichBatchSlices,,,]
    predictionArray <- aperm( drop( predictionPerDimension ), permutations[[dimensionsToPredict[d]]] )
    predictionImage <- antsCopyImageInfo( flairPreprocessedWarped,
      padOrCropImageToSize( as.antsImage( predictionArray ), dim( flairPreprocessedWarped ) ) )
    predictionImageAverage <- predictionImageAverage + ( predictionImage - predictionImageAverage ) / d
    currentStartSlice <- currentEndSlice + 1
    }

  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      predictionImageAverage, flair )

  return( probabilityImage = probabilityImage )
}

#' White matter hypterintensity probabilistic segmentation
#'
#' Perform White matter hypterintensity probabilistic segmentation 
#' using deep learning
#'
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * brain extraction, and
#'    * affine registration to MNI.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param flair input 3-D FLAIR brain image.
#' @param t1 input 3-D T1-weighted brain image (assumed to be aligned to
#' the flair).
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @param debug return feature images in the last layer of the u-net model.
#' @return list consisting of the segmentation image and probability images for
#' each label.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' t1 <- antsImageRead( "t1.nii.gz" )
#' flair <- antsImageRead( "flair.nii.gz" )
#' results <- ewDavid( t1, flair )
#' }
#' @export
ewDavid <- function( flair, t1, doPreprocessing = TRUE,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( flair@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }
  if( t1@dimension != 3 )
    {
    stop( "Input t1 image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  ################################
  #
  # Preprocess image
  #
  ################################

  t1Preprocessed <- t1
  flairPreprocessed <- flair
  if( doPreprocessing == TRUE )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.001, 0.995 ),
        doBrainExtraction = TRUE,
        template = "croppedMni152",
        templateTransformType = "AffineFast",
        doBiasCorrection = FALSE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage # * t1Preprocessing$brainMask

    flairPreprocessed <- antsApplyTransforms( fixed = t1Preprocessed, moving = flair, 
      transformlist = t1Preprocessing$templateTransforms$fwdtransforms, interpolator = "linear",
      verbose = verbose )
    flairPreprocessed[t1Preprocessing$brainMask == 0] <- 0
    }

  ################################
  #
  # Build model and load weights
  #
  ################################

  patchSize <- c( 112L, 112L, 112L )
  strideLength <- dim( t1Preprocessed ) - patchSize

  classes <- c( "background", "wmh" )
  numberOfClassificationLabels <- length( classes )
  labels <- seq.int( numberOfClassificationLabels ) - 1

  imageModalities <- c( "T1", "FLAIR" )
  channelSize <- length( imageModalities )

  unetModel <- createUnetModel3D( c( patchSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5, nnUnetActivationStyle = FALSE, addAttentionGating = TRUE )

  if( verbose == TRUE )
    {
    cat( "ewDavid:  retrieving model weights.\n" )
    }
  weightsFileName <- getPretrainedNetwork( "ewDavidWmhSegmentationWeights", 
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )

  ################################
  #
  # Do prediction and normalize to native space
  #
  ################################

  if( verbose == TRUE )
    {
    message( "ewDavid:  prediction.\n" )
    }

  batchX <- array( data = 0, dim = c( 8, patchSize, channelSize ) )

  t1Preprocessed <- ( t1Preprocessed - mean( t1Preprocessed ) ) / sd( t1Preprocessed )
  t1Patches <- extractImagePatches( t1Preprocessed, patchSize, maxNumberOfPatches = "all",
                                    strideLength = strideLength, returnAsArray = TRUE )
  batchX[,,,,1] <- t1Patches

  flairPreprocessed <- ( flairPreprocessed - mean( flairPreprocessed ) ) / sd( flairPreprocessed )
  flairPatches <- extractImagePatches( flairPreprocessed, patchSize, maxNumberOfPatches = "all",
                                       strideLength = strideLength, returnAsArray = TRUE )
  batchX[,,,,2] <- flairPatches

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )

  probabilityImages <- list()
  for( i in seq.int( dim( predictedData )[5] ) )
    {
    message( "ewDavid:  reconstructing image ", classes[i], "\n" )
    reconstructedImage <- reconstructImageFromPatches( predictedData[,,,,i],
        domainImage = t1Preprocessed, strideLength = strideLength )
    if( doPreprocessing == TRUE )
      {
      probabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = reconstructedImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      } else {
      probabilityImages[[i]] <- reconstructedImage
      }
    }

  return( probabilityImages[[2]] )
}

