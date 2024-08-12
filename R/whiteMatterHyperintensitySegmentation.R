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
#' The original implementation used global thresholding as a quick
#' brain extraction approach.  Due to possible generalization difficulties,
#' we leave such post-processing steps to the user.  For brain or white
#' matter masking see functions brainExtraction or deepAtropos,
#' respectively.
#'
#' @param flair input 3-D FLAIR brain image.
#' @param t1 input 3-D T1-weighted brain image (assumed to be aligned to
#' the flair, if specified).
#' @param useEnsemble boolean to check whether to use all 3 sets of weights.
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
  useEnsemble = TRUE, antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( flair@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  imageSize <- c( 200, 200 )

  ################################
  #
  # Preprocess images
  #
  ################################

  flairPreprocessed <- flair
  numberOfChannels <- 1
  if( ! is.null( t1 ) )
    {
    t1Preprocessed <- t1
    numberOfChannels <- 2
    }

  referenceImage <- makeImage( c( 170, 256, 256 ),
                               voxval = 1,
                               spacing = c( 1, 1, 1 ),
                               origin = c( 0, 0, 0 ),
                               direction = diag( 3 ) )
  centerOfMassReference <- getCenterOfMass( referenceImage )
  centerOfMassImage <- getCenterOfMass( flair )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassReference,
        translation = centerOfMassImage - centerOfMassReference )
  flairPreprocessedWarped <- applyAntsrTransformToImage( xfrm, flairPreprocessed, referenceImage )
  if( ! is.null( t1 ) )
    {
    t1PreprocessedWarped <- applyAntsrTransformToImage( xfrm, t1Preprocessed, referenceImage )
    }

  ################################
  #
  # Gaussian normalize intensity based on brain mask
  #
  ################################

  meanFlair <- mean( flairPreprocessedWarped, na.rm = TRUE )
  sdFlair <- sd( flairPreprocessedWarped, na.rm = TRUE )
  flairPreprocessedWarped <- ( flairPreprocessedWarped - meanFlair ) / sdFlair
  if( numberOfChannels == 2 )
    {
    meanT1 <- mean( t1PreprocessedWarped, na.rm = TRUE )
    sdT1 <- sd( t1PreprocessedWarped, na.rm = TRUE )
    t1PreprocessedWarped <- ( t1PreprocessedWarped - meanT1 ) / sdT1
    }

  ################################
  #
  # Build models and load weights
  #
  ################################

  numberOfModels <- 1
  if( useEnsemble )
    {
    numberOfModels <- 3
    }

  if( verbose )
    {
    cat( "White matter hyperintensity:  retrieving model weights.\n" )
    }

  unetModels <- list()
  for( i in seq.int( numberOfModels ) )
    {
    weightsFileName <- ''
    if( numberOfChannels == 1 )
      {
      weightsFileName <- getPretrainedNetwork( paste0( "sysuMediaWmhFlairOnlyModel", i - 1 ),
        antsxnetCacheDirectory = antsxnetCacheDirectory )
      } else {
      weightsFileName <- getPretrainedNetwork( paste0( "sysuMediaWmhFlairT1Model", i - 1 ),
        antsxnetCacheDirectory = antsxnetCacheDirectory )
      }

    unetModels[[i]] <- createSysuMediaUnetModel2D( c( imageSize, numberOfChannels ), anatomy = "wmh" )
    unetModels[[i]]$load_weights( weightsFileName )
    }

  ################################
  #
  # Extract slices
  #
  ################################

  rotate <- function( x ) t( apply( x, 2, rev ) )
  rotateReverse <- function( x ) apply( t( x ), 2, rev )

  dimensionsToPredict <- c( 3 )

  batchX <- array( data = 0,
    c( sum( dim( flairPreprocessedWarped )[dimensionsToPredict]), imageSize, numberOfChannels ) )

  sliceCount <- 1
  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    numberOfSlices <- dim( flairPreprocessedWarped )[dimensionsToPredict[d]]

    if( verbose )
      {
      cat( "Extracting slices for dimension", dimensionsToPredict[d], "\n" )
      pb <- txtProgressBar( min = 1, max = numberOfSlices, style = 3 )
      }

    for( i in seq.int( numberOfSlices ) )
      {
      if( verbose )
        {
        setTxtProgressBar( pb, i )
        }

      flairSlice <- padOrCropImageToSize( extractSlice( flairPreprocessedWarped, i, dimensionsToPredict[d] ), imageSize )
      batchX[sliceCount,,,1] <- rotateReverse( as.array( flairSlice ) )
      if( numberOfChannels == 2 )
        {
        t1Slice <- padOrCropImageToSize( extractSlice( t1PreprocessedWarped, i, dimensionsToPredict[d] ), imageSize )
        batchX[sliceCount,,,2] <- rotateReverse( as.array( t1Slice ) )
        }
      sliceCount <- sliceCount + 1
      }
    if( verbose )
      {
      cat( "\n" )
      }
    }

  ################################
  #
  # Do prediction and then restack into the image
  #
  ################################

  if( verbose )
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
  for( i in seq.int( dim( flairPreprocessedWarped )[3] ) )
    {
    prediction[i,,,1] <- rotate( prediction[i,,,1] )
    }

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

  return( probabilityImage )
}

#' hyperMapp3rSegmentation
#'
#' Perform HyperMapp3r (white matter hyperintensity) segmentation described in
#'
#'  https://pubmed.ncbi.nlm.nih.gov/35088930/
#'
#' with models and architecture ported from
#'
#' https://github.com/AICONSlab/HyperMapp3r
#'
#' Additional documentation and attribution resources found at
#'
#' https://hypermapp3r.readthedocs.io/en/latest/
#'
#' Preprocessing consists of:
#'    * n4 bias correction and
#'    * brain extraction.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param t1 input 3-D t1-weighted MR image.  Assumed to be aligned with the flair.
#' @param flair input 3-D flair MR image.  Assumed to be aligned with the t1.
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return white matter hyperintensity probability mask
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' wmh <- hyperMapp3rSegmentation( image, verbose = TRUE )
#' }
#' @export
hyperMapp3rSegmentation <- function( t1, flair, doPreprocessing = TRUE,
  numberOfMonteCarloIterations = 30, antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( t1@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  #########################################
  #
  # Perform preprocessing
  #

  if( verbose )
    {
    cat( "*************  Preprocessing  ***************\n" )
    cat( "\n" )
    }

  t1Preprocessed <- t1
  brainMask <- NULL
  if( doPreprocessing )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.01, 0.99 ),
        brainExtractionModality = "t1",
        template = NULL,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    brainMask <- t1Preprocessing$brainMask
    t1Preprocessed <- t1Preprocessing$preprocessedImage * brainMask
    } else {
    # If we don't generate the mask from the preprocessing, we assume that we
    # can extract the brain directly from the foreground of the t1 image.
    brainMask <- thresholdImage( t1, 0, 0, 0, 1 )
    }

  t1PreprocessedMean <- mean( t1Preprocessed[brainMask > 0] )
  t1PreprocessedSd <- sd( t1Preprocessed[brainMask > 0] )
  t1Preprocessed[brainMask > 0] <- ( t1Preprocessed[brainMask > 0] - t1PreprocessedMean ) / t1PreprocessedSd

  flairPreprocessed <- flair
  if( doPreprocessing )
    {
    flairPreprocessing <- preprocessBrainImage( flair,
        truncateIntensity = c( 0.01, 0.99 ),
        brainExtractionModality = "flair",
        template = NULL,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    flairPreprocessed <- flairPreprocessing$preprocessedImage * brainMask
    }

  flairPreprocessedMean <- mean( flairPreprocessed[brainMask > 0] )
  flairPreprocessedSd <- sd( flairPreprocessed[brainMask > 0] )
  flairPreprocessed[brainMask > 0] <- ( flairPreprocessed[brainMask > 0] - flairPreprocessedMean ) / flairPreprocessedSd

  # Reorient to hypermapp3r space
  if( verbose )
    {
    cat( "    HyperMapp3r: reorient input images.\n" )
    }

  channelSize <- 2
  inputImageSize <- c( 224, 224, 224 )
  templateArray <- array( data = 1, dim = inputImageSize )
  templateDirection <- diag( 3 )
  reorientTemplate <- as.antsImage( templateArray, origin = c( 0, 0, 0 ),
     spacing = c( 1, 1, 1 ), direction = templateDirection )

  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass( brainMask )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
    center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate )

  batchX <- array( data = 0, dim = c( 1, inputImageSize, channelSize ) )

  t1PreprocessedWarped <- applyAntsrTransformToImage( xfrm, t1Preprocessed, reorientTemplate )
  batchX[1,,,,1] <- as.array( t1PreprocessedWarped )

  flairPreprocessedWarped <- applyAntsrTransformToImage( xfrm, flairPreprocessed, reorientTemplate )
  batchX[1,,,,2] <- as.array( flairPreprocessedWarped )

  if( verbose )
    {
    cat( "    HyperMapp3r: generate network and load weights.\n" )
    }
  model <- createHyperMapp3rUnetModel3D( c( inputImageSize, 2 ) )
  weightsFileName <- getPretrainedNetwork( "hyperMapp3r",
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  model$load_weights( weightsFileName )

  if( verbose )
    {
    cat( "    HyperMapp3r: prediction.\n" )
    }


  if( verbose )
    {
    cat( "    HyperMapp3r: do Monte Carlo iterations (SpatialDropout).\n" )
    }

  predictionArray <- array( data = 0, dim = c( inputImageSize ) )
  for( i in seq_len( numberOfMonteCarloIterations ) )
    {
    if( verbose )
      {
      cat( "        Monte Carlo iteration", i, "out of", numberOfMonteCarloIterations, "\n" )
      }
    predictionArray <- ( model$predict( batchX, verbose = verbose )[1,,,,1] +
                              ( i - 1 ) * predictionArray ) / i
    }
  predictionImage <- as.antsImage( predictionArray, origin = antsGetOrigin( reorientTemplate ),
      spacing = antsGetSpacing( reorientTemplate ), direction = antsGetDirection( reorientTemplate ) )

  xfrmInv <- invertAntsrTransform( xfrm )
  probabilityImage <- applyAntsrTransformToImage( xfrmInv, predictionImage, t1 )

  return( probabilityImage )
  }


#' White matter hyperintensity probabilistic segmentation
#'
#' Perform white matter hyperintensity probabilistic segmentation
#' using deep learning
#'
#' @param flair input 3-D FLAIR brain image.
#' @param t1 input 3-D T1-weighted brain image (assumed to be aligned to
#' the flair).
#' @param whiteMatterMask input white matter mask for patch extraction.  If None,
#' calculated using deepAtropos (labels 3 and 4).
#' @param useCombinedModel Original or combined.
#' @param predictionBatchSize Control memory usage for prediction.  More consequential 
#' for GPU-usage.
#' @param patchStrideLength  3-D vector or int.   Dictates the stride length for 
#' accumulating predicting patches.
#' @param doPreprocessing perform n4 bias correction, intensity truncation, brain
#' extraction.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return probabilistic image.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' t1 <- antsImageRead( "t1.nii.gz" )
#' flair <- antsImageRead( "flair.nii.gz" )
#' results <- wmhSegmentation( t1, flair )
#' }
#' @export
wmhSegmentation <- function( flair, t1, whiteMatterMask = NULL, 
  useCombinedModel = TRUE, predictionBatchSize = 16,
  patchStrideLength = 32,
  doPreprocessing = TRUE, antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( any( dim( t1 ) < c( 64, 64, 64 ) ) )
    {
    stop( "Images must be > 64 voxels per dimension." )
    }

  ################################
  #
  # Preprocess images
  #
  ################################

  if( is.null( whiteMatterMask ) )
    {
    if( verbose )
      {
      message( "Calculate white matter mask.\n" )
      atropos <- deepAtropos( t1, doPreprocessing = TRUE, verbose = verbose )
      whiteMatterMask <- thresholdImage( atropos$segmentationImage, 3, 4, 1, 0 )
      }
    }

  t1Preprocessed <- NULL
  flairPreprocessed <- NULL

  if( doPreprocessing )
    {
    if( verbose )
      {
      message( "Preprocess T1 and FLAIR images.\n" )
      }
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.001, 0.995 ),
        brainExtractionModality = "t1",
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    brainMask <- thresholdImage( t1Preprocessing$brainMask, 0.5, 1, 1, 0 )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * brainMask

    flairPreprocessing <- preprocessBrainImage( flair,
        truncateIntensity = NULL,
        brainExtractionModality = NULL,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    flairPreprocessed <- flairPreprocessing$preprocessedImage * brainMask
    } else {
    t1Preprocessed <- antsImageClone( t1 )
    flairPreprocessed <- antsImageClone( flair )
    }

  t1PreprocessedMin <- min( t1Preprocessed[whiteMatterMask > 0] )
  t1PreprocessedMax <- max( t1Preprocessed[whiteMatterMask > 0] )
  flairPreprocessedMin <- min( flairPreprocessed[whiteMatterMask > 0] )
  flairPreprocessedMax <- max( flairPreprocessed[whiteMatterMask > 0] )

  t1Preprocessed <- ( t1Preprocessed - t1PreprocessedMin ) / ( t1PreprocessedMax - t1PreprocessedMin )
  flairPreprocessed <- ( flairPreprocessed - flairPreprocessedMin ) / ( flairPreprocessedMax - flairPreprocessedMin )

  ################################
  #
  # Build model and load weights
  #
  ################################

  if( verbose )
    {
    message( "Load model and weights.\n" )
    }

  patchSize <- c( 64L, 64L, 64L )
  strideLength <- c( 32L, 32L, 32L )
  numberOfFilters <- c( 64, 96, 128, 256, 512 )
  channelSize <- 2

  model <- createSysuMediaUnetModel3D( c( patchSize, channelSize ),
                                       numberOfFilters = numberOfFilters )

  weightsFileName <- ""
  if( useCombinedModel )
    {
    weightsFileName <- getPretrainedNetwork( "antsxnetWmhOr",
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    } else {
    weightsFileName <- getPretrainedNetwork( "antsxnetWmh",
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    }
  load_model_weights_hdf5( model, filepath = weightsFileName )

  ################################
  #
  # Extract patches
  #
  ################################

  if( verbose )
    {
    message( "Extract patches." )
    }

  t1Patches <- extractImagePatches( t1Preprocessed,
                                    patchSize = patchSize,
                                    maxNumberOfPatches = "all",
                                    strideLength = strideLength,
                                    maskImage = whiteMatterMask,
                                    randomSeed = NULL,
                                    returnAsArray = TRUE )
  flairPatches <- extractImagePatches( flairPreprocessed,
                                       patchSize = patchSize,
                                       maxNumberOfPatches = "all",
                                       strideLength = strideLength,
                                       maskImage = whiteMatterMask,
                                       randomSeed = NULL,
                                       returnAsArray = TRUE )

  totalNumberOfPatches <- dim( t1Patches )[1]
 

  ################################
  #
  # Do prediction and then restock into the image
  #
  ################################

  numberOfBatches <- floor( totalNumberOfPatches / predictionBatchSize )
  residualNumberOfPatches <- totalNumberOfPatches - numberOfBatches * predictionBatchSize
  if( residualNumberOfPatches > 0 )
    {
    numberOfBatches <- numberOfBatches + 1 
    }

  if( verbose )
    {
    message( "Total number of patches: ", totalNumberOfPatches )
    message( "Prediction batch size: ", predictionBatchSize )
    message( "Number of batches: ", numberOfFullBatches )
    }
 
  prediction <- array( data = 0, dim = c( totalNumberOfPatches, patchSize, 1 ) )
  for( b in seq.int( numberOfBatches ) )
    {
    batchX <- NULL
    if( b < numberOfFullBatches || residualNumberOfPatches == 0 )
      {
      batchX <- array( data = 0, dim = c( predictionBatchSize, patchSize, channelSize ) ) 
      } else {
      
      batchX <- array( data = 0, dim = c( residualNumberOfPatches, patchSize, channelSize ) ) 
      }

    indices <- ( ( b - 1 ) * predictionBatchSize + 1):( ( b - 1 ) * predictionBatchSize + dim( batchX )[1] )
    batchX[,,,,1] <- flairPatches[indices,,,]
    batchX[,,,,2] <- t1Patches[indices,,,]

    if( verbose )
      {
      message( "Predicting batch ", b, " of ", numberOfBatches )
      }
    prediction[indices,,,,] <- model %>% predict( batchX, verbose = verbose )
    }  

  wmhProbabilityImage <- reconstructImageFromPatches( drop( prediction ),
                                                      strideLength = strideLength,
                                                      domainImage = whiteMatterMask,
                                                      domainImageIsMask = TRUE )
  }

#' PVS/VRS segmentation.
#'
#' Perform segmentation of perivascular (PVS) or Vircho-Robin spaces (VRS).
#'    \url{https://pubmed.ncbi.nlm.nih.gov/34262443/}
#' with the original implementation available here:
#'    https://github.com/pboutinaud/SHIVA_PVS
#'
#' @param t1 input 3-D T1-weighted brain image.
#' @param flair (Optional) input 3-D FLAIR brain image (aligned to T1 image).
#' @param whichModel integer or string. Several models were trained for the 
#' case of T1-only or T1/FLAIR image pairs.  One can use a specific single 
#' trained model or the average of the entire ensemble.  I.e., options are:
#'            * For T1-only:  0, 1, 2, 3, 4, 5.
#'            * For T1/FLAIR: 0, 1, 2, 3, 4.
#'            * Or "all" for using the entire ensemble.
#' @param doPreprocessing perform n4 bias correction, intensity truncation, brain
#' extraction.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return probabilistic image.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' t1 <- antsImageRead( "t1.nii.gz" )
#' flair <- antsImageRead( "flair.nii.gz" )
#' results <- wmhSegmentation( t1, flair )
#' }
#' @export
shivaPvsSegmentation <- function( t1, flair = NULL,
  whichModel = "all", doPreprocessing = TRUE, 
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{
  ################################
  #
  # Preprocess images
  #
  ################################

  t1Preprocessed <- NULL
  flairPreprocessed <- NULL

  if( doPreprocessing )
    {
    if( verbose )
      {
      message( "Preprocess T1 and FLAIR images.\n" )
      }
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.0, 0.99 ),
        brainExtractionModality = "t1",
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    brainMask <- thresholdImage( t1Preprocessing$brainMask, 0.5, 1, 1, 0 )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * brainMask
    
    if( ! is.null( flair ) )
      { 
      flairPreprocessing <- preprocessBrainImage( flair,
          truncateIntensity = NULL,
          brainExtractionModality = NULL,
          doBiasCorrection = TRUE,
          doDenoising = FALSE,
          antsxnetCacheDirectory = antsxnetCacheDirectory,
          verbose = verbose )
      flairPreprocessed <- flairPreprocessing$preprocessedImage * brainMask
      }
    } else {
    t1Preprocessed <- antsImageClone( t1 )
    if( ! is.null( flair ) )
      {
      flairPreprocessed <- antsImageClone( flair )
      }
    }

  imageShape <- c( 160, 214, 176 )  
  onesArray <- data( data = 1, dim = imageShape )
  reorientTemplate <- as.antsImage( onesArray, origin = c( 0, 0, 0 ),
                                    spacing = c( 1, 1, 1 ),
                                    direction = diag( 3 ) )

  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass( t1Preprocessed * 0 + 1 )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
    center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate )

  t1Preprocessed <- applyAntsrTransformToImage( xfrm, t1Preprocessed, 
                                                reorientTemplate ) 
  if( ! is.null( flair ) )
    {
    flairPreprocessed <- applyAntsrTransformToImage( xfrm, flairPreprocessed, 
                                                     reorientTemplate ) 
    }

  ################################
  #
  # Load models and predict
  #
  ################################

  batchY <- NULL

  if( is.null( flair ) )
    {
    batchX <- array( data = 0, dim = c( 1, imageShape, 1 ) )
    batchX[1,,,,1] <- as.array( t1Preprocessed )
    
    modelIds <- c( whichModel )
    if( whichModel == "all" )
      {
      modelIds <- c( 0, 1, 2, 3, 4, 5 )
      }

    for( i in seq.int( length( modelIds ) ) )
      {
      modelFile <- getPretrainedNetwork( paste0( "pvs_shiva_t1_", modelIds[i] ),
                                         antsxnetCacheDirectory = antsxnetCacheDirectory ) 
      if( verbose )
        {
        cat( "Loading", modelFile, "\n" )
        }
      model <- tensorflow::tf$keras$models$load_model( modelFile )
      if( i == 1 )
        {
        batchY <- model$predict( batchX, verbose = verbose )
        } else {
        batchY <- batchY + model$predict( batchX, verbose = verbose )
        }
      }
    batchY <- batchY / length( modelIds )
    } else {
    batchX <- array( data = 0, dim = c( 1, imageShape, 2 ) )
    batchX[1,,,,1] <- as.array( t1Preprocessed )
    batchX[1,,,,2] <- as.array( flairPreprocessed )
    
    modelIds <- c( whichModel )
    if( whichModel == "all" )
      {
      modelIds <- c( 0, 1, 2, 3, 4, 5 )
      }

    for( i in seq.int( length( modelIds ) ) )
      {
      modelFile <- getPretrainedNetwork( paste0( "pvs_shiva_t1_flair_", modelIds[i] ),
                                         antsxnetCacheDirectory = antsxnetCacheDirectory ) 
      if( verbose )
        {
        cat( "Loading", modelFile, "\n" )
        }
      model <- tensorflow::tf$keras$models$load_model( modelFile )
      if( i == 1 )
        {
        batchY <- model$predict( batchX, verbose = verbose )
        } else {
        batchY <- batchY + model$predict( batchX, verbose = verbose )
        }
      }
    batchY <- batchY / length( modelIds )
    }

  pvs <- as.antsImage( drop( batchY ), origin = antsGetOrigin( t1 ),
                       spacing = antsGetSpacing( t1 ),
                       direction = antsGetDirection( direction ) )
  pvs <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ), pvs, t1 )     
  return( pvs )
}