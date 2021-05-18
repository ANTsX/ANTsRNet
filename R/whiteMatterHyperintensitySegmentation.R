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

  imageSize <- c( 200, 200 )

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
        brainExtractionModality = NULL,
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
          brainExtractionModality = NULL,
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

  referenceImage <- makeImage( c( 170, 256, 256 ),
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

  flairPreprocessedWarped <- flairPreprocessedWarped * brainMaskWarped
  if ( ! is.null( t1 ) )
    {
    t1PreprocessedWarped <- t1PreprocessedWarped * brainMaskWarped
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

  if( verbose == TRUE )
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

  dimensionsToPredict <- c( 3 )
  if( useAxialSlicesOnly == FALSE )
    {
    dimensionsToPredict <- 1:3
    }

  batchX <- array( data = 0,
    c( sum( dim( flairPreprocessedWarped )[dimensionsToPredict]), imageSize, numberOfChannels ) )

  sliceCount <- 1
  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    numberOfSlices <- dim( flairPreprocessedWarped )[dimensionsToPredict[d]]

    if( verbose == TRUE )
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
      batchX[sliceCount,,,1] <- as.array( flairSlice )
      if( numberOfChannels == 2 )
        {
        t1Slice <- padOrCropImageToSize( extractSlice( t1PreprocessedWarped, i, dimensionsToPredict[d] ), imageSize )
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
      predictionImageAverage, flair ) * brainMask

  return( probabilityImage = probabilityImage )
}

#' White matter hypterintensity probabilistic segmentation
#'
#' Perform white matter hypterintensity probabilistic segmentation
#' using deep learning
#'
#' @param flair input 3-D FLAIR brain image.
#' @param t1 input 3-D T1-weighted brain image (assumed to be aligned to
#' the flair).
#' @param whichModel one of "sysu", "sysuT1Only", "sysuPlus",
#' "sysuPlusSeg", "sysuWithSite", "sysuWithSiteT1Only".
#' @param whichAxes apply 2-D model to 1 or more axes.  In addition to a scalar
#' or vector, e.g., \code{whichAxes = c(1, 3)}, one can use "max" for the
#' axis with maximum anisotropy (default) or "all" for all axes.
#' @param numberOfSimulations number of random affine perturbations to
#' transform the input.
#' @param sdAffine define the standard deviation of the affine transformation
#' parameter for the simulations.
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
ewDavid <- function( flair, t1, doPreprocessing = TRUE, whichModel = "sysu",
  whichAxes = "max", numberOfSimulations = 0, sdAffine = 0.01,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  doT1Only <- FALSE

  if( ( is.null( flair ) || missing( flair ) ) && ! missing( t1 ) )
    {
    doT1Only <- TRUE
    } else if( missing( t1 ) && missing( flair ) ) {
    stop( "Either supply a t1 or a t1 and flair.")
    }

  if( doT1Only && ! grepl( "T1Only", whichModel ) )
    {
    stop( "Specify a T1-only model if flair is not supplied." )
    }

  useT1Segmentation <- FALSE
  if( grepl( "Seg", whichModel ) )
    {
    useT1Segmentation <- TRUE
    }
  if( useT1Segmentation && doPreprocessing == True )
    {
    stop( "Using the t1 segmentation requires doPreprocessing = FALSE.")
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  doSlicewise <- TRUE

  if( doSlicewise == FALSE )
    {

    # ################################
    # #
    # # Preprocess images
    # #
    # ################################

    # t1Preprocessed <- t1
    # t1Preprocessing <- NULL
    # if( doPreprocessing == TRUE )
    #   {
    #   t1Preprocessing <- preprocessBrainImage( t1,
    #       truncateIntensity = c( 0.001, 0.995 ),
    #       brainExtractionModality = "t1",
    #       template = "croppedMni152",
    #       templateTransformType = "antsRegistrationSyNQuickRepro[a]",
    #       doBiasCorrection = FALSE,
    #       doDenoising = FALSE,
    #       antsxnetCacheDirectory = antsxnetCacheDirectory,
    #       verbose = verbose )
    #   t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    #   }

    # flairPreprocessed <- flair
    # if( doPreprocessing == TRUE )
    #   {
    #   flairPreprocessing <- preprocessBrainImage( flair,
    #       truncateIntensity = c( 0.01, 0.99 ),
    #       brainExtractionModality = NULL,
    #       doBiasCorrection = TRUE,
    #       doDenoising = FALSE,
    #       antsxnetCacheDirectory = antsxnetCacheDirectory,
    #       verbose = verbose )

    #   flairPreprocessed <- antsApplyTransforms( fixed = t1Preprocessed,
    #     moving = flairPreprocessing$preprocessedImage,
    #     transformlist = t1Preprocessing$templateTransforms$fwdtransforms,
    #     interpolator = "linear", verbose = verbose )
    #   flairPreprocessed <- flairPreprocessed * t1Preprocessing$brainMask
    #   }

    # ################################
    # #
    # # Build model and load weights
    # #
    # ################################

    # patchSize <- c( 112L, 112L, 112L )
    # strideLength <- dim( t1Preprocessed ) - patchSize

    # classes <- c( "background", "wmh" )
    # numberOfClassificationLabels <- length( classes )
    # labels <- seq.int( numberOfClassificationLabels ) - 1

    # imageModalities <- c( "T1", "FLAIR" )
    # channelSize <- length( imageModalities )

    # unetModel <- createUnetModel3D( c( patchSize, channelSize ),
    #   numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
    #   numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
    #   convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    #   weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

    # if( verbose == TRUE )
    #   {
    #   cat( "ewDavid:  retrieving model weights.\n" )
    #   }
    # weightsFileName <- getPretrainedNetwork( "ewDavidWmhSegmentationWeights",
    #   antsxnetCacheDirectory = antsxnetCacheDirectory )
    # load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    # ################################
    # #
    # # Do prediction and normalize to native space
    # #
    # ################################

    # if( verbose == TRUE )
    #   {
    #   message( "ewDavid:  prediction.\n" )
    #   }

    # batchX <- array( data = 0, dim = c( 8, patchSize, channelSize ) )

    # t1Preprocessed <- ( t1Preprocessed - mean( t1Preprocessed ) ) / sd( t1Preprocessed )
    # t1Patches <- extractImagePatches( t1Preprocessed, patchSize, maxNumberOfPatches = "all",
    #                                   strideLength = strideLength, returnAsArray = TRUE )
    # batchX[,,,,1] <- t1Patches

    # flairPreprocessed <- ( flairPreprocessed - mean( flairPreprocessed ) ) / sd( flairPreprocessed )
    # flairPatches <- extractImagePatches( flairPreprocessed, patchSize, maxNumberOfPatches = "all",
    #                                     strideLength = strideLength, returnAsArray = TRUE )
    # batchX[,,,,2] <- flairPatches

    # predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    # probabilityImages <- list()
    # for( i in seq.int( dim( predictedData )[5] ) )
    #   {
    #   message( "ewDavid:  reconstructing image ", classes[i], "\n" )
    #   reconstructedImage <- reconstructImageFromPatches( predictedData[,,,,i],
    #       domainImage = t1Preprocessed, strideLength = strideLength )
    #   if( doPreprocessing == TRUE )
    #     {
    #     probabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = reconstructedImage,
    #         transformlist = t1Preprocessing$templateTransforms$invtransforms,
    #         whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
    #     } else {
    #     probabilityImages[[i]] <- reconstructedImage
    #     }
    #   }

    # return( probabilityImages[[2]] )

    } else {  # doSlicewise

    ################################
    #
    # Preprocess images
    #
    ################################

    t1Preprocessed <- t1
    t1Preprocessing <- NULL
    brainMask <- NULL
    if( doPreprocessing == TRUE )
      {
      t1Preprocessing <- preprocessBrainImage( t1,
          truncateIntensity = c( 0.01, 0.99 ),
          brainExtractionModality = "t1",
          doBiasCorrection = TRUE,
          doDenoising = FALSE,
          antsxnetCacheDirectory = antsxnetCacheDirectory,
          verbose = verbose )
      brainMask <- t1Preprocessing$brainMask
      t1Preprocessed <- t1Preprocessing$preprocessedImage * brainMask
      }

    t1Segmentation <- NULL
    if( useT1Segmentation )
      {
      atroposSeg <- deepAtropos( t1, doPreprocessing = TRUE, verbose = verbose )
      t1Segmentation <- atroposSeg$segmentationImage
      }

    flairPreprocessed <- NULL
    if( ! doT1Only )
      {
      flairPreprocessed <- flair
      if( doPreprocessing == TRUE )
        {
        flairPreprocessing <- preprocessBrainImage( flair,
            truncateIntensity = c( 0.01, 0.99 ),
            brainExtractionModality = "flair",
            doBiasCorrection = TRUE,
            doDenoising = FALSE,
            antsxnetCacheDirectory = antsxnetCacheDirectory,
            verbose = verbose )
        brainMask <- flairPreprocessing$brainMask
        flairPreprocessed <- flairPreprocessing$preprocessedImage * brainMask
        }
      }

    resamplingParams <- antsGetSpacing( t1Preprocessed )

    doResampling <- FALSE
    for( d in seq.int( length( resamplingParams ) ) )
      {
      if( resamplingParams[d] < 0.8 )
        {
        resamplingParams[d] = 1.0
        doResampling <- TRUE
        }
      }

    if( doResampling )
      {
      if( ! doT1Only )
        {
        flairPreprocessed <- resampleImage( flairPreprocessed, resamplingParams, useVoxels = FALSE, interpType = 0 )
        }
      t1Preprocessed <- resampleImage( t1Preprocessed, resamplingParams, useVoxels = FALSE, interpType = 0 )
      if( ! is.null( t1Segmentation ) )
        {
        t1Segmentation <- resampleImage( t1Segmentation, resamplingParams, useVoxels = FALSE, interpType = 1 )
        }
      }

    ################################
    #
    # Build model and load weights
    #
    ################################

    templateSize = c( 208, 208 )

    imageModalities <- c( "T1", "FLAIR" )
    if( doT1Only )
      {
      imageModalities <- c( "T1" )
      }
    if( useT1Segmentation )
      {
      imageModalities <- c( imageModalities, "T1Seg" )
      }
    channelSize <- length( imageModalities )

    unetModel <- NULL
    if( whichModel == "sysu" || whichModel == "sysuT1Only" )
      {
      unetModel <- createUnetModel2D( c( templateSize, channelSize ),
        numberOfOutputs = 1, mode = 'sigmoid',
        numberOfFilters = c( 64, 96, 128, 256, 512 ), dropoutRate = 0.0,
        convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
        weightDecay = 0, additionalOptions = c( "initialConvolutionKernelSize[5]" ) )
      } else if( grepl( "WithSite", whichModel ) ) {
      unetModel <- createUnetModel2D( c( templateSize, channelSize ),
        numberOfOutputs = 1, mode = 'sigmoid',
        scalarOutputSize = 3, scalarOutputActivation = "softmax",
        numberOfFilters = c( 64, 96, 128, 256, 512 ), dropoutRate = 0.0,
        convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
        weightDecay = 0, additionalOptions = c( "initialConvolutionKernelSize[5]" ) )
      } else {
      unetModel <- createUnetModel2D( c( templateSize, channelSize ),
        numberOfOutputs = 1, mode = 'sigmoid',
        numberOfFilters = c( 64, 96, 128, 256, 512 ), dropoutRate = 0.0,
        convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
        weightDecay = 1e-5,
        additionalOptions = c( "nnUnetActivationStyle", "attentionGating", "initialConvolutionKernelSize[5]" ) )
      }

    if( verbose == TRUE )
      {
      cat( "ewDavid:  retrieving model weights.\n" )
      }

    weightsFileName <- NULL
    if( whichModel == "sysu" )
      {
      weightsFileName <- getPretrainedNetwork( "ewDavidSysu", antsxnetCacheDirectory = antsxnetCacheDirectory )
      } else if( whichModel == "sysuT1Only" ) {
      weightsFileName <- getPretrainedNetwork( "ewDavidSysuT1Only", antsxnetCacheDirectory = antsxnetCacheDirectory )
      } else if( whichModel == "sysuPlus" ) {
      weightsFileName <- getPretrainedNetwork( "ewDavidSysuPlus", antsxnetCacheDirectory = antsxnetCacheDirectory )
      } else if( whichModel == "sysuPlusSeg" ) {
      weightsFileName <- getPretrainedNetwork( "ewDavidSysuPlusSeg", antsxnetCacheDirectory = antsxnetCacheDirectory )
      } else if( whichModel == "sysuWithSite" ) {
      weightsFileName <- getPretrainedNetwork( "ewDavidSysuWithSite", antsxnetCacheDirectory = antsxnetCacheDirectory )
      } else if( whichModel == "sysuWithSiteT1Only" ) {
      weightsFileName <- getPretrainedNetwork( "ewDavidSysuWithSiteT1Only", antsxnetCacheDirectory = antsxnetCacheDirectory )
      }
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Extract slices
    #
    ################################

    dimensionsToPredict <- c( 1 )
    if( whichAxes == "max" )
      {
      dimensionsToPredict <- c( which.max( antsGetSpacing( t1Preprocessed ) )[1] )
      } else if( whichAxes == "all" ) {
      dimensionsToPredict <- 1:3
      } else {
      dimensionsToPredict <- whichAxes
      }

    batchX <- array( data = 0,
      c( sum( dim( t1Preprocessed )[dimensionsToPredict]), templateSize, channelSize ) )

    dataAugmentation <- NULL
    if( numberOfSimulations > 0 )
      {
      if( doT1Only )
        {
        if( useT1Segmentation )
          {
          dataAugmentation <-
            randomlyTransformImageData( t1Preprocessed,
            list( list( t1Preprocessed ) ),
            list( t1Segmentation ),
            numberOfSimulations = numberOfSimulations,
            transformType = 'affine',
            sdAffine = sdAffine,
            inputImageInterpolator = 'linear',
            segmentationImageInterpolator = 'nearestNeighbor' )
          } else {
          dataAugmentation <-
            randomlyTransformImageData( t1Preprocessed,
            list( list( t1Preprocessed ) ),
            numberOfSimulations = numberOfSimulations,
            transformType = 'affine',
            sdAffine = sdAffine,
            inputImageInterpolator = 'linear' )
          }
        } else {
        if( useT1Segmentation )
          {
          dataAugmentation <-
            randomlyTransformImageData( t1Preprocessed,
            list( list( flairPreprocessed, t1Preprocessed ) ),
            list( t1Segmentation ),
            numberOfSimulations = numberOfSimulations,
            transformType = 'affine',
            sdAffine = sdAffine,
            inputImageInterpolator = 'linear',
            segmentationImageInterpolator = 'nearestNeighbor' )
          } else {
          dataAugmentation <-
            randomlyTransformImageData( t1Preprocessed,
            list( list( flairPreprocessed, t1Preprocessed ) ),
            numberOfSimulations = numberOfSimulations,
            transformType = 'affine',
            sdAffine = sdAffine,
            inputImageInterpolator = 'linear' )
          }
        }
      }

    wmhProbabilityImage <- antsImageClone( t1 ) * 0
    wmhSite <- rep( 0, 3 )

    for( n in seq.int( numberOfSimulations + 1 ) )
      {
      batchFlair <- flairPreprocessed
      batchT1 <- t1Preprocessed
      batchT1Segmentation <- t1Segmentation

      if( n > 1 )
        {
        if( doT1Only )
          {
          batchT1 <- dataAugmentation$simulatedImages[[n-1]][[1]]
          } else {
          batchFlair <- dataAugmentation$simulatedImages[[n-1]][[1]]
          batchT1 <- dataAugmentation$simulatedImages[[n-1]][[2]]
          }
        if( useT1Segmentation )
          {
          batchT1Segmentation <- dataAugmentation$simulatedSegmentationImages[[n-1]]
          }
        }
      if( ! doT1Only )
        {
        batchFlair <- ( batchFlair - mean( batchFlair ) ) / sd( batchFlair )
        }
      batchT1 <- ( batchT1 - mean( batchT1 ) ) / sd( batchT1 )

      sliceCount <- 1
      for( d in seq.int( length( dimensionsToPredict ) ) )
        {
        numberOfSlices <- dim( batchT1 )[dimensionsToPredict[d]]

        if( verbose == TRUE )
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

          t1Slice <- padOrCropImageToSize( extractSlice( batchT1, i, dimensionsToPredict[d] ), templateSize )

          if( ! doT1Only )
            {
            flairSlice <- padOrCropImageToSize( extractSlice( batchFlair, i, dimensionsToPredict[d] ), templateSize )
            batchX[sliceCount,,,1] <- as.array( flairSlice )
            batchX[sliceCount,,,2] <- as.array( t1Slice )
            if( ! is.null( t1Segmentation ) )
              {
              t1SegmentationSlice <- padOrCropImageToSize( extractSlice( batchT1Segmentation, i, dimensionsToPredict[d] ), templateSize )
              batchX[sliceCount,,,3] <- as.array( t1SegmentationSlice ) / 6
              }
            } else {
            batchX[sliceCount,,,1] <- as.array( t1Slice )
            if( ! is.null( t1Segmentation ) )
              {
              t1SegmentationSlice <- padOrCropImageToSize( extractSlice( batchT1Segmentation, i, dimensionsToPredict[d] ), templateSize )
              batchX[sliceCount,,,2] <- as.array( t1SegmentationSlice ) / 6
              }
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
        if( n == 1 )
          {
          cat( "Prediction\n" )
          } else {
          cat( paste0( "Prediction (simulation ", n - 1, ")\n" ) )
          }
        }

      prediction <- predict( unetModel, batchX, verbose = verbose )

      permutations <- list()
      permutations[[1]] <- c( 1, 2, 3 )
      permutations[[2]] <- c( 2, 1, 3 )
      permutations[[3]] <- c( 2, 3, 1 )

      predictionImageAverage <- antsImageClone( t1Preprocessed ) * 0

      currentStartSlice <- 1
      for( d in seq.int( length( dimensionsToPredict ) ) )
        {
        currentEndSlice <- currentStartSlice - 1 + dim( t1Preprocessed )[dimensionsToPredict[d]]
        whichBatchSlices <- currentStartSlice:currentEndSlice
        if( is.list( prediction ) )
          {
          predictionPerDimension <- prediction[[1]][whichBatchSlices,,,1]
          } else {
          predictionPerDimension <- prediction[whichBatchSlices,,,1]
          }
        predictionArray <- aperm( drop( predictionPerDimension ), permutations[[dimensionsToPredict[d]]] )
        predictionImage <- antsCopyImageInfo( t1Preprocessed,
          padOrCropImageToSize( as.antsImage( predictionArray ), dim( t1Preprocessed ) ) )
        predictionImageAverage <- predictionImageAverage + ( predictionImage - predictionImageAverage ) / d
        currentStartSlice <- currentEndSlice + 1
        }

      if( doResampling )
        {
        predictionImageAverage <- resampleImageToTarget( predictionImageAverage, t1 )
        }

      wmhProbabilityImage <- wmhProbabilityImage + ( predictionImageAverage - wmhProbabilityImage ) / n
      if( is.list( prediction ) )
        {
        wmhSite <- wmhSite + ( colMeans( prediction[[2]] ) - wmhSite ) / n
        }
      }

    if( is.list( prediction ) )
      {
      return( list( wmhProbabilityImage, wmhSite ) )
      } else {
      return( wmhProbabilityImage )
      }
    }
}

