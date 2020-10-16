#' hippMapp3rSegmentation
#'
#' Perform HippMapp3r (hippocampal) segmentation described in
#'
#'  https://www.ncbi.nlm.nih.gov/pubmed/31609046
#'
#' with models and architecture ported from
#'
#' https://github.com/mgoubran/HippMapp3r
#'
#' Additional documentation and attribution resources found at
#'
#' https://hippmapp3r.readthedocs.io/en/latest/
#'
#' Preprocessing consists of:
#'    * n4 bias correction and
#'    * brain extraction.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param image input 3-D T1-weighted brain image.
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return labeled hippocampal mask (ANTsR image)
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( ANTsRNet )
#' library( keras )
#'
#' url <- "https://github.com/mgoubran/HippMapp3r/blob/master/data/test_case/mprage.nii.gz?raw=true"
#' imageFile <- "head.nii.gz"
#' download.file( url, imageFile )
#' image <- antsImageRead( imageFile )
#' imageN4 <- n4BiasFieldCorrection( image, verbose = TRUE )
#' segmentation <- hippMapp3rSegmentation( imageN4, verbose = TRUE )
#' }
#' @export
hippMapp3rSegmentation <- function( t1, doPreprocessing = TRUE,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {
  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  #########################################
  #
  # Perform preprocessing
  #

  if( verbose == TRUE )
    {
    cat( "*************  Preprocessing  ***************\n" )
    cat( "\n" )
    }

  t1Preprocessed <- t1
  if( doPreprocessing == TRUE )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = NULL,
        doBrainExtraction = TRUE,
        template = NULL,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    }

  #########################################
  #
  # Perform initial (stage 1) segmentation
  #

  if( verbose == TRUE )
    {
    cat( "*************  Initial stage segmentation  ***************\n" )
    cat( "\n" )
    }

  # Normalize to mprage_hippmapp3r space
  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: template normalization.\n" )
    }

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: retrieving template.\n" )
    }
  templateFileNamePath <- getANTsXNetData( "mprage_hippmapp3r",
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  templateImage <- antsImageRead( templateFileNamePath )

  registration <- antsRegistration( fixed = templateImage, moving = t1Preprocessed,
    typeofTransform = "antsRegistrationSyNQuick[t]", verbose = verbose )
  image <- registration$warpedmovout
  transforms <- list( fwdtransforms = registration$fwdtransforms,
                      invtransforms = registration$invtransforms )

  # Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: threshold.\n" )
    }
  imageArray <- as.array( image )
  imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
  thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
  thresholdedMask <- thresholdImage( image, -10000, thresholdValue, 0, 1 )
  thresholdedImage <- image * thresholdedMask

  # Standardize image
  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: standardize.\n" )
    }
  meanImage <- mean( thresholdedImage[thresholdedMask == 1] )
  sdImage <- sd( thresholdedImage[thresholdedMask == 1] )
  imageNormalized <- ( image - meanImage ) / sdImage
  imageNormalized <- imageNormalized * thresholdedMask

  # Trim and resample image
  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: trim and resample to (160, 160, 128).\n" )
    }
  imageCropped <- cropImage( imageNormalized, thresholdedMask, 1 )
  shapeInitialStage <- c( 160, 160, 128 )
  imageResampled <- resampleImage( imageCropped, shapeInitialStage,
    useVoxels = TRUE, interpType = "linear" )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: generate first network and download weights.\n" )
    }
  modelInitialStage <- createHippMapp3rUnetModel3D( c( shapeInitialStage, 1 ), doFirstNetwork = TRUE )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: retrieving model weights.\n" )
    }
  initialStageWeightsFileName <- getPretrainedNetwork( "hippMapp3rInitial",
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  modelInitialStage$load_weights( initialStageWeightsFileName )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: prediction.\n" )
    }
  dataInitialStage <- array( data = as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )
  maskArray <- modelInitialStage$predict( dataInitialStage, verbose = verbose )
  maskImageResampled <- as.antsImage( maskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
  maskImage <- resampleImage( maskImageResampled, dim( image ), useVoxels = TRUE,
    interpType = "nearestNeighbor" )
  maskImage[maskImage >= 0.5] <- 1
  maskImage[maskImage < 0.5] <- 0

  #########################################
  #
  # Perform refined (stage 2) segmentation
  #

  if( verbose == TRUE )
    {
    cat( "\n" )
    cat( "\n" )
    cat( "*************  Refine stage segmentation  ***************\n" )
    cat( "\n" )
    }

  maskArray <- drop( maskArray )
  centroidIndices <- which( maskArray == 1, arr.ind = TRUE, useNames = FALSE )
  centroid <- rep( 0, 3 )
  centroid[1] <- mean( centroidIndices[, 1] )
  centroid[2] <- mean( centroidIndices[, 2] )
  centroid[3] <- mean( centroidIndices[, 3] )

  shapeRefineStage <- c( 112, 112, 64 )
  lower <- floor( centroid - 0.5 * shapeRefineStage )
  upper <- lower + shapeRefineStage - 1

  imageTrimmed <- cropIndices( imageResampled, lower, upper )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: generate second network and retrieve weights.\n" )
    }
  modelRefineStage <- createHippMapp3rUnetModel3D( c( shapeRefineStage, 1 ), FALSE )
  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: retrieving model weights.\n" )
    }
  refineStageWeightsFileName <- getPretrainedNetwork( "hippMapp3rRefine",
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  modelRefineStage$load_weights( refineStageWeightsFileName )

  dataRefineStage <- array( data = as.array( imageTrimmed ), dim = c( 1, shapeRefineStage, 1 ) )

  cat( "    HippMapp3r:  do Monte Carlo iterations (SpatialDropout).\n" )
  numberOfMCIterations <- 30
  predictionRefineStage <- array( data = 0, dim = c( shapeRefineStage ) )
  for( i in seq_len( numberOfMCIterations ) )
    {
    if( verbose == TRUE )
      {
      cat( "        Monte Carlo iteration", i, "out of", numberOfMCIterations, "\n" )
      }
    predictionRefineStage <- ( modelRefineStage$predict( dataRefineStage, verbose = verbose )[1,,,,1] +
                              ( i - 1 ) * predictionRefineStage ) / i
    }

  predictionRefineStageArray <- array( data = 0, dim = dim( imageResampled ) )
  predictionRefineStageArray[lower[1]:upper[1],lower[2]:upper[2],lower[3]:upper[3]] <- predictionRefineStage
  probabilityMaskRefineStageResampled <- as.antsImage( predictionRefineStageArray ) %>% antsCopyImageInfo2( imageResampled )

  segmentationImageResampled <- labelClusters(
    thresholdImage( probabilityMaskRefineStageResampled, 0.0, 0.5, 0, 1 ), minClusterSize = 10 )
  segmentationImageResampled[segmentationImageResampled > 2] <- 0
  geom <- labelGeometryMeasures( segmentationImageResampled )
  if( length( geom$VolumeInMillimeters ) < 2 )
    {
    stop( "Error:  left and right hippocampus not found.")
    }
  if( geom$Centroid_x[1] < geom$Centroid_x[2] )
    {
    segmentationImageResampled[segmentationImageResampled == 1] <- 3
    segmentationImageResampled[segmentationImageResampled == 2] <- 1
    segmentationImageResampled[segmentationImageResampled == 3] <- 2
    }

  segmentationImage <- antsApplyTransforms( fixed = t1,
    moving = segmentationImageResampled, transformlist = transforms$invtransforms,
    whichtoinvert = c( TRUE ), interpolator = "genericLabel", verbose = verbose )

  return( segmentationImage )
  }
