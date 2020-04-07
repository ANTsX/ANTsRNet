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
#' @param image input 3-D T1-weighted brain image.
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return brain mask (ANTsR image)
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
#' hippocampalProbabilityMask <- hippMapp3rSegmentation( imageN4, verbose = TRUE )
#' }
#' @export
hippMapp3rSegmentation <- function( image, outputDirectory = NULL, verbose = FALSE )
  {
  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  #########################################
  #
  # Perform initial (stage 1) segmentation
  #

  if( verbose == TRUE )
    {
    cat( "*************  Initial stage segmentation  ***************\n" )
    cat( "  (warning:  steps are somewhat different in the publication.)\n" )
    cat( "\n" )
    }

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

  # Resample image
  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: resample to (160, 160, 128).\n" )
    }
  shapeInitialStage <- c( 160, 160, 128 )
  imageResampled <- resampleImage( imageNormalized, shapeInitialStage,
    useVoxels = TRUE, interpType = "linear" )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: generate first network and download weights.\n" )
    }
  modelInitialStage <- createHippMapp3rUnetModel3D( c( shapeInitialStage, 1 ), doFirstNetwork = TRUE )

  initialStageWeightsFileName <- paste0( outputDirectory, "/hippMapp3rInitial.h5" )
  if( ! file.exists( initialStageWeightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "    HippMapp3r: downloading model weights.\n" )
      }
    initialStageWeightsFileName <- getPretrainedNetwork( "hippMapp3rInitial", initialStageWeightsFileName )
    }
  modelInitialStage$load_weights( initialStageWeightsFileName )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: prediction.\n" )
    }
  dataInitialStage <- array( data = as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )
  maskArray <- modelInitialStage$predict( dataInitialStage )
  maskImageResampled <- as.antsImage( maskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
  maskImage <- resampleImage( maskImageResampled, dim( image ), useVoxels = TRUE,
    interpType = "nearestNeighbor" )
  maskImage[maskImage >= 0.5] <- 1
  maskImage[maskImage < 0.5] <- 0
  antsImageWrite( maskImage, "/Users/ntustison/Desktop/maskInitialStage.nii.gz" )

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

  maskTrimmed <- cropIndices( maskImageResampled, lower, upper )
  imageTrimmed <- cropIndices( imageResampled, lower, upper )

  if( verbose == TRUE )
    {
    cat( "    HippMapp3r: generate second network and download weights.\n" )
    }
  modelRefineStage <- createHippMapp3rUnetModel3D( c( shapeRefineStage, 1 ), FALSE )
  refineStageWeightsFileName <- paste0( outputDirectory, "/hippMapp3rRefine.h5" )
  if( ! file.exists( refineStageWeightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "    HippMapp3r: downloading model weights.\n" )
      }
    refineStageWeightsFileName <- getPretrainedNetwork( "hippMapp3rRefine", refineStageWeightsFileName )
    }
  modelRefineStage$load_weights( refineStageWeightsFileName )

  dataRefineStage <- array( data = as.array( imageTrimmed ), dim = c( 1, shapeRefineStage, 1 ) )

  cat( "    HippMapp3r:  do Monte Carlo iterations (SpatialDropout).\n" )
  numberOfMCIterations <- 30
  predictionRefineStage <- array( data = 0, dim = c( numberOfMCIterations, shapeRefineStage ) )
  for( i in seq_len( numberOfMCIterations ) )
    {
    cat( "        Monte Carlo iteration", i, "out of", numberOfMCIterations, "\n" )
    predictionRefineStage[i,,,] <- modelRefineStage$predict( dataRefineStage )[1,,,,1]
    }
  predictionRefineStage <- apply( predictionRefineStage, c( 2, 3, 4 ), mean )

  cat( "    HippMapp3r:  Average Monte Carlo results.\n" )
  predictionRefineStageArray <- array( data = 0, dim = dim( imageResampled ) )
  predictionRefineStageArray[lower[1]:upper[1],lower[2]:upper[2],lower[3]:upper[3]] <- predictionRefineStage
  probabilityMaskRefineStageResampled <- as.antsImage( predictionRefineStageArray ) %>% antsCopyImageInfo2( imageResampled )
  probabilityMaskRefineStage <- resampleImageToTarget( probabilityMaskRefineStageResampled, image )

  return( probabilityMaskRefineStage )
  }
