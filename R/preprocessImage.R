#' Basic preprocessing pipeline for T1-weighted brain MRI
#'
#' Various preprocessing steps that have been previously described in 
#' various papers including the cortical thickness pipeline:
#'
#'         \url{https://www.ncbi.nlm.nih.gov/pubmed/24879923}
#'
#' @param image input t1-weighted brain MRI
#' @param truncateIntensity 2-element vector giving the low and high quantiles
#' for intensity truncation.
#' @param doBrainExtraction boolean for performing ANTsRNet brain extraction
#' If \code{TRUE}, returns mask.  3-D input only.
#' @param doBiasCorrection boolean for performing N4 bias field correction.
#' @param doDenoising boolean for performing NLM denoising.
#' @param intensityMatchingType Either regression- or histogram-based.  Only is
#' performed if \code{!is.null(referenceImage)}.
#' @param referenceImage reference image for intensity matching.
#' @param intensityNormalizationType Either rescale the intensities to [0,1] 
#' (i.e., "01") or zero-mean, unit variance (i.e., "0mean").  If \code{NULL}
#' normalization is not performed.
#' @param verbose print progress to the screen.
#' @return preprocessed image and (optionally) the brain mask
#' @author Tustison NJ, Avants BB
#' @examples
#'
#' library( ANTsRNet )
#'
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' preprocessedImage <- preprocessBrainImage( image, 
#'   truncateIntensity = c( 0.01, 0.99 ), doBrainExtraction = FALSE,
#'   doBiasCorrection = TRUE, doDenoising = TRUE, 
#'   intensityNormalizationType = "01", verbose = TRUE )
#'
#' @import keras
#' @export
preprocessBrainImage <- function( image, truncateIntensity = c( 0.01, 0.99 ), 
  doBrainExtraction = TRUE, doBiasCorrection = TRUE, doDenoising = TRUE, 
  intensityMatchingType = c( "regression", "histogram" ), referenceImage = NULL, 
  intensityNormalizationType = c( "01", "0mean" ), verbose = TRUE )
  {
  preprocessedImage <- antsImageClone( image )

  # Truncate intensity
  if( ! is.null( truncateIntensity ) )
    {
    quantiles <- quantile( image, truncateIntensity )
    if( verbose == TRUE )
      {
      message( paste0( "Preprocessing:  truncate intensities (low = ", quantiles[1], ", high = ", quantiles[2], ").\n" ) )
      }
    preprocessedImage[image < quantiles[1]] <- quantiles[1]  
    preprocessedImage[image > quantiles[2]] <- quantiles[2]  
    }

  # Brain extraction
  mask <- NULL
  if( doBrainExtraction == TRUE )
    {
    if( verbose == TRUE )
      {
      message( "Preprocessing:  brain extraction.\n" )
      }
    probabilityMask <- brainExtraction( preprocessedImage, verbose = verbose )
    mask <- thresholdImage( probabilityMask, 0.5, 1, 1, 0 )
    }

  # Do bias correction
  if( doBiasCorrection == TRUE )
    {
    if( verbose == TRUE )
      {
      message( "Preprocessing:  bias correction.\n" )
      }
    if( is.null( mask ) )
      {
      preprocessedImage <- n4BiasFieldCorrection( preprocessedImage, shrinkFactor = 4, verbose = verbose )
      } else {
      preprocessedImage <- n4BiasFieldCorrection( preprocessedImage, mask, shrinkFactor = 4, verbose = verbose )
      }
    }

  # Denoising
  if( doDenoising == TRUE )
    {
    if( verbose == TRUE )
      {
      message( "Preprocessing:  denoising.\n" )
      }
    if( is.null( mask ) )
      {
      preprocessedImage <- denoiseImage( preprocessedImage, shrinkFactor = 1, verbose = verbose )
      } else {
      preprocessedImage <- denoiseImage( preprocessedImage, mask, shrinkFactor = 1, verbose = verbose )
      }
    }

  # Image matching
  if( ! is.null( referenceImage ) )
    {
    if( verbose == TRUE )
      {
      message( "Preprocessing:  intensity matching.\n" )
      }
    if( intensityMatchingType == "regression" )
      {
      preprocessedImage <- regressionMatchImage( preprocessedImage, referenceImage )
      } else if( intensityatchingType == "histogram" ) {
      preprocessedImage <- histogramMatchImage( preprocessedImage, referenceImage )
      } else {
      stop( paste0( "Error:  unrecognized match type = ", intensityMatchingType, "\n" ) )
      }
    }

  # Intensity normalization
  if( ! is.null( intensityNormalizationType ) )
    {
    if( verbose == TRUE )
      {
      message( "Preprocessing:  intensity normalization.\n" ) 
      }
    if( intensityNormalizationType == "01" )
      {
      preprocessedImage <- preprocessedImage %>% iMath( "Normalize" )
      } else if( intensityNormalizationType == "0mean" ) {
      preprocessedImage <- ( preprocessedImage - mean( preprocessedImage ) ) / sd( preprocessedImage )
      } else {
      stop( paste0( "Error:  unrecognized intensity normalization type = ", intensityNormalizationType, "\n" ) )
      }
    }

  if( is.null( mask ) )
    {
    return( preprocessedImage )
    } else {
    return( list( preprocessedImage = preprocessedImage,
                  brainMask = mask ) )
    }
  }
