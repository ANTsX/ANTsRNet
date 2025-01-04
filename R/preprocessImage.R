#' Basic preprocessing pipeline for T1-weighted brain MRI
#'
#' Various preprocessing steps that have been previously described in
#' various papers including the cortical thickness pipeline:
#'
#'     \url{https://www.ncbi.nlm.nih.gov/pubmed/24879923}
#'
#' @param image input t1-weighted brain MRI
#' @param truncateIntensity 2-element vector giving the low and high quantiles
#' for intensity truncation.
#' @param brainExtractionModality string or NULL.  Perform brain extraction
#' using antsxnet tools.  One of "t1", "t1v0", "t1nobrainer", "t1combined",
#' "flair", "t2", "bold", "fa", "t1infant", "t2infant", "t1threetissue", or NULL.
#' @param templateTransformType see Details in help for \code{antsRegistration}.
#' Typically "Rigid" or "Affine".
#' @param template an ANTs image (not skull-stripped). Other premade templates
#'  are "biobank" and "croppedMni152".
#' @param doBiasCorrection boolean for performing N4 bias field correction.
#' @param returnBiasField if TRUE, return bias field as an additional output
#' *without* bias correcting the preprocessed image.
#' @param doDenoising boolean for performing non-local means denoising.
#' @param intensityMatchingType Either "regression" or "histogram".  Only is
#' performed if \code{!is.null(referenceImage)}.
#' @param referenceImage reference image for intensity matching.
#' @param intensityNormalizationType Either rescale the intensities to [0,1]
#' (i.e., "01") or zero-mean, unit variance (i.e., "0mean").  If \code{NULL}
#' normalization is not performed.
#' @param verbose print progress to the screen.
#' @return preprocessed image and, optionally, the brain mask, bias field, and
#' template transforms.
#' @author Tustison NJ, Avants BB
#' @examples
#'
#' library( ANTsR )
#' library( ANTsRNet )
#'
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' preprocessedImage <- preprocessBrainImage( image,
#'   truncateIntensity = c( 0.01, 0.99 ),
#'   doBiasCorrection = TRUE, doDenoising = TRUE,
#'   intensityNormalizationType = "01", verbose = FALSE )
#'
#' @import keras
#' @export
preprocessBrainImage <- function( image, truncateIntensity = c( 0.01, 0.99 ),
  brainExtractionModality = NULL, templateTransformType = NULL, template = "biobank",
  doBiasCorrection = TRUE, returnBiasField = FALSE, doDenoising = TRUE,
  intensityMatchingType = NULL, referenceImage = NULL,
  intensityNormalizationType = NULL, verbose = TRUE )
  {

  preprocessedImage <- antsImageClone( image, out_pixeltype = "float" )

  # Truncate intensity
  if( ! is.null( truncateIntensity ) )
    {
    quantiles <- quantile( image, truncateIntensity )
    if( verbose )
      {
      message( paste0( "Preprocessing:  truncate intensities (low = ", quantiles[1], ", high = ", quantiles[2], ").\n" ) )
      }
    preprocessedImage[image < quantiles[1]] <- quantiles[1]
    preprocessedImage[image > quantiles[2]] <- quantiles[2]
    }

  # Brain extraction
  mask <- NULL
  if( ! is.null( brainExtractionModality ) )
    {
    if( verbose )
      {
      message( "Preprocessing:  brain extraction.\n" )
      }
    bext <- brainExtraction( preprocessedImage, modality = brainExtractionModality, verbose = verbose )
    if( brainExtractionModality == "t1threetissue" ) 
      {
      mask <- thresholdImage( bext$segmentationImage, 1, 1, 1, 0 ) 
      } else if( brainExtractionModality == "t1combined" ) { 
      mask <- thresholdImage( bext$segmentationImage, 2, 3, 1, 0 ) 
      } else {
      mask <- thresholdImage( bext, 0.5, 1, 1, 0 )
      mask <- morphology( mask, "close", 6 ) %>% iMath("FillHoles")
      }
    }

  # Template normalization
  transforms <- NULL
  if( ! is.null( templateTransformType ) )
    {
    templateImage <- NULL
    if( is.character( template ) )
      {
      templateFileNamePath <- getANTsXNetData( template )
      templateImage <- antsImageRead( templateFileNamePath )
      } else {
      templateImage <- template
      }
    if( is.null( mask ) )
      {
      registration <- antsRegistration( fixed = templateImage, moving = preprocessedImage,
        typeofTransform = templateTransformType, verbose = verbose )
      preprocessedImage <- registration$warpedmovout
      transforms <- list( fwdtransforms = registration$fwdtransforms,
                          invtransforms = registration$invtransforms )
      } else {
      templateProbabilityMask <- brainExtraction( templateImage, modality = "t1",
        verbose = verbose )
      templateMask <- thresholdImage( templateProbabilityMask, 0.5, 1, 1, 0 )
      templateBrainImage <- templateMask * templateImage

      preprocessedBrainImage <- preprocessedImage * mask

      registration <- antsRegistration( fixed = templateBrainImage, moving = preprocessedBrainImage,
        typeofTransform = templateTransformType, verbose = verbose )
      transforms <- list( fwdtransforms = registration$fwdtransforms,
                          invtransforms = registration$invtransforms )

      preprocessedImage <- antsApplyTransforms( fixed = templateImage, moving = preprocessedImage,
        transformlist = registration$fwdtransforms, interpolator = "linear",
        verbose = verbose )
      mask <- antsApplyTransforms( fixed = templateImage, moving = mask,
        transformlist = registration$fwdtransforms, interpolator = "genericLabel",
        verbose = verbose )
      }
    }

  # Do bias correction
  biasField <- NULL
  if( doBiasCorrection == TRUE )
    {
    if( verbose )
      {
      message( "Preprocessing:  bias correction.\n" )
      }
    n4Output <- NULL
    if( is.null( mask ) )
      {
      n4Output <- n4BiasFieldCorrection( preprocessedImage, shrinkFactor = 4, returnBiasField = returnBiasField, verbose = verbose )
      } else {
      n4Output <- n4BiasFieldCorrection( preprocessedImage, mask, shrinkFactor = 4, returnBiasField = returnBiasField, verbose = verbose )
      }
    if( returnBiasField == TRUE )
      {
      biasField <- n4Output
      } else {
      preprocessedImage <- n4Output
      }
    }

  # Denoising
  if( doDenoising == TRUE )
    {
    if( verbose )
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
  if( ! is.null( referenceImage ) && ! is.null( intensityatchingType ) )
    {
    if( verbose )
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
    if( verbose )
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

  returnList <- list( preprocessedImage = preprocessedImage )
  if( ! is.null( mask ) )
    {
    returnList[["brainMask"]] <- mask
    }
  if( ! is.null( biasField ) )
    {
    returnList[["biasField"]] <- biasField
    }
  if( ! is.null( transforms ) )
    {
    returnList[["templateTransforms"]] <- transforms
    }
  return( returnList )
  }
