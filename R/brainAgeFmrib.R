#' brainAge FMRIB
#'
#' Estimate BrainAge from a T1-weighted MR image using the SCFN
#' architecture and weights described here:
#'
#' \url{https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain}
#'
#' and described in the following article:
#'
#' \url{https://doi.org/10.1101/2019.12.17.879346}
#'
#' Note that age prediction for this particular set of weights is 
#' in the range 42-82 years trained from the biobank data.  The
#' authors mention a separate data set from the PAC 2019 challenge
#' where the range is 14-94 years but, as far as I can tell, those
#' weights aren't available.
#'
#' @param image input 3-D T1-weighted brain image.
#' @param skipPreprocessing boolean dictating whether prescribed
#' preprocessing is performed (brain extraction, bias correction,
#' normalization to template).
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return predicted age and binned confidence values
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' brainAge <- brainAgeFmrib( image )
#' }
#' @export
brainAgeFmrib <- function( image, skipPreprocessing = FALSE, 
  outputDirectory = NULL, verbose = FALSE )
  {
  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  preprocessedImage <- image
  if( skipPreprocessing == FALSE )
    {
    # Perform preprocessing
    preprocessing <- preprocessBrainImage( image, 
      truncateIntensity = c( 0.01, 0.99 ), 
      doBrainExtraction = TRUE, doBiasCorrection = TRUE, 
      returnBiasField = FALSE, doDenoising = FALSE, 
      templateTransformType = "Affine", template = "biobank",
      intensityNormalizationType = NULL, outputDirectory = outputDirectory, 
      verbose = verbose )
    preprocessedImage <- preprocessing$preprocessedImage * preprocessing$brainMask
    }  
  # Note that this is the only intensity normalization I could find in the repo.
  # The authors don't mention it in the paper or elsewhere.  

  preprocessedImage <- preprocessedImage / mean( preprocessedImage )

  # Construct the model and load the weights

  model <- createSimpleFullyConvolutionalNeuralNetworkModel3D( list( NULL, NULL, NULL, 1 ) )

  weightsFileName <- paste0( outputDirectory, "/brainAgeFmribWeights.h5" )
  if( ! file.exists( weightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Brain age (FMRIB):  downloading model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "brainAgeFmrib", weightsFileName )
    }
  model$load_weights( weightsFileName )

  batchX <- array( data = 0, dim = c( 1, dim( preprocessedImage ), 1 ) )
  batchX[1,,,,1] <- as.array( preprocessedImage )
  prediction <- model$predict( batchX, verbose = verbose )
  ageConfidences <- drop( prediction )

  # Each bin (of 40) represents the 42-82 years of age
  ageYears <- seq( from = 42, to = 82, length.out = length( ageConfidences ) )
  predictedAge = sum( ageConfidences * ageYears )

  return( list( predictedAge = predictedAge, ageConfidences = ageConfidences ) )
  }


