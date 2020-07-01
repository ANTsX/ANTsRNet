#' BrainAGE
#'
#' Estimate BrainAge from a T1-weighted MR image using the DeepBrainNet
#' architecture and weights described here:
#'
#' \url{https://github.com/vishnubashyam/DeepBrainNet}
#'
#' and described in the following article:
#'
#' \url{https://academic.oup.com/brain/article-abstract/doi/10.1093/brain/awaa160/5863667?redirectedFrom=fulltext}
#'
#'
#' @param image input 3-D T1-weighted brain image.
#' @param doPreprocessing boolean dictating whether prescribed
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
#' estimatedBrainAge <- brainAge( image )
#' }
#' @export
brainAge <- function( image, doPreprocessing = TRUE,
  outputDirectory = NULL, verbose = TRUE )
  {
  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  preprocessedImage <- image
  if( doPreprocessing == TRUE )
    {
    # Perform preprocessing
    preprocessing <- preprocessBrainImage( image,
      truncateIntensity = c( 0.01, 0.99 ),
      doBrainExtraction = TRUE, doBiasCorrection = TRUE,
      returnBiasField = FALSE, doDenoising = FALSE,
      templateTransformType = "AffineFast", template = "croppedMni152",
      outputDirectory = outputDirectory, verbose = verbose )
    preprocessedImage <- preprocessing$preprocessedImage * preprocessing$brainMask
    }

  preprocessedImage <- ( preprocessedImage - min( preprocessedImage ) ) /
    ( max( preprocessedImage ) - min( preprocessedImage ) )

  # Load the model and weights

  modelWeightsFileName <- paste0( outputDirectory, "/DeepBrainNetModel.h5" )
  if( ! file.exists( modelWeightsFileName ) )
    {
    if( verbose == TRUE )
      {
      message( "Brain age (DeepBrainNet):  downloading model weights.\n" )
      }
    modelWeightsFileName <- getPretrainedNetwork( "brainAgeDeepBrainNet", modelWeightsFileName )
    }
  if( verbose == TRUE )
    {
    message( "Brain age (DeepBrainNet):  loading model.\n" )
    }
  model <- load_model_hdf5( modelWeightsFileName )

  # The paper only specifies that 80 slices are used for prediction.  I just picked
  # a reasonable range spanning the center of the brain

  whichSlices <- seq( from = 46, to = 125 )

  batchX <- array( data = 0, dim = c( length( whichSlices ), dim( preprocessedImage )[1:2], 3 ) )

  for( i in seq.int( length( whichSlices ) ) )
    {

    # The model requires a three-channel input.  The paper doesn't specify but I'm
    # guessing that the previous and next slice are included.

    batchX[i,,,1] <- as.array( extractSlice( preprocessedImage, whichSlices[i] - 1, 3 ) )
    batchX[i,,,2] <- as.array( extractSlice( preprocessedImage, whichSlices[i], 3 ) )
    batchX[i,,,3] <- as.array( extractSlice( preprocessedImage, whichSlices[i] + 1, 3 ) )
    }

  if( verbose == TRUE )
    {
    message( "Brain age (DeepBrainNet):  predicting brain age per slice.\n" )
    }
  brainAgePerSlice <- model %>% predict( batchX, verbose = verbose )

  predictedAge <- median( brainAgePerSlice )

  return( list( predictedAge = predictedAge, brainAgePerSlice = brainAgePerSlice ) )
  }


