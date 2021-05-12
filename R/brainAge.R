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
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * brain extraction, and
#'    * affine registration to MNI.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param image input 3-D T1-weighted brain image.
#' @param doPreprocessing boolean dictating whether prescribed
#' preprocessing is performed (brain extraction, bias correction,
#' normalization to template).
#' @param numberOfSimulations number of random affine perturbations to
#' transform the input.
#' @param sdAffine define the standard deviation of the affine transformation
#' parameter for the simulations.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
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
  numberOfSimulations = 0, sdAffine = 0.01, antsxnetCacheDirectory = NULL, verbose = TRUE )
  {
  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  preprocessedImage <- image
  if( doPreprocessing == TRUE )
    {
    # Perform preprocessing
    preprocessing <- preprocessBrainImage( image,
      truncateIntensity = c( 0.01, 0.99 ),
      brainExtractionModality = "t1", doBiasCorrection = TRUE,
      returnBiasField = FALSE, doDenoising = TRUE,
      templateTransformType = "AffineFast", template = "croppedMni152",
      antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )
    preprocessedImage <- preprocessing$preprocessedImage * preprocessing$brainMask
    }

  preprocessedImage <- ( preprocessedImage - min( preprocessedImage ) ) /
    ( max( preprocessedImage ) - min( preprocessedImage ) )

  # Load the model and weights

  modelWeightsFileName <- getPretrainedNetwork( "brainAgeDeepBrainNet" )
  model <- load_model_hdf5( modelWeightsFileName )

  # The paper only specifies that 80 slices are used for prediction.  I just picked
  # a reasonable range spanning the center of the brain

  whichSlices <- seq( from = 46, to = 125 )

  batchX <- array( data = 0, dim = c( length( whichSlices ), dim( preprocessedImage )[1:2], 3 ) )

  if( numberOfSimulations > 0 )
    {
    dataAugmentation <-
      randomlyTransformImageData( preprocessedImage,
      list( list( preprocessedImage ) ),
      numberOfSimulations = numberOfSimulations,
      transformType = 'affine',
      sdAffine = sdAffine,
      inputImageInterpolator = 'linear' )
    }

  brainAgePerSlice <- c()
  for( i in seq.int( numberOfSimulations + 1 ) )
    {
    batchImage <- preprocessedImage
    if( i > 1 )
      {
      batchImage <- dataAugmentation$simulatedImages[[i-1]][[1]]
      }
    for( j in seq.int( length( whichSlices ) ) )
      {
      slice <- as.array( extractSlice( batchImage, whichSlices[j], 3 ) )
      batchX[j,,,1] <- slice
      batchX[j,,,2] <- slice
      batchX[j,,,3] <- slice
      }
    if( verbose == TRUE )
      {
      message( "Brain age (DeepBrainNet):  predicting brain age per slice (batch = ", i, ").\n" )
      }

    if( i == 1 )
      {
      brainAgePerSlice <- model %>% predict( batchX, verbose = verbose )
      } else {
      prediction <- model %>% predict( batchX, verbose = verbose )
      brainAgePerSlice <- brainAgePerSlice + ( prediction - brainAgePerSlice ) / ( i + 1 )
      }
    }

  predictedAge <- median( brainAgePerSlice )

  return( list( predictedAge = predictedAge, brainAgePerSlice = brainAgePerSlice ) )
  }


