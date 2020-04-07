#' noBrainerBrainExtraction
#'
#' Perform "NoBrainer" brain extraction using U-net and FreeSurfer
#' training data.  Training and weights ported from
#'
#'  https://github.com/neuronets/nobrainer-models
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
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' imageN4 <- n4BiasFieldCorrection( image )
#' brainMask <- noBrainerBrainExtraction( imageN4, verbose = TRUE )
#' }
#' @export
noBrainerBrainExtraction <- function( image, outputDirectory = NULL, verbose = FALSE )
  {
  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  if( verbose == TRUE )
    {
    cat( "NoBrainer:  generating network.\n")
    }
  model <- createNoBrainerUnetModel3D( list( NULL, NULL, NULL, 1 ) )

  weightsFileName <- paste0( outputDirectory, "/noBrainerWeights.h5" )
  if( ! file.exists( weightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "NoBrainer:  downloading model weights.\n" )
      }
    weightsUrl <- "https://github.com/neuronets/nobrainer-models/releases/download/0.1/brain-extraction-unet-128iso-weights.h5"
    download.file( weightsUrl, weightsFileName, quiet = !verbose )
    }
  model$load_weights( weightsFileName )

  imageArray <- as.array( image )
  imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
  thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
  thresholdedMask <- thresholdImage( image, -10000, thresholdValue, 0, 1 )
  thresholdedImage <- image * thresholdedMask

  imageResampled <- resampleImage( image, rep( 256, 3 ), useVoxels = TRUE )
  imageArray <- array( as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )

  if( verbose == TRUE )
    {
    cat( "NoBrainer:  predicting mask.\n" )
    }
  brainMaskArray <- predict( model, imageArray )
  brainMaskResampled <- as.antsImage( brainMaskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
  brainMaskImage = resampleImage( brainMaskResampled, dim( image ),
    useVoxels = TRUE, interpType = "nearestneighbor" )
  minimumBrainVolume <- round( 649933.7 / prod( antsGetSpacing( image ) ) )
  brainMaskLabeled = labelClusters( brainMaskImage, minimumBrainVolume )

  return( brainMaskLabeled )
  }
