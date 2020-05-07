#' brainExtraction
#'
#' Perform brain extraction using U-net and ANTs-based
#' training data.
#'
#' @param image input 3-D T1-weighted brain image.
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return brain probability mask (ANTsR image)
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' probabilityMask <- brainExtraction( image )
#' }
#' @export
brainExtraction <- function( image, modality = "t1", outputDirectory = NULL, verbose = FALSE )
  {
  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  classes <- c( "background", "brain" )
  numberOfClassificationLabels <- length( classes )
  imageModalities <- c( "T1" )
  channelSize <- length( imageModalities )

  reorientTemplateFileName <- paste0( outputDirectory, "/S_template3_resampled.nii.gz" )
  if( ! file.exists( reorientTemplateFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Brain extraction:  downloading template.\n" )
      }
    reorientTemplateUrl <- "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Template/S_template3_resampled.nii.gz?raw=true"
    download.file( reorientTemplateUrl, reorientTemplateFileName, quiet = !verbose )
    }
  reorientTemplate <- antsImageRead( reorientTemplateFileName )
  resampledImageSize <- dim( reorientTemplate )

  unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels,
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5 )

  weightsFileName <- paste0( outputDirectory, "/brainExtractionWeights.h5" )
  if( ! file.exists( weightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Brain extraction:  downloading model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "brainExtraction", weightsFileName )
    }
  unetModel$load_weights( weightsFileName )

  if( verbose == TRUE )
    {
    cat( "Brain extraction:  normalizing image to the template.\n" )
    }
  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass( image )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
    center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate )
  warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )

  batchX <- array( data = as.array( warpedImage ),
    dim = c( 1, resampledImageSize, channelSize ) )
  batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

  if( verbose == TRUE )
    {
    cat( "Brain extraction:  prediction and decoding.\n" )
    }
  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

  if( verbose == TRUE )
    {
    cat( "Brain extraction:  renormalize probability mask to native space.\n" )
    }
  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    probabilityImagesArray[[1]][[2]], image )

  return( probabilityImage )
  }
