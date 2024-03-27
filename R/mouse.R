#' Mouse brain extraction
#'
#' Perform brain extraction of mouse MRI
#'
#' @param image input 3-D brain image (or list of images for multi-modal scenarios).
#' @param modality image type.  Options include:
#' \itemize{
#'   \item{"t2": }{T2-weighted MRI}
#'   \item{"ex5coronal": }{E13.5 and E15.5 mouse embroyonic histology data.}
#'   \item{"ex5sagittal": }{E13.5 and E15.5 mouse embroyonic histology data.}
#' }
#' @param returnIsotropicOutput The network actually learns an interpolating 
#' function specific to the mouse brain.  Setting this to true, the output 
#' images are returned isotropically resampled.
#' @param whichAxis Specify direction for ex5 modalities..
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return brain probability mask 
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "brain.nii.gz" )
#' probabilityMask <- mouseBrainExtraction( image, modality = "t2" )
#' }
#' @export
mouseBrainExtraction <- function( image,
  modality = c( "t2", "ex5coronal", "ex5sagittal" ),
  returnIsotropicOutput = FALSE, whichAxis = 2,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( whichAxis < 1 || whichAxis > 3 )
    {
    stop( "Chosen axis not supported." )

    }

  if( modality == "t2" )
    {
    templateShape <- c( 176, 176, 176 ) 

    template <- antsImageRead( getANTsXNetData( "bsplineT2MouseTemplate" ) )
    template <- resampleImage( template, templateShape, useVoxels = TRUE, interpType = 0 )
    templateMask <- antsImageRead( getANTsXNetData( "bsplineT2MouseTemplateBrainMask" ) )
    templateMask <- resampleImage( template, templateShape, useVoxels = TRUE, interpType = 1 )

    if( verbose )
      {
      message( "Preprocessing:  Warping to B-spline T2w mouse template." )
      }

    centerOfMassReference <- getCenterOfMass( templateMask )
    centerOfMassImage <- getCenterOfMass( image )
    translation <- as.array( centerOfMassImage ) - as.array( centerOfMassReference )
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassReference,
      translation = centerOfMassImage - centerOfMassReference )

    imageWarped <- applyAntsrTransformToImage( xfrm, image, templateMask, interpolation = "linear" )
    imageWarped <- iMath( imageWarped, "Normalize" )

    unetModel <- createUnetModel3D( c( templateShape, 1 ),
      numberOfOutputs = 1, mode = "sigmoid",
      numberOfFilters = c( 16, 32, 64, 128 ), 
      convolutionKernelSize = 3, deconvolutionKernelSize = 2 )
    weightsFileName <- getPretrainedNetwork( "mouseT2wBrainExtraction3D" )
    unetModel$load_weights( weightsFileName )

    batchX <- array( data = 0, dim = c( 1, templateShape, 1 ) )
    batchX[1,,,,1] = as.array( imageWarped )

    if( verbose )
      {
      message( "Prediction." )
      }
    predictedData <- drop( unetModel$predict( batchX, verbose = verbose ) )
    predictedImage <- as.antsImage( predictedData, reference = template )

    probabilityMask <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      predictedImage, image )

    return( probabilityMask )
    }
  }
