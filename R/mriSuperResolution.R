#' Super-resolution for MRI
#'
#' Perform super-resolution (2x) of MRI data using deep back projection network.
#'
#' @param image magnetic resonance image
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return super-resolution image.
#' @author Avants BB
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' imageSr <- mriSuperResolution( image )
#' }
#' @export
mriSuperResolution <- function( image, outputDirectory = NULL, verbose = FALSE )
{
  if( image@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  modelAndWeightsFileName <- paste0( outputDirectory, "mindmapsSR_16_ANINN222_0.h5" )
  if( ! file.exists( modelAndWeightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "MRI super-resolution:  downloading model weights.\n" )
      }
    modelAndWeightsFileName <- getPretrainedNetwork( "mriSuperResolution", modelAndWeightsFileName )
    }
  modelSR <- load_model_hdf5( modelAndWeightsFileName )

  imageSR <- applySuperResolutionModelToImage( image, modelSR, targetRange = c( -127.5, 127.5 ) )
  imageSR = regressionMatchImage( imageSR, resampleImageToTarget( image, imageSR ), polyOrder = 1 )

  return( imageSR )
}

