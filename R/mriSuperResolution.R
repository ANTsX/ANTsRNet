#' Super-resolution for MRI
#'
#' Perform super-resolution of MRI data using deep back projection 
#' network.  Work described in
#'  
#'  https://www.medrxiv.org/content/10.1101/2023.02.02.23285376v1
#'   
#'  with the GitHub repo located at https://github.com/stnava/siq
#'  
#'  Note that some preprocessing possibilities for the input includes:
#'    * Truncate intensity (see ants.iMath(..., 'TruncateIntensity', ...)
#'
#' @param image magnetic resonance image
#' @param expansionFactor Specifies the increase in resolution per 
#' dimension.  Possibilities include: \code{c(1,1,2)}, \code{c(1,1,3)}, 
#' \code{c(1,1,4)}, \code{c(1,1,6)}, \code{c(2,2,2)}, and \code{c(2,2,4)}.
#' @param feature "grader" or "vgg"
#' @param targetRange Range for applySuperResolutionModel.
#' @param polyOrder int or the string 'hist'.  Parameter for regression 
#' matching or specification of histogram matching.
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
mriSuperResolution <- function( image, expansionFactor = c( 1, 1, 2 ),
  feature = "vgg", targetRange = c( 1, 0 ), polyOrder = "hist", verbose = FALSE )
{

  if( image@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  networkBasename <- paste0( "sig_smallshort_train_",
                        paste( expansionFactor, collapse = "x" ),
                        '_1chan_feat', feature, 'L6_best_mdl')
  modelAndWeightsFileName <- getPretrainedNetwork( networkBasename )
  modelSR <- tensorflow::tf$keras$models$load_model( modelAndWeightsFileName, compile = FALSE )

  imageSR <- applySuperResolutionModelToImage( image, modelSR, targetRange = targetRange, 
                   regressionOrder = NA, verbose = verbose )
  if( is.numeric( polyOrder ) || polyOrder == "hist" )
    {
    if( verbose )
      {
      cat( "Match intensity with ", polyOrder, "\n" )
      }
    if( polyOrder == "hist" )
      {
      if( verbose )
        {
        cat( "Histogram match input/output images.\n" )
        }
      imageSR <- histogramMatchImage( imageSR, image )
      } else {
      if( verbose )
        {
        cat( "Regression match input/output images.\n" )
        }
      imageResampled <- resampleImageToTarget( image, imageSR )
      imageSR = regressionMatchImage( imageSR, imageResampled, polyOrder = polyOrder )
      }
    }

  return( imageSR )
}

