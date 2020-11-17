#' Cortical thickness using deep learning
#'
#' Perform KellyKapowski cortical thickness using \code{deepAtropos} for
#' segmentation.  Description concerning implementaiton and evaluation:
#'
#'  \url{https://www.medrxiv.org/content/10.1101/2020.10.19.20215392v1}
#'
#' @param t1 input 3-D unprocessed T1-weighted brain image
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return Cortical thickness image and segmentation probability images.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' kk <- corticalThickness( image )
#' }
#' @export
corticalThickness <- function( t1, antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( t1@dimension != 3 )
    {
    stop( "Input image must be 3-D" )
    }

  atropos <- deepAtropos( t1, doPreprocessing = TRUE,
    antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )

  # Kelly Kapowski cortical thickness

  kkSegmentation <- atropos$segmentationImage
  kkSegmentation[kkSegmentation == 4] <- 3
  grayMatter <- atropos$probabilityImages[[3]]
  whiteMatter <- atropos$probabilityImages[[4]] + atropos$probabilityImages[[5]]
  kk <- kellyKapowski( s = kkSegmentation, g = grayMatter, w = whiteMatter,
                      its = 45, r = 0.025, m = 1.5, x = 0, verbose = verbose )

  return( list(
          thicknessImage = kk,
          csfProbabilityImage = atropos$probabilityImages[[2]],
          grayMatterProbabilityImage = atropos$probabilityImages[[3]],
          whiteMatterProbabilityImage = atropos$probabilityImages[[4]],
          deepGrayMatterProbabilityImage = atropos$probabilityImages[[5]],
          brainStemProbabilityImage = atropos$probabilityImages[[6]],
          cerebellumProbabilityImage = atropos$probabilityImages[[7]]
        ) )
  }
