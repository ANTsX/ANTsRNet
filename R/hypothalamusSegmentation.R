#' Hypothalamus segmentation
#'
#' Described here:
#'
#'     \url{https://pubmed.ncbi.nlm.nih.gov/32853816/}
#'
#' with the implementation available at:
#'
#'     \url{https://github.com/BBillot/hypothalamus_seg}
#'
#' @param t1 input 3-D T1-weighted brain image.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return hypothalamic subunit segmentation and probability images
#'    \itemize{
#'       \item{Label 1:  left anterior-inferior}
#'       \item{Label 2:  left anterior-superior}
#'       \item{Label 3:  left posterior}
#'       \item{Label 4:  left tubular inferior}
#'       \item{Label 5:  left tubular superior}
#'       \item{Label 6:  right anterior-inferior}
#'       \item{Label 7:  right anterior-superior}
#'       \item{Label 8:  right posterior}
#'       \item{Label 9:  right tubular inferior}
#'       \item{Label 10: right tubular superior}
#'    }
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' hypo <- hypothalamusSegmentation( image )
#' }
#' @export
hypothalamusSegmentation <- function( t1,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  classes <- c( "background",
                "left anterior-inferior",
                "left anterior-superior",
                "left posterior",
                "left tubular inferior",
                "left tubular superior",
                "right anterior-inferior",
                "right anterior-superior",
                "right posterior",
                "right tubular inferior",
                "right tubular superior"
                )

  ################################
  #
  # Rotate to proper orientation
  #
  ################################

  referenceImage <- makeImage( c( 256, 256, 256 ),
                               voxval = 0,
                               spacing = c( 1, 1, 1 ),
                               origin = c( 0, 0, 0 ),
                               direction = diag( c( -1, -1, 1 ) ) )
  centerOfMassReference <- getCenterOfMass( referenceImage + 1 )
  centerOfMassImage <- getCenterOfMass( t1 * 0 + 1 )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassReference,
        translation = centerOfMassImage - centerOfMassReference )
  xfrmInv <- invertAntsrTransform( xfrm )

  croppedImage <- antsImageClone( t1 ) * 0 + 1
  croppedImage <- applyAntsrTransformToImage( xfrm, croppedImage, referenceImage )
  croppedImage <- cropImage( croppedImage, labelImage = croppedImage, label = 1 )

  t1Warped <- applyAntsrTransformToImage( xfrm, t1, croppedImage )

  ################################
  #
  # Gaussian normalize intensity based on brain mask
  #
  ################################

  t1Warped <- ( t1Warped - min( t1Warped ) ) / ( max( t1Warped ) - min( t1Warped ) )

  ################################
  #
  # Build models and load weights
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Hypothalamus:  retrieving model weights.\n" )
    }

  unetModel <- createHypothalamusUnetModel3D( dim( t1Warped ) )

  weightsFileName <- getPretrainedNetwork( "hypothalamus",
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  unetModel$load_weights( weightsFileName )

  ################################
  #
  # Do prediction
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Prediction.\n" )
    }

  batchX <- array( data = as.array( t1Warped ), dim = c( 1, dim( t1Warped ), 1 ) )

  predictedData <- unetModel$predict( batchX, verbose = verbose )

  probabilityImages <- list()
  for( i in seq_len( length( classes ) ) )
    {
    if( verbose == TRUE )
      {
      cat( "Processing image ", classes[i], "\n" )
      }

    probabilityImage <- as.antsImage( drop( predictedData[1,,,,i] ), reference = t1Warped )
    probabilityImages[[i]] <- applyAntsrTransformToImage( xfrmInv, probabilityImage, t1 )
    }

  imageMatrix <- imageListToMatrix( probabilityImages, t1 * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]] - 1

  results <- list( segmentationImage = segmentationImage,
                   probabilityImages = probabilityImages )

  return( results )
}

