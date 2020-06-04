#' Crop the center of an image.
#'
#' @param image Input ANTs image
#' @param cropSize width, height, depth (if 3-D), and time (if 4-D) of the cropped image.
#'
#' @return a cropped image
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsR )
#'
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' croppedImage <- cropImageCenter( image, c( 64, 64 ) )
#'
#' @export
cropImageCenter <- function( image, cropSize )
{
  imageSize <- dim( image )  

  if( length( imageSize ) != length( cropSize ) )
    {
    stop( "cropSize does not match image size." )
    }

  if( any( cropSize > imageSize ) )
    {
    stop( "A cropSize dimension is larger than imageSize." )
    }

  labelImage <- antsImageClone( image ) * 0
  startIndex <- floor( 0.5 * ( imageSize - cropSize ) ) + 1
  endIndex <- startIndex + cropSize - 1

  if( image@dimension == 2 )
    {
    labelImage[startIndex[1]:endIndex[1],
               startIndex[2]:endIndex[2]] <- 1
    } else if( image@dimension == 3 ) {
    labelImage[startIndex[1]:endIndex[1],
               startIndex[2]:endIndex[2],
               startIndex[3]:endIndex[3]] <- 1
    } else if( image@dimension == 4 ) {
    labelImage[startIndex[1]:endIndex[1],
               startIndex[2]:endIndex[2],
               startIndex[3]:endIndex[3],
               startIndex[4]:endIndex[4]] <- 1
    }
  croppedImage <- cropImage( image, labelImage, label = 1 )

  return( croppedImage )
}
