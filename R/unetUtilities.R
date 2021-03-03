#' One-hot encoding function
#'
#' Function for translating the segmentations to a one-hot representation.
#'
#' @param segmentationsArray an array of shape (\code{batchSize}, \code{width},
#' \code{height}, \code{<depth>})
#' @param segmentationLabels vector of segmentation labels.  Note that a
#' background label (typically 0) needs to be included.
#'
#' @return an n-D array of shape
#' \eqn{ batchSize \times width \times height \times <depth> \times numberOfSegmentationLabels }
#'
#' @author Tustison NJ
#' @export
encodeUnet <- function( segmentationsArray, segmentationLabels = NULL )
{
  if( segmentationLabels == NULL )
    {
    segmentationLabels <- order( unique( segmentationsArray ) )
    }
  numberOfLabels <- length( segmentationLabels )

  dimSegmentations <- dim( segmentationsArray )

  imageDimension <- 2
  if( length( dimSegmentations ) == 4 )
    {
    imageDimension <- 3
    }

  if( numberOfLabels < 2 )
    {
    stop( "At least two segmentation labels need to be specified." )
    }

  oneHotArray <- array( 0, dim = c( dimSegmentations, numberOfLabels ) )
  for( i in seq_len( numberOfLabels ) )
    {
    perLabel <- segmentationsArray
    perLabel[which( segmentationsArray == segmentationLabels[i] )] <- 1L
    perLabel[which( segmentationsArray != segmentationLabels[i] )] <- 0L
    if( imageDimension == 2 )
      {
      oneHotArray[,,,i] <- perLabel
      } else {
      oneHotArray[,,,,i] <- perLabel
      }
    }
  return( oneHotArray )
}

#' Decoding function for the u-net prediction outcome
#'
#' Function for translating the U-net predictions to ANTsR probability
#' images.
#'
#' @param yPredicted an array of shape (\code{batchSize}, \code{width},
#' \code{height}, \code{<depth>}, \code{numberOfSegmentationLabels})
#' @param domainImage image definining the geometry of the returned probability
#' images.
#'
#' @return a list of list of probability images.
#'
#' @author Tustison NJ
#' @importFrom utils tail
#' @importFrom stats predict
#' @importFrom stats kmeans
#' @importFrom magrittr %>%
#' @importFrom ANTsRCore as.antsImage
#' @export
decodeUnet <- function( yPredicted, domainImage )
{
  batchSize <- dim( yPredicted )[1]
  numberOfLabels <- tail( dim( yPredicted ), 1 )

  imageDimension <- 2
  if( length( dim( yPredicted ) ) == 5 )
    {
    imageDimension <- 3
    }

  batchProbabilityImages <- list()
  for( i in seq_len( batchSize ) )
    {
    probabilityImages <- list()
    for( j in seq_len( numberOfLabels ) )
      {
      if( imageDimension == 2 )
        {
        imageArray <- yPredicted[i,,,j]
        } else {
        imageArray <- yPredicted[i,,,,j]
        }
      probabilityImages[[j]] <- as.antsImage( imageArray,
        reference = domainImage )
      }
    batchProbabilityImages[[i]] <- probabilityImages
    }
  return( batchProbabilityImages )
}




