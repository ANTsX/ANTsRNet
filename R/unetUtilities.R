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
  if( is.null( segmentationLabels ) )
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
    perLabel <- array( 0, dim = dim( segmentationsArray ))
    perLabel[which( segmentationsArray == segmentationLabels[i] )] <- 1L
    if( imageDimension == 2 )
      {
      oneHotArray[,,,i] <- perLabel
      } else {
      oneHotArray[,,,,i] <- perLabel
      }
    }
  return( oneHotArray )
}

