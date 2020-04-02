#' Image intensity normalization using linear regression.
#'
#' Image intensity normalization by regressing the image
#' intensities of the reference image with the source image.
#'
#' @param sourceImage image whose intensities we will match to the
#'                    \code{referenceImage} intensities.
#' @param referenceImage defines the reference intensity function.
#' @param maskImage
#' @param polyOrder of polynomial fit.  Default is 1 (linear fit).
#' @param truncate boolean which turns on/off the clipping of intensities.
#' @return the \code{sourceImage} matched to the \code{referenceImage}.
#' @author Avants BB
#'
#' @examples
#' library(ANTsRCore)
#' sourceImage <- antsImageRead( getANTsRData( "r16" ) )
#' referenceImage <- antsImageRead( getANTsRData( "r64" ) )
#' matchedImage <- regressionMatchImage( sourceImage, referenceImage )
#' bad_source = sourceImage[1:200, 1:200]
#' testthat::expect_error(regressionMatchImage( bad_source, referenceImage ))
#' @export
regressionMatchImage <- function( sourceImage, referenceImage,
  maskImage = NULL, polyOrder = 1, truncate = TRUE )
  {
  if( any( dim( sourceImage ) != dim( referenceImage ) ) )
    {
    stop( "Images do not have the same dimension." )
    }

  sourceIntensities <- c()
  referenceIntensities <- c()
  if( ! is.null( maskImage ) )
    {
    sourceIntensities <- as.numeric( sourceImage[maskImage >= 0.5] )
    referenceIntensities <- as.numeric( referenceImage[maskImage >= 0.5] )
    } else {
    sourceIntensities <- as.numeric( sourceImage )
    referenceIntensities <- as.numeric( referenceImage )
    }

  model <- lm( referenceIntensities ~
    stats::poly( sourceIntensities, polyOrder ) )
  matchedSourceIntensities <- predict( model )

  if( truncate )
    {
    minReferenceValue <-  min( referenceIntensities )
    maxReferenceValue <-  max( referenceIntensities )
    matchedSourceIntensities[matchedSourceIntensities < minReferenceValue] <-
      minReferenceValue
    matchedSourceIntensities[matchedSourceIntensities > maxReferenceValue] <-
      maxReferenceValue
    }

  matchedSourceImage <- makeImage( dim( sourceImage ), matchedSourceIntensities )
  matchedSourceImage <- antsCopyImageInfo( sourceImage,  matchedSourceImage )

  return( matchedSourceImage )
  }
