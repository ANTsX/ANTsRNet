#' Extract 2-D or 3-D image patches.
#'
#' @param image Input ANTs image
#' @param patchSize Width, height, and depth (if 3-D) of patches.
#' @param maxNumberOfPatches Maximum number of patches returned.  If
#' "all" is specified, then all patches in sequence (defined by the
#" strideLength are extracted.
#' @param strideLength Defines the sequential patch overlap for
#' maxNumberOfPatches = all.  Can be a image-dimensional vector or a scalar.
#' @param maskImage optional image specifying the sampling region for
#' the patches when \code{maximumNumberOfPatches} does not equal "all".
#' @param randomSeed integer seed that allows reproducible patch extraction
#' across runs.
#' @param returnAsArray specifies the return type of the function.  If
#' \code{FALSE} (default) the return type is a list where each element is
#' a single patch.  Otherwise the return type is an array of size
#' \code{dim( numberOfPatches, patchSize )}.
#'
#' @return a list (or array) of image patches.
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsR )
#'
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' maskImage <- getMask( image, 1, 1000 )
#' patchSet1 <- extractImagePatches( image, c( 32, 32 ), 10, c( 32, 32 ), randomSeed = 0 )
#' patchSet2 <- extractImagePatches( image, c( 32, 32 ), 10, c( 32, 32 ), randomSeed = 1 )
#' patchSet3 <- extractImagePatches( image, c( 32, 32 ), 10, c( 32, 32 ), maskImage, randomSeed = 0 )
#'
#' @export
extractImagePatches <- function( image, patchSize, maxNumberOfPatches = 'all',
  strideLength = 1, maskImage = NA, randomSeed, returnAsArray = FALSE )
{
  if ( ! missing( randomSeed ) )
    {
    set.seed( randomSeed )
    }
  imageSize <- dim( image )
  dimensionality <- length( imageSize )

  if( length( imageSize ) != length( patchSize ) )
    {
    stop( "Mismatch between the image size and the specified patch size.\n" )
    }
  if( any( patchSize > imageSize ) )
    {
    stop( "Patch size is greater than the image size.\n")
    }

  imageArray <- as.array( image )

  patchList <- list()
  patchArray <- array( data = NA )
  midPatchIndex <- round( patchSize / 2 )

  numberOfExtractedPatches <- maxNumberOfPatches

  if( tolower( maxNumberOfPatches ) == 'all' )
    {
    strideLengthVector <- strideLength
    if( length( strideLength ) == 1 )
      {
      strideLengthVector <- rep.int( strideLength, dimensionality )
      } else if( length( strideLength ) != dimensionality ) {
      stop( paste0( "strideLength is not a scalar or vector of
        length dimensionality." ) )
      } else if( any( strideLength < 1 ) ) {
      stop( paste0( "strideLength must be a positive integer." ) )
      }

    numberOfExtractedPatches <- 1

    indices <- list()
    for( d in seq_len( dimensionality ) )
      {
      indices[[d]] <- seq.int( from = 1, to = imageSize[d] - patchSize[d] + 1,
        by = strideLengthVector[d] )
      numberOfExtractedPatches <- numberOfExtractedPatches * length( indices[[d]] )
      }

    if( returnAsArray )
      {
      patchArray <- array( data = NA,
        dim = c( numberOfExtractedPatches, patchSize ) )
      }

    count <- 1
    if( dimensionality == 2 )
      {
      for( i in indices[[1]] )
        {
        for( j in indices[[2]] )
          {
          startIndex <- c( i, j )
          endIndex <- startIndex + patchSize - 1

          patch <- imageArray[startIndex[1]:endIndex[1],
            startIndex[2]:endIndex[2]]

          if( returnAsArray )
            {
            patchArray[count,,] <- patch
            } else {
            patchList[[count]] <- patch
            }

          count <- count + 1
          }
        }
      } else if( dimensionality == 3 ) {
      for( i in indices[[1]] )
        {
        for( j in indices[[2]] )
          {
          for( k in indices[[3]] )
            {
            startIndex <- c( i, j, k )
            endIndex <- startIndex + patchSize - 1

            patch <- imageArray[startIndex[1]:endIndex[1],
              startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]]

            if( returnAsArray )
              {
              patchArray[count,,,] <- patch
              } else {
              patchList[[count]] <- patch
              }

            count <- count + 1
            }
          }
        }
      } else {
      stop( "Unsupported dimensionality." )
      }
    } else {

    randomIndices <- array( data = NA, dim = c( maxNumberOfPatches, dimensionality ) )
    if( !is.na( maskImage ) )
      {
      maskArray <- as.array( maskImage )
      maskIndices <- which( maskArray != 0, arr.ind = TRUE )

      shiftedMaskIndices <- maskIndices
      negativeIndices <- c()
      for( d in seq_len( dimensionality ) )
        {
        shiftedMaskIndices[, d] <- maskIndices[, d] + patchSize[d]
        negativeIndices <- append( negativeIndices, which( shiftedMaskIndices[, d] > imageSize[d] ) )
        shiftedMaskIndices[, d] <- maskIndices[, d] - midPatchIndex[d]
        negativeIndices <- append( negativeIndices, which( shiftedMaskIndices[, d] <= 0 ) )
        shiftedMaskIndices[, d] <- maskIndices[, d] + midPatchIndex[d]
        negativeIndices <- append( negativeIndices, which( shiftedMaskIndices[, d] > imageSize[d] ) )
        }

      negativeIndices <- unique( negativeIndices )
      if( length( negativeIndices ) > 0 )
        {
        maskIndices <- maskIndices[-negativeIndices,]
        }
      numberOfExtractedPatches <- min( maxNumberOfPatches, nrow( maskIndices ) )

      randomIndices <- maskIndices[
        sample.int( nrow( maskIndices ), numberOfExtractedPatches ),]
      } else {
      for( d in seq_len( dimensionality ) )
        {
        randomIndices[, d] <- sample.int(
          imageSize[d] - patchSize[d] + 1, maxNumberOfPatches, replace = TRUE )
        }
      }

    if( returnAsArray )
      {
      patchArray <- array( data = NA,
        dim = c( numberOfExtractedPatches, patchSize ) )
      }

    startIndex <- rep( 1, dimensionality )
    for( i in seq_len( numberOfExtractedPatches ) )
      {
      startIndex <- randomIndices[i,]
      endIndex <- startIndex + patchSize - 1

      if( dimensionality == 2 )
        {
        patch <- imageArray[startIndex[1]:endIndex[1],
          startIndex[2]:endIndex[2]]
        } else if( dimensionality == 3 ) {
        patch <- imageArray[startIndex[1]:endIndex[1],
          startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]]
        } else {
        stop( "Unsupported dimensionality." )
        }

      if( returnAsArray )
        {
        if( dimensionality == 2 )
          {
          patchArray[i,,] <- patch
          } else {
          patchArray[i,,,] <- patch
          }
        } else {
        patchList[[i]] <- patch
        }
      }
    }

  if( returnAsArray )
    {
    return( patchArray )
    } else {
    return( patchList )
    }
}

#' Reconstruct image from a list of patches.
#'
#' @param patchList List of overlapping patches defining an image.
#' @param domainImage Image or mask to define the geometric information of the
#' reconstructed image.  If this is a mask image, the reconstruction will only
#' use patches in the mask.
#' @param strideLength Defines the sequential patch overlap for
#' maxNumberOfPatches = all.  Can be a image-dimensional vector or a scalar.
#' @param domainImageIsMask boolean specifying whether the domain image is a
#' mask used to limit the region of reconstruction from the patches.
#'
#' @return an ANTs image.
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsR )
#'
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' patchSet <- extractImagePatches( image, c( 64, 64 ), "all", c( 8, 8 ) )
#' imageReconstructed <-
#'   reconstructImageFromPatches( patchSet, image, c( 8, 8 ) )
#'
#' @importFrom ANTsRCore as.antsImage
#' @export
reconstructImageFromPatches <- function( patchList, domainImage,
  strideLength = 1, domainImageIsMask = FALSE )
{
  imageSize <- dim( domainImage )
  dimensionality <- length( imageSize )
  imageArray <- array( data = 0, dim = imageSize )

  patchSize <- dim( patchList[[1]] )
  midPatchIndex <- round( patchSize / 2 )

  strideLengthVector <- strideLength
  if( length( strideLength ) == 1 )
    {
    strideLengthVector <- rep.int( strideLength, dimensionality )
    } else if( length( strideLength ) != dimensionality ) {
    stop( paste0( "strideLength is not a scalar or vector of
      length dimensionality." ) )
    } else if( any( strideLength < 1 ) ) {
    stop( paste0( "strideLength must be a positive integer." ) )
    }

  if( domainImageIsMask )
    {
    maskArray <- as.array( domainImage )
    maskArray[maskArray != 0] <- 1
    }

  count <- 1
  if( dimensionality == 2 )
    {
    if( all( strideLengthVector == 1 ) )
      {
      for( i in seq_len( imageSize[1] - patchSize[1] + 1 ) )
        {
        for( j in seq_len( imageSize[2] - patchSize[2] + 1 ) )
          {
          startIndex <- c( i, j )
          endIndex <- startIndex + patchSize - 1

          doAdd <- TRUE
          if( domainImageIsMask )
            {
            if( maskArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]]
              [midPatchIndex[1], midPatchIndex[2]] == 0 )
              {
              doAdd <- FALSE
              }
            }

          if( doAdd )
            {
            imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] <-
              imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] +
              patchList[[count]]
            }
          count <- count + 1
          }
        }
      if( !domainImageIsMask )
        {
        for( i in seq_len( imageSize[1] ) )
          {
          for( j in seq_len( imageSize[2] ) )
            {
            factor <- min( i, patchSize[1], imageSize[1] - i + 1 ) *
              min( j, patchSize[2], imageSize[2] - j + 1 )

            imageArray[i, j] <- imageArray[i, j] / factor
            }
          }
        }
      } else {
      countArray <- array( 0, dim = dim( imageArray ) )
      for( i in seq.int( from = 1, to = imageSize[1] - patchSize[1] + 1,
        by = strideLengthVector[1] ) )
        {
        for( j in seq.int( from = 1, to = imageSize[2] - patchSize[2] + 1,
          by = strideLengthVector[2] ) )
          {
          startIndex <- c( i, j )
          endIndex <- startIndex + patchSize - 1

          imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] <-
            imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] +
            patchList[[count]]
          countArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] <-
            countArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] +
            array( data = 1, dim = patchSize )
          count <- count + 1
          }
        }
      for( i in seq_len( imageSize[1] ) )
        {
        for( j in seq_len( imageSize[2] ) )
          {
          factor <- max( 1, countArray[i, j] )
          imageArray[i, j] <- imageArray[i, j] / factor
          }
        }
      }
    } else if( dimensionality == 3 ) {
    if( all( strideLengthVector == 1 ) )
      {
      for( i in seq_len( imageSize[1] - patchSize[1] + 1 ) )
        {
        for( j in seq_len( imageSize[2] - patchSize[2] + 1 ) )
          {
          for( k in seq_len( imageSize[3] - patchSize[3] + 1 ) )
            {
            startIndex <- c( i, j, k )
            endIndex <- startIndex + patchSize - 1

            doAdd <- TRUE
            if( domainImageIsMask )
              {
              if( maskArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2],
                startIndex[3]:endIndex[3]][midPatchIndex[1], midPatchIndex[2], midPatchIndex[3]] == 0 )
                {
                doAdd <- FALSE
                }
              }

            if( doAdd )
              {
              imageArray[startIndex[1]:endIndex[1],
                startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]] <-
                imageArray[startIndex[1]:endIndex[1],
                  startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]] +
                patchList[[count]]
              count <- count + 1
              }
            }
          }
        }

      if( !domainImageIsMask )
        {
        for( i in seq_len( imageSize[1] ) )
          {
          for( j in seq_len( imageSize[2] ) )
            {
            for( k in seq_len( imageSize[3] ) )
              {
              factor <- min( i, patchSize[1], imageSize[1] - i + 1 ) *
                min( j, patchSize[2], imageSize[2] - j + 1 ) *
                min( k, patchSize[3], imageSize[3] - k + 1 )

              imageArray[i, j, k] <- imageArray[i, j, k] / factor
              }
            }
          }
        }
      } else {
      countArray <- array( 0, dim = dim( imageArray ) )
      for( i in seq.int( from = 1, to = imageSize[1] - patchSize[1] + 1,
        by = strideLengthVector[1] ) )
        {
        for( j in seq.int( from = 1, to = imageSize[2] - patchSize[2] + 1,
          by = strideLengthVector[2] ) )
          {
          for( k in seq.int( from = 1, to = imageSize[3] - patchSize[3] + 1,
            by = strideLengthVector[3] ) )
            {
            startIndex <- c( i, j, k )
            endIndex <- startIndex + patchSize - 1

            imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2],
                startIndex[3]:endIndex[3]] <-
              imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2],
                startIndex[3]:endIndex[3]] + patchList[[count]]
            countArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2],
                startIndex[3]:endIndex[3]] <-
              countArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2],
                startIndex[3]:endIndex[3]] + array( data = 1, dim = patchSize )
            count <- count + 1
            }
          }
        }
      for( i in seq_len( imageSize[1] ) )
        {
        for( j in seq_len( imageSize[2] ) )
          {
          for( k in seq_len( imageSize[3] ) )
            {
            factor <- max( 1, countArray[i, j, k] )
            imageArray[i, j, k] <- imageArray[i, j, k] / factor
            }
          }
        }
      }
    } else {
    stop( "Unsupported dimensionality.\n" )
    }

  return( as.antsImage( imageArray, reference = domainImage ) )
}

#' Mean square error of a single image or between two images.
#'
#' @param x input image.
#' @param y input image.
#'
#' @return the mean squared error
#' @author Avants BB (from redr)
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' mseValue <- MSE( r16, r85 )
#'
#' @export
MSE <- function( x, y = NULL )
{
  if( is.null( y ) )
    {
    return( mean( x^2 ) )
    } else {
    return( mean ( ( x - y )^2 ) )
    }
}

#' Mean absolute error of a single image or between two images.
#'
#' @param x input image.
#' @param y input image.
#'
#' @return the mean absolute error
#' @author Avants BB (from redr)
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' maeValue <- MAE( r16, r85 )
#'
#' @export
MAE <- function( x, y = NULL )
{
  if( is.null( y ) )
    {
    return( mean( abs( x ) ) )
    } else {
    return( mean ( abs( x - y ) ) )
    }
}

#' Peak signal-to-noise ratio between two images.
#'
#' @param x input image.
#' @param y input image.
#'
#' @return the peak signal-to-noise ratio
#' @author Avants BB
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' psnrValue <- PSNR( r16, r85 )
#'
#' @export
PSNR <- function( x, y )
{
  return( 20 * log10( max( x ) ) - 10 * log10( MSE( x, y ) ) )
}

#' Structural similarity index (SSI) between two images.
#'
#' Implementation of the SSI quantity for two images proposed in
#'
#' Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli. "Image quality
#' assessment: from error visibility to structural similarity". IEEE TIP.
#' 13 (4): 600–612.
#'
#' @param x input image.
#' @param y input image.
#' @param K vector of length 2 which contain SSI parameters meant to stabilize
#' the formula in case of weak denominators.
#'
#' @return the structural similarity index
#' @author Avants BB
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' ssimValue <- SSIM( r16, r85 )
#'
#' @export
SSIM <- function( x, y, K = c( 0.01, 0.03 ) )
{
  globalMax <- max( max( x ), max( y ) )
  globalMin <- abs( min( min( x ), min( y ) ) )
  L <- globalMax - globalMin

  C1 <- ( K[1] * L )^2
  C2 <- ( K[2] * L )^2
  C3 <- C2 / 2

  mu_x <- mean( x )
  mu_y <- mean( y )

  mu_x_sq <- mu_x * mu_x
  mu_y_sq <- mu_y * mu_y
  mu_xy <- mu_x * mu_y

  sigma_x_sq <- mean( x * x ) - mu_x_sq
  sigma_y_sq <- mean( y * y ) - mu_y_sq
  sigma_xy <- mean( x * y ) - mu_xy

  numerator <- ( 2 * mu_xy + C1 ) * ( 2 * sigma_xy + C2 )
  denominator <- ( mu_x_sq + mu_y_sq + C1 ) * ( sigma_x_sq + sigma_y_sq + C2 )

  SSI <- numerator / denominator

  return( SSI )
}
