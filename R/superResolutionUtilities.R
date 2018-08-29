#' Model loss function for super-resolution---peak-signal-to-noise ratio.
#'
#' Based on the keras loss function (losses.R):
#'
#'    \url{https://github.com/rstudio/keras/blob/master/R/losses.R}
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#'
#' @details Loss functions are to be supplied in the loss parameter of the
#' \code{compile()} function.
#'
#' @export
peak_signal_to_noise_ratio <- function( y_true, y_pred )
{
  K <- keras::backend()

  return( 10 * K$log( K$mean( K$square( y_pred - y_true ) ) ) / K$log( 10 ) )
}
attr( peak_signal_to_noise_ratio, "py_function_name" ) <-
  "peak_signal_to_noise_ratio"

#' Peak-signal-to-noise ratio.
#'
#' @param y_true true encoded labels
#' @param y_pred predicted encoded labels
#'
#' @rdname loss_peak_signal_to_noise_ratio_error
#' @export
loss_peak_signal_to_noise_ratio_error <- function( y_true, y_pred )
{
  return( -peak_signal_to_noise_ratio( y_true, y_pred ) )
}
attr( loss_peak_signal_to_noise_ratio_error, "py_function_name" ) <-
  "peak_signal_to_noise_ratio_error"

#' Extract 2-D or 3-D image patches.
#'
#' @param image Input ANTs image
#' @param patchSize Width, height, and depth (if 3-D) of patches.
#' @param maxNumberOfPatches Maximum number of patches returned.  If
#' "all" is specified, then all overlapping patches are extracted.
#'
#' @return a randomly selected list of patches.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' }
#' @export
extractImagePatches <- function( image, patchSize, maxNumberOfPatches = 'all' )
{
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

  patches <- list()

  if( tolower( maxNumberOfPatches ) == 'all' )
    {
    count <- 1
    if( dimensionality == 2 )
      {
      for( i in seq_len( inputImageSize[1] - patchSize[1] + 1 ) )
        {
        for( j in seq_len( inputImageSize[2] - patchSize[2] + 1 ) )
          {
          startIndex <- c( i, j )
          endIndex <- startIndex + patchSize - 1
          patches[[count]] <-
            imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]]
          count <- count + 1
          }
        }
      } else if( dimensionality == 3 ) {
      for( i in seq_len( inputImageSize[1] - patchSize[1] + 1 ) )
        {
        for( j in seq_len( inputImageSize[2] - patchSize[2] + 1 ) )
          {
          for( k in seq_len( inputImageSize[3] - patchSize[3] + 1 ) )
            {
            startIndex <- c( i, j, k )
            endIndex <- startIndex + patchSize - 1
            patches[[count]] <- imageArray[startIndex[1]:endIndex[1],
              startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]]
            count <- count + 1
            }
          }
        }
      } else {
      stop( "Unsupported dimensionality.\n" )
      }
    } else {
    startIndex <- rep( 0, dimensionality )
    for( i in seq_len( maxNumberOfPatches ) )
      {
      for( d in seq_len( dimensionality ) )
        {
        startIndex[d] <- sample.int( imageSize[d] - patchSize[d] + 1, 1 )
        }

      endIndex <- startIndex + patchSize - 1
      if( dimensionality == 2 )
        {
        patches[[i]] <-
          imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]]
        } else if( dimensionality == 3 ) {
        patches[[i]] <- imageArray[startIndex[1]:endIndex[1],
          startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]]
        } else {
        stop( "Unsupported dimensionality.\n" )
        }
      }
    }
  return( patches )
}

#' Reconstruct image from a list of patches.
#'
#' @param patchList list of overlapping patches defining an image.
#' @param domainImage Image to define the geometric information of the
#' reconstructed image.
#'
#' @return an ANTs image.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' }
#' @importFrom ANTsRCore as.antsImage
#' @export
reconstructImageFromPatches <- function( patchList, domainImage )
{
  imageSize <- dim( domainImage )
  dimensionality <- length( imageSize )  
  patchSize <- dim( patchList[[1]] )

  numberOfPatches <- 1
  for( d in 1:dimensionality )
    {
    numberOfPatches <- numberOfPatches *
      ( imageSize[d] - patchSize[d] + 1 )
    }
  if( numberOfPatches != length( patchList ) )
    {
    stop( "Not the right number of patches.\n" )
    }

  imageArray <- array( data = 0, dim = imageSize )
  
  count <- 1
  if( dimensionality == 2 )  
    {
    for( i in seq_len( imageSize[1] - patchSize[1] + 1 ) )
      {
      for( j in seq_len( imageSize[2] - patchSize[2] + 1 ) )
        {
        startIndex <- c( i, j )
        endIndex <- startIndex + patchSize - 1
        
        imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] <- 
          imageArray[startIndex[1]:endIndex[1], startIndex[2]:endIndex[2]] + 
          patchList[[count]]
        count <- count + 1  
        }
      }  

    for( i in seq_len( imageSize[1] ) )
      {
      for( j in seq_len( imageSize[2] ) )
        {
        factor <- min( i, patchSize[1], imageSize[1] - i + 1 ) * 
          min( j, patchSize[2], imageSize[2] - j + 1 ) 

        imageArray[i, j] <- imageArray[i, j] / factor
        }
      }  

    } else if( dimensionality == 3 ) {
    for( i in seq_len( imageSize[1] - patchSize[1] + 1 ) )
      {
      for( j in seq_len( imageSize[2] - patchSize[2] + 1 ) )
        {
        for( k in seq_len( imageSize[3] - patchSize[3] + 1 ) )
          {
          startIndex <- c( i, j, k )
          endIndex <- startIndex + patchSize - 1

          imageArray[startIndex[1]:endIndex[1],
            startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]] <- 
            imageArray[startIndex[1]:endIndex[1],
            startIndex[2]:endIndex[2], startIndex[3]:endIndex[3]] +
            patchList[[count]]
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
          factor <- min( i, patchSize[1], imageSize[1] - i + 1 ) * 
            min( j, patchSize[2], imageSize[2] - j + 1 ) *
            min( k, patchSize[3], imageSize[3] - k + 1 )

          imageArray[i, j, k] <- imageArray[i, j, k] / factor
          count <- count + 1  
          }  
        }
      }
    } else {
    stop( "Unsupported dimensionality.\n" )
    }

  return( as.antsImage( imageArray, reference = domainImage ) )
}
