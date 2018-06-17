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
#' @param maxNumberOfPatches Maximum number of patches returned.
#' 
#' @return a randomly selected list of patches.
#' @author Tustison NJ
#' @examples
#' \dontrun{ 
#' }
#' @export
extractImagePatches <- function( image, patchSize, maxNumberOfPatches = 1 )
{
  inputImageSize <- dim( image )
  dimensionality <- length( inputImageSize )
  
  if( length( inputImageSize ) != length( patchSize ) )
    {
    stop( "Mismatch between the image size and the specified patch size.\n" )  
    }
  if( any( patchSize > inputImageSize ) ) 
    {
    stop( "Patch size is greater than the image size.\n")  
    }

  imageArray <- as.array( image )

  patches = list()

  startIndex <- rep( 0, dimensionality )
  for( i in 1:maxNumberOfPatches )
    {
    for( d in 1:dimensionality )
      {
      startIndex[d] <- sample.int( 1, inputImageSize[d] - patchSize[d] + 1 )
      }

    endIndex <- startIndex + patchSize - 1
    if( dimensionality == 2 )
      {
      patches[[i]] <- 
        imageArray[startIndex[1]:endIndex[1],startIndex[2]:endIndex[2]]
      } else if( dimensionality == 3 ) {
      patches[[i]] <- imageArray[startIndex[1]:endIndex[1],
        startIndex[2]:endIndex[2],startIndex[3]:endIndex[3]]
      } else {
      stop( "Unsupported dimensionality.\n" )  
      }
    }
  return( patches )   
}

