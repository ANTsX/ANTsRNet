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

