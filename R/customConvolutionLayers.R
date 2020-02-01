#' Creates a 2-D attention augmented convolutional layer
#'
#' Creates a 2-D attention augmented convolutional layer as described in the paper
#'
#'   \url{https://arxiv.org/abs/1904.09925}
#'
#' with the implementation ported from the following python implementation
#'
#'   \url{https://github.com/titu1994/keras-attention-augmented-convs/blob/master/}
#'
#' @param inputLayer input keras layer.
#' @param numberOfOutputFilters number of output filters.
#' @param kernelSize convolution kernel size.
#' @param strides convolution strides.
#' @param kDepth Defines the number of filters for \code{k}.  Either absolute
#' or, if \code{kDepth < 1.0}, number of \code{k} filters =
#' \code{kDepth * numberOfOutputFilters}.
#' @param vDepth Defines the number of filters for \code{v}.  Either absolute
#' or, if \code{vDepth < 1.0}, number of \code{v} filters =
#' \code{vDepth * numberOfOutputFilters}.
#' @param numberOfAttentionHeads number of attention heads.  Note that
#' \code{as.integer(kDepth/numberOfAttentionHeads)>0} (default = 8).
#' @param useRelativeEncodings boolean for whether to use relative encodings
#' (default = TRUE).
#'
#' @return a keras tensor
#' @author Tustison NJ
#' @export

layer_augmented_conv_2d <- function( inputLayer,
                                     numberOfOutputFilters,
                                     kernelSize = c( 3, 3 ),
                                     strides = c( 1, 1 ),
                                     kDepth = 0.2,
                                     vDepth = 0.2,
                                     numberOfAttentionHeads = 8,
                                     useRelativeEncodings = TRUE )
{
  stop( "Not finished yet." )
  channelAxis <- 2L
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1L
    }



}