#' Implementation of the "SCFN" architecture for Brain/Gender prediction
#'
#' Creates a keras model implementation of the Simple Fully Convolutional
#' Network model from the FMRIB group:
#'
#'         \url{https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param numberOfFiltersPerLayer number of filters for the convolutional layers
#'
#' @param doDropout
#'
#' @return a SCFN keras model
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#'
#' model <- createSimpleFullyConvolutionalNeuralNetworkModel3D( list( NULL, NULL, NULL, 1 ) )
#'
#' @import keras
#' @export
createSimpleFullyConvolutionalNeuralNetworkModel3D <- function( 
  inputImageSize, 
  numberOfFiltersPerLayer = c( 32, 64, 128, 256, 256, 64, 40 ),
  doDropout = TRUE )
{
  numberOfLayers <- length( numberOfFiltersPerLayer )

  input <- layer_input( shape = inputImageSize )

  output <- input
  for( i in seq_len( numberOfLayers - 1 ) ) 
    {
    if(  i < numberOfLayers - 1 ) 
      {
      output <- output %>% layer_conv_3d( numberOfFiltersPerLayer[i], 
        kernel_size = c( 3L, 3L, 3L ), padding = "valid" )  
      output <- output %>% layer_batch_normalization()
      output <- output %>% layer_max_pooling_3d( pool_size = c( 2L, 2L, 2L ), 
        strides = c( 2L, 2L, 2L ) )                    
      output <- output %>% layer_activation_relu()  
      } else {
      output <- output %>% layer_conv_3d( numberOfFiltersPerLayer[i], 
        kernel_size = c( 3L, 3L, 3L ), padding = "valid" )  
      output <- output %>% layer_batch_normalization()
      output <- output %>% layer_activation_relu()  
      }
    }

  output <- output %>% layer_average_pooling_3d( pool_size = c( 5L, 6L, 5L ) )
   
  if( doDropout == TRUE )
    {
    output <- output %>% layer_dropout( rate = 0.5 )  
    }

  output <- output %>% layer_conv_3d( numberOfFiltersPerLayer[numberOfLayers], 
    kernel_size = c( 1L, 1L, 1L ), padding = "same" )  
  output <- output %>%
    layer_dense( units = 1, activation = 'linear' )

  model <- keras_model( inputs = input, outputs = output )

  return( model )
}

