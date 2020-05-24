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
#' @param numberOfBins number of bins for final softmax output.
#' @param dropoutRate dropout rate before final convolution layer. 
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
  numberOfFiltersPerLayer = c( 32, 64, 128, 256, 256, 64 ),
  numberOfBins = 40,
  dropoutRate = 0.5 )
{
  numberOfLayers <- length( numberOfFiltersPerLayer )

  input <- layer_input( shape = inputImageSize )

  output <- input
  for( i in seq_len( numberOfLayers ) ) 
    {
    if( i < numberOfLayers ) 
      {
      output <- output %>% layer_conv_3d( numberOfFiltersPerLayer[i], 
        kernel_size = c( 3L, 3L, 3L ), padding = "valid" )  
      output <- output %>% layer_zero_padding_3d( padding = c( 1L, 1L, 1L ) )
      output <- output %>% layer_batch_normalization( momentum = 0.1, epsilon = 1e-5 )
      output <- output %>% layer_max_pooling_3d( pool_size = c( 2L, 2L, 2L ), 
        strides = c( 2L, 2L, 2L ) )                    
      } else {
      output <- output %>% layer_conv_3d( numberOfFiltersPerLayer[i], 
        kernel_size = c( 1L, 1L, 1L ), padding = "valid" )  
      output <- output %>% layer_batch_normalization( momentum = 0.1, epsilon = 1e-5 )
      }
    output <- output %>% layer_activation_relu()  
    }

  output <- output %>% layer_average_pooling_3d( pool_size = c( 5L, 6L, 5L ),
    strides = c( 5L, 6L, 5L ) )
   
  if( dropoutRate > 0.0 )
    {
    output <- output %>% layer_dropout( rate = dropoutRate )  
    }

  output <- output %>% layer_conv_3d( numberOfBins, 
    kernel_size = c( 1L, 1L, 1L ), padding = "valid" )  
  # output <- output %>% layer_activation_log_softmax()
  output <- output %>% tensorflow::tf$keras$layers$Lambda( tensorflow::tf$log_softmax_v2 )
 
  model <- keras_model( inputs = input, outputs = output )

  return( model )
}

