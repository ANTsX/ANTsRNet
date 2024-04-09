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
  dropoutRate = 0.5,
  doExperimentalVariant = FALSE )
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

  if( doExperimentalVariant == TRUE )
    {
    output <- output %>%
      layer_dense( units = 1, activation = 'linear' )
    } else {
    output <- output %>% layer_activation_softmax()  
    }

  model <- keras_model( inputs = input, outputs = output )

  return( model )
}

#' Implementation of the "RMNet" generator architecture for inpainting
#'
#' Creates a keras model implementation of the model:
#'
#'         \url{https://github.com/Jireh-Jam/R-MNet-Inpainting-keras}
#'
#' @return a keras model
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#'
#' model <- createRmnetGenerator()
#'
#' @import keras
#' @export
createRmnetGenerator <- function()
{
  imgShape <- c( 256, 256, 3 )
  imgShapeMask <- c( 256, 256, 1 )
  gf <- 64
  channels <- 3

  # compute inputs 
  inputImg <- layer_input( shape = imgShape, dtype = "float32", name = "image_input" ) 
  inputMask <- layer_input( shape = imgShapeMask, dtype = "float32", name = "mask_input" ) 

  reversedMask <- inputMask %>% layer_lambda( f = function( x )
                                                {
                                                return( 1-x )
                                                } )
  maskedImage <- list( inputImg, reversedMask ) %>% layer_multiply()

  # encoder 
  x <- maskedImage %>% layer_conv_2d( gf, c( 5, 5 ), dilation_rate = 2, input_shape = imgShape, padding = "same", name = "enc_conv_1" )
  x <- x %>% layer_activation_leaky_relu( alpha = 0.2 )
  x <- x %>% layer_batch_normalization( momentum = 0.8 )
  
  pool1 <- x %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )

  x <- pool1 %>% layer_conv_2d( gf, c( 5, 5 ), dilation_rate = 2, padding = "same", name = "enc_conv_2" )
  x <- x %>% layer_activation_leaky_relu( alpha = 0.2 )
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  pool2 <- x %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )

  x <- pool2 %>% layer_conv_2d( gf * 2, c( 5, 5 ), dilation_rate = 2, padding = "same", name = "enc_conv_3" )
  x <- x %>% layer_activation_leaky_relu( alpha = 0.2 )
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  pool3 <- x %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )

  x <- pool3 %>% layer_conv_2d( gf * 4, c( 5, 5 ), dilation_rate = 2, padding = "same", name = "enc_conv_4" )
  x <- x %>% layer_activation_leaky_relu( alpha = 0.2 )
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  pool4 <- x %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )

  x <- pool4 %>% layer_conv_2d( gf * 8, c( 5, 5 ), dilation_rate = 2, padding = "same", name = "enc_conv_5" )
  x <- x %>% layer_activation_leaky_relu( alpha = 0.2 )
  x <- x %>% layer_dropout( 0.5 )

  # decoder
  x <- x %>% layer_upsampling_2d( size = c( 2, 2 ), interpolation = "bilinear" )
  x <- x %>% layer_conv_2d_transpose( gf * 8, c( 3, 3 ), padding = "same", name = "upsample_conv_1" )
  x <- x %>% layer_lambda( f = function( x ) {
                            tensorflow::tf$pad( x, list( c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ) ), 
                            'REFLECT' )   
                            } )
  x <- x %>% layer_activation_relu()
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  x <- x %>% layer_upsampling_2d( size = c( 2, 2 ), interpolation = "bilinear" )
  x <- x %>% layer_conv_2d_transpose( gf * 4, c( 3, 3 ), padding = "same", name = "upsample_conv_2" )
  x <- x %>% layer_lambda( f = function( x ) {
                            tensorflow::tf$pad( x, list( c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ) ), 
                            'REFLECT' )   
                            } )
  x <- x %>% layer_activation_relu()
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  x <- x %>% layer_upsampling_2d( size = c( 2, 2 ), interpolation = "bilinear" )
  x <- x %>% layer_conv_2d_transpose( gf * 2, c( 3, 3 ), padding = "same", name = "upsample_conv_3" )
  x <- x %>% layer_lambda( f = function( x ) {
                            tensorflow::tf$pad( x, list( c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ) ), 
                            'REFLECT' )   
                            } )
  x <- x %>% layer_activation_relu()
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  x <- x %>% layer_upsampling_2d( size = c( 2, 2 ), interpolation = "bilinear" )
  x <- x %>% layer_conv_2d_transpose( gf, c( 3, 3 ), padding = "same", name = "upsample_conv_4" )
  x <- x %>% layer_lambda( f = function( x ) {
                            tensorflow::tf$pad( x, list( c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ), c( 0L, 0L ) ), 
                            'REFLECT' )   
                            } )
  x <- x %>% layer_activation_relu()
  x <- x %>% layer_batch_normalization( momentum = 0.8 )

  x <- x %>% layer_conv_2d_transpose( channels, c( 3, 3 ), padding = "same", name = "final_output" )
  x <- x %>% layer_activation( 'tanh' )

  decodedOutput <- x
  reversedMaskImage <- list( decodedOutput, inputMask ) %>% layer_multiply()
  outputImg <- list( maskedImage, reversedMaskImage ) %>% layer_add()
  concatOutputImg <- list( outputImg, inputMask ) %>% layer_concatenate()
  model <- keras_model( inputs = list( inputImg, inputMask ), outputs = concatOutputImg )

  return( model )
}

