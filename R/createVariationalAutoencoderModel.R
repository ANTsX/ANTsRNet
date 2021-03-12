#' Function for creating 2-D MMD variational autoencoder model.
#'
#'
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#'
#' @export
createMMDVariationalAutoencoderModel2D <- function( inputImageSize,
                                                    latentDimension = 2L,
                                                    numberOfEncodingLayers = 2L,
                                                    numberOfFiltersAtBaseLayer = 64L,
                                                    kernelSize = 4L,
                                                    strides = 2L,
                                                    numberOfDenseUnits = 1024L )
{
  # Build the encoder

  encoderInput <- layer_input( shape = inputImageSize )
  encoderBranch <- encoderInput

  for( i in seq_len( numberOfEncodingLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )
    encoderBranch <- encoderBranch %>% layer_conv_2d(
       filters = numberOfFilters, kernel_size = kernelSize,
       strides = strides, padding = 'same' )
    encoderBranch <- encoderBranch %>% layer_activation_leaky_relu( 0.1 )
    }
  encoderBranch <- encoderBranch %>% layer_flatten()
  encoderBranch <- encoderBranch %>% layer_dense( numberOfDenseUnits )
  encoderBranch <- encoderBranch %>% layer_activation_leaky_relu( 0.1 )

  encoderOutput <- encoderBranch %>% layer_dense( latentDimension )

  encoder <- keras_model( inputs = encoderInput, outputs = encoderOutput )

  # Build the decoder

  decoderKernelShape <- c( 7L, 7L )

  decoderInput <- layer_input( shape = c( latentDimension ) )
  decoderBranch <- decoderInput
  decoderBranch <- decoderBranch %>% layer_dense( numberOfDenseUnits, activation = "relu" )
  decoderBranch <- decoderBranch %>% layer_dense( prod( decoderKernelShape ) * numberOfFilters, activation = "relu" )
  decoderBranch <- decoderBranch %>% layer_reshape( c( decoderKernelShape, numberOfFilters ) )

  # for( i in seq_len( numberOfEncodingLayers - 1 ) )
  #   {
  #   numberOfFilters <- numberOfFiltersAtBaseLayer / ( 2 ^ i ) )
  #   decoderBranch <- decoderBranch %>% layer_conv_2d_transpose(
  #      filters = numberOfFilters, kernel_size = kernelSize,
  #      strides = strides, activation = "relu", padding = 'same' )
  #   }

  # decoderBranch <- decoderBranch %>% layer_conv_2d_transpose(
  #    filters = 1, kernel_size = kernelSize,
  #    strides = strides, activation = "sigmoid", padding = 'same' )

  # decoder <- keras_model( inputs = decoderInput, outputs = decoderOutput )


}
