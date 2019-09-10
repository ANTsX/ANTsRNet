#' Function for creating a symmetric autoencoder model.
#'
#' Builds an autoencoder based on the specified array definining the
#' number of units in the encoding branch.  Ported to Keras R from the
#' Keras python implementation here:
#'
#' \url{https://github.com/XifengGuo/DEC-keras}
#'
#' @param numberOfUnitsLayer vector defining the number of units
#' in the encoding branch
#' @param activation activation type for the dense layers
#' @param initializer initializer type for the dense layers
#'
#' @return two models:  the encoder and auto-encoder
#'
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' ae <- createAutoencoderModel( c( 784, 500, 500, 2000, 10 ) )
#'
#' @export

createAutoencoderModel <- function( numberOfUnitsPerLayer,
                                    activation = 'relu',
                                    initializer = 'glorot_uniform' )
{
  numberOfEncodingLayers <- length( numberOfUnitsPerLayer ) - 1

  inputs <- layer_input( shape = numberOfUnitsPerLayer[1] )

  encoder <- inputs

  for( i in seq_len( numberOfEncodingLayers - 1 ) )
    {
    encoder <- encoder %>%
      layer_dense( units = numberOfUnitsPerLayer[i+1],
         activation = activation, kernel_initializer = initializer )
    }

  encoder <- encoder %>%
    layer_dense( units = tail( numberOfUnitsPerLayer, 1 ) )

  autoencoder <- encoder

  for( i in seq( from = numberOfEncodingLayers, to = 2, by = -1 ) )
    {
    autoencoder <- autoencoder %>%
      layer_dense( units = numberOfUnitsPerLayer[i],
         activation = activation, kernel_initializer = initializer )
    }

  autoencoder <- autoencoder %>%
    layer_dense( numberOfUnitsPerLayer[1], kernel_initializer = initializer )

  return( list(
    autoencoderModel = keras_model( inputs = inputs, outputs = autoencoder ),
    encoderModel = keras_model( inputs = inputs, outputs = encoder ) ) )
}

