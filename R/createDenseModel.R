#' Simple multilayer dense network.
#'
#' @param inputVectorSize Specifies the length of the input vector.
#' @param numberOfFiltersAtBaseLayer Number of filters at the initial dense layer.
#' This number is halved for each subsequent layer.
#' @param numberOfLayers Number of dense layers defining the model.
#' @param mode 'classification' or 'regression'.
#' @param numberOfClassificationLabels Specifies output for "classification" networks.
#' @return a keras model
#' @author Tustison NJ
#' @examples
#'
#' library( ANTsRNet )
#'
#' model <- createDenseModel(166)
#'
#' @import keras
#' @export
createDenseModel <- function( inputVectorSize,
                              numberOfFiltersAtBaseLayer = 512,
                              numberOfLayers = 2,
                              mode = 'classification',
                              numberOfClassificationLabels = 1000
                            )
{
  input <- layer_input( shape = c( inputVectorSize ) )

  output <- NULL

  numberOfFilters = numberOfFiltersAtBaseLayer
  for( i in seq.int( numberOfLayers ) )
    {
    if( i == 1 )
      {
      output <- input %>% layer_dense( units = numberOfFilters )
      } else {
      output <- output %>% layer_dense( units = numberOfFilters )
      }

    output <- output %>% layer_activation_leaky_relu( alpha = 0.2 )
    numberOfFilters <- floor( numberOfFilters / 2 )
    }

  if( mode == "classification" )
    {
    output <- output %>% layer_dense( units = numberOfClassificationLabels,
                                     activation = "softmax" )
    } else if( mode == "regression" ) {
    output <- output %>% layer_dense( units = 1, activation = "linear" )
    } else {
    stop( "Unrecognized activation." )
    }

  model <-  keras_model( inputs = input, outputs = output )

  return( model )
}
