#' Custom scale layer
#'
#' @docType class
#'
#'
#' @section Arguments:
#' \describe{
#'  \item{axis}{integer specifying which axis to normalize.}
#'  \item{momentum}{momentum value used for computation of the exponential
#'                  average of the mean and standard deviation.}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return
#' @examples
#' library(keras)
#' inputImageSize = c( 256L, 256L, 1L )
#' inputs <- keras::layer_input( shape = inputImageSize )
#' outputs <- inputs %>% layer_zero_padding_2d( padding = c( 3L, 3L ) )
#' layer_scale = ANTsRNet:::layer_scale
#' outputs %>%
#'   layer_scale(axis = -1L)
#' lay = ScaleLayer$new()
#' \dontrun{
#' lay$build(input_shape = inputImageSize)
#' }
#' @export
ScaleLayer <- R6::R6Class(
  "ScaleLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    axis = -1L,

    momentum = 0.9,

    initialize = function( axis = -1L, momentum = 0.9 )
    {
      self$axis <- axis
      self$momentum <- momentum
    },

    build = function( input_shape )
    {
      # K <- keras::backend()

      # print("Input shape is ")
      # print(input_shape)
      # input_shape = as.integer(input_shape)
      self$inputShape <- input_shape

      index <- self$axis
      if( self$axis == -1 )
      {
        index <- length( self$inputShape )
      }
      output_shape <- reticulate::tuple( input_shape[[index]] )
      self$gamma <- self$add_weight(
        name = "gamma",
        shape = output_shape,
        initializer = initializer_ones(),
        # initializer = "one",
        trainable = TRUE )
      self$beta <- self$add_weight(
        name = "beta",
        shape = output_shape,
        initializer = initializer_zeros(),
        # initializer = "zero",
        trainable = TRUE )
    },

    call = function( inputs, mask = NULL )
    {
      K <- keras::backend()

      broadcastShape <- as.list( rep( 1L, length( self$inputShape ) ) )

      index <- self$axis
      if( self$axis == -1 )
      {
        index <- length( self$inputShape )
      }
      broadcastShape[[index]] <- self$inputShape[[index]]
      broadcastShape <- K$cast( broadcastShape, dtype = "int32" )

      output <- K$reshape( self$gamma, broadcastShape ) * inputs +
        K$reshape( self$beta, broadcastShape )

      return( output )
    }

  )
)

layer_scale <- function(
  object,
  axis = -1L, momentum = 0.9, name = NULL, trainable = TRUE ) {
  axis = as.integer(axis)
  create_layer( ScaleLayer, object,
                list( axis = axis, momentum = momentum,
                      name = name, trainable = trainable)
  )
}
