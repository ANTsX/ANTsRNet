#' Creates 2D partial convolution layer
#'
#' Creates 2D partial convolution layer as described in the paper
#'
#'   \url{https://arxiv.org/abs/1804.07723}
#'
#' with the implementation ported from the following python implementation
#'
#'   \url{https://github.com/MathiasGruber/PConv-Keras}
#'
#' @docType class
#'
#' @section Arguments:
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
#' @return a partial convolution layer 2D
#'
#' @name PartialConv2DLayer
NULL

#' @export
PartialConv2DLayer <- R6::R6Class( "PartialConv2DLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    dimensionality = NULL,

    filters = NULL,

    kernel_size = NULL,

    strides = c( 1L, 1L ),

    padding = "valid",

    data_format = "channels_last",

    dilation_rate = c( 1L, 1L ),

    activation = NULL,

    use_bias = TRUE,

    kernel_initializer = "glorot_uniform",

    bias_initializer = "zeros",

    kernel_regularizer = NULL,

    bias_regularizer = NULL,

    activity_regularizer = NULL,

    initialize = function()
      {
      },

    build = function( input_shape )
      {
      K <- tensorflow::tf$keras$backend

      if( self$data_format == 'channels_first' )
        {
        self$channelAxis <- 2
        } else {
        self$channelAxis <- 4
        }

      if( is.null( input_shape[[channelAxis]] ) )
        {
        stop('The channel dimension of the inputs should be defined. Found `None`.')
        }
      dimensionality <- as.integer( length( input_shape ) )

      # Image kernel
      kernel_shape <- list( self$kernel_size[1], self$kernel_size[2],
                            dimensionality, self$filters )
      self$kernel <- self$add_weight( shape = kernel_shape,
                                      initializer = self$kernel_initializer,
                                      name = "image_kernel",
                                      regularizer = self$kernel_regularizer,
                                      constraint = self.kernel_constraint )
      # Mask kernel
      self$kernelMask <- K$ones( shape = kernel_shape )

      # Calculate padding size to achieve zero-padding
      self$pconvPadding <- list(
         c( floor( ( self$kernel_size[1]-1 ) / 2 ), floor( ( self$kernel_size[1] - 1 ) / 2 ) ),
         c( floor( ( self$kernel_size[2]-1 ) / 2 ), floor( ( self$kernel_size[2] - 1 ) / 2 ) )
      )

     self$windowSize <- self$kernel_size[1] * self$kernel_size[2]

     if( self$use_bias )
       {
       self$bias <- self$add_weight( shape = shape( self$filters ),
                                     initializer = self$bias_initializer,
                                     name = 'bias',
                                     regularizer = self$bias_regularizer,
                                     constraint = self$bias_constraint )
       } else {
       self$bias <- NULL
       }
     },

   call = function( inputs, mask = NULL )
     {
     K <- tensorflow::tf$keras$backend

     # Both image and mask must be supplied
     if( ! is.list( inputs ) || length( inputs ) != 2 )
       {
       stop( "PartialConvolution2D must be called on a list of two tensors [img, mask]" )
       }

     # Padding done explicitly so that padding becomes part of the masked partial convolution
     images <- K$spatial_2d_padding( inputs[[1]], self$pconvPadding, self$data_format )
     masks <- K$spatial_2d_padding( inputs[[2]], self$pconvPadding, self$data_format )

     # Apply convolutions to mask
     maskOutput <- K$conv2d(
         masks, self$kernelMask,
         strides = self$strides,
         padding = 'valid',
         data_format = self$data_format,
         dilation_rate = self$dilation_rate
     )

     # Apply convolutions to image
     imageOutput <- K$conv2d(
         images * masks, self$kernel,
         strides = self$strides,
         padding = 'valid',
         data_format = self$data_format,
         dilation_rate = self$dilation_rate
     )

     # Calculate the mask ratio on each pixel in the output mask
     maskRatio <- self$windowSize / ( maskOutput + 1e-8 )

     # Clip output to be between 0 and 1
     maskOutput <- K$clip( maskOutput, 0, 1 )

     # Remove ratio values where there are holes
     maskRatio <- maskRatio * maskOutput

     # Normalize iamge output
     imageOutput <- imageOutput * maskRatio

     # Apply bias only to the image (if chosen to do so)
     if( self$use_bias )
       {
       imageOutput <- K$bias_add( imageOutput, self$bias, data_format = self$data_format )
       }

     # Apply activations on the image
     if( ! is.null( self$activation ) )
       {
       imageOutput = self$activation( imageOutput )
       }

     return( list( img_output, mask_output ) )
     },

   compute_output_shape = function( self, input_shape )
     {
     newShape <- NULL
     if( self$data_format == "channels_first" )
       {
       new_shape = list( input_shape[[1]],
                         self$filters,
                         floor( input_shape[[3]] / self$strides[1] + 1 ),
                         floor( input_shape[[4]] / self$strides[2] + 1 )
                       )
       } else if( self$data_format == "channels_last" ) {
       new_shape = list( input_shape[[1]],
                         floor( input_shape[[2]] / self$strides[1] + 1 ),
                         floor( input_shape[[3]] / self$strides[2] + 1 ),
                         self$filters
                       )
       }
     return( list( newShape, newShape ) )
     }
  )
)

#' Partial convolution layer 2D
#'
#' Creates an 2D partial convolution layer
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param filters number of filters
#' @param kernel_size kernel size
#' @param strides strides
#' @param padding padding
#' @param data_format format
#' @param dilation_rate dilate rate
#' @param activation activation
#' @param kernel_initializer kernel initializer
#' @param bias_initializer bias initializer
#' @param kernel_regularizer kernel regularizer
#' @param bias_regularizer bias regularizer
#' @param activity_regularizer activity regularizer
#' @param use_bias use bias
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
layer_partial_convolution_2d <- function( object, filters, kernel_size,
    strides = c( 1L, 1L ), padding = "valid",
    data_format = "channels_last", dilation_rate = c( 1L, 1L ), activation = NULL,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    use_bias = TRUE, trainable = TRUE )
    {
    create_layer( PartialConv2DLayer, object,
      list( filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
            data_format = data_format, dilation_rate = dilation_rate, activation = activation,
            kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer, use_bias = use_bias, trainable = trainable )
       )
    }


#' Creates 3D partial convolution layer
#'
#' Creates 3D partial convolution layer as described in the paper
#'
#'   \url{https://arxiv.org/abs/1804.07723}
#'
#' with the implementation ported from the following python implementation
#'
#'   \url{https://github.com/MathiasGruber/PConv-Keras}
#'
#' @docType class
#'
#' @section Arguments:
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
#' @return a partial convolution layer 3D
#'
#' @name PartialConv3DLayer
NULL

#' @export
PartialConv3DLayer <- R6::R6Class( "PartialConv3DLayer",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    dimensionality = NULL,

    filters = NULL,

    kernel_size = NULL,

    strides = c( 1L, 1L, 1L),

    padding = "valid",

    data_format = "channels_last",

    dilation_rate = c( 1L, 1L, 1L ),

    activation = NULL,

    use_bias = TRUE,

    kernel_initializer = "glorot_uniform",

    bias_initializer = "zeros",

    kernel_regularizer = NULL,

    bias_regularizer = NULL,

    activity_regularizer = NULL,

    initialize = function()
      {
      },

    build = function( input_shape )
      {
      K <- tensorflow::tf$keras$backend

      if( self$data_format == 'channels_first' )
        {
        self$channelAxis <- 2
        } else {
        self$channelAxis <- 5
        }

      if( is.null( input_shape[[channelAxis]] ) )
        {
        stop('The channel dimension of the inputs should be defined. Found `None`.')
        }
      dimensionality <- as.integer( length( input_shape ) )

      # Image kernel
      kernel_shape <- list( self$kernel_size[1], self$kernel_size[2], self$kernel_size[3],
                            dimensionality, self$filters )
      self$kernel <- self$add_weight( shape = kernel_shape,
                                      initializer = self$kernel_initializer,
                                      name = "image_kernel",
                                      regularizer = self$kernel_regularizer,
                                      constraint = self.kernel_constraint )
      # Mask kernel
      self$kernelMask <- K$ones( shape = kernel_shape )

      # Calculate padding size to achieve zero-padding
      self$pconvPadding <- list(
         c( floor( ( self$kernel_size[1]-1 ) / 2 ), floor( ( self$kernel_size[1] - 1 ) / 2 ) ),
         c( floor( ( self$kernel_size[2]-1 ) / 2 ), floor( ( self$kernel_size[2] - 1 ) / 2 ) ),
         c( floor( ( self$kernel_size[3]-1 ) / 2 ), floor( ( self$kernel_size[3] - 1 ) / 2 ) )
      )

     self$windowSize <- self$kernel_size[1] * self$kernel_size[2] * self$kernel_size[3]

     if( self$use_bias )
       {
       self$bias <- self$add_weight( shape = shape( self$filters ),
                                     initializer = self$bias_initializer,
                                     name = 'bias',
                                     regularizer = self$bias_regularizer,
                                     constraint = self$bias_constraint )
       } else {
       self$bias <- NULL
       }
     },

   call = function( inputs, mask = NULL )
     {
     K <- tensorflow::tf$keras$backend

     # Both image and mask must be supplied
     if( ! is.list( inputs ) || length( inputs ) != 2 )
       {
       stop( "PartialConvolution2D must be called on a list of two tensors [img, mask]" )
       }

     # Padding done explicitly so that padding becomes part of the masked partial convolution
     images <- K$spatial_3d_padding( inputs[[1]], self$pconvPadding, self$data_format )
     masks <- K$spatial_3d_padding( inputs[[2]], self$pconvPadding, self$data_format )

     # Apply convolutions to mask
     maskOutput <- K$conv3d(
         masks, self$kernelMask,
         strides = self$strides,
         padding = 'valid',
         data_format = self$data_format,
         dilation_rate = self$dilation_rate
     )

     # Apply convolutions to image
     imageOutput <- K$conv3d(
         images * masks, self$kernel,
         strides = self$strides,
         padding = 'valid',
         data_format = self$data_format,
         dilation_rate = self$dilation_rate
     )

     # Calculate the mask ratio on each pixel in the output mask
     maskRatio <- self$windowSize / ( maskOutput + 1e-8 )

     # Clip output to be between 0 and 1
     maskOutput <- K$clip( maskOutput, 0, 1 )

     # Remove ratio values where there are holes
     maskRatio <- maskRatio * maskOutput

     # Normalize iamge output
     imageOutput <- imageOutput * maskRatio

     # Apply bias only to the image (if chosen to do so)
     if( self$use_bias )
       {
       imageOutput <- K$bias_add( imageOutput, self$bias, data_format = self$data_format )
       }

     # Apply activations on the image
     if( ! is.null( self$activation ) )
       {
       imageOutput = self$activation( imageOutput )
       }

     return( list( img_output, mask_output ) )
     },

   compute_output_shape = function( self, input_shape )
     {
     newShape <- NULL
     if( self$data_format == "channels_first" )
       {
       new_shape = list( input_shape[[1]],
                         self$filters,
                         floor( input_shape[[3]] / self$strides[1] + 1 ),
                         floor( input_shape[[4]] / self$strides[2] + 1 ),
                         floor( input_shape[[5]] / self$strides[3] + 1 )
                       )
       } else if( self$data_format == "channels_last" ) {
       new_shape = list( input_shape[[1]],
                         floor( input_shape[[2]] / self$strides[1] + 1 ),
                         floor( input_shape[[3]] / self$strides[2] + 1 ),
                         floor( input_shape[[4]] / self$strides[3] + 1 ),
                         self$filters
                       )
       }
     return( list( newShape, newShape ) )
     }
  )
)

#' Partial convolution layer 3D
#'
#' Creates an 3D partial convolution layer
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param filters number of filters
#' @param kernel_size kernel size
#' @param strides strides
#' @param padding padding
#' @param data_format format
#' @param dilation_rate dilate rate
#' @param activation activation
#' @param kernel_initializer kernel initializer
#' @param bias_initializer bias initializer
#' @param kernel_regularizer kernel regularizer
#' @param bias_regularizer bias regularizer
#' @param activity_regularizer activity regularizer
#' @param use_bias use bias
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
layer_partial_convolution_3d <- function( object, filters, kernel_size,
    strides = c( 1L, 1L, 1L ), padding = "valid",
    data_format = "channels_last", dilation_rate = c( 1L, 1L, 1L ), activation = NULL,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    use_bias = TRUE, trainable = TRUE )
    {
    create_layer( PartialConv3DLayer, object,
      list( filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
            data_format = data_format, dilation_rate = dilation_rate, activation = activation,
            kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer, use_bias = use_bias, trainable = trainable )
       )
    }


