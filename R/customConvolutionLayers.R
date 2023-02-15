#' Compute output length in single dimension for convolution filter
#'
#' Utility function to calculate the output shape of a convolutional
#' filter along a single dimension.  It's based on the python keras
#' utility located here
#'
#' https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py#L118-L141
#'
#' but I can't see to locate it's R analog so it's reproduced explicitly
#' in ANTsRNet.
#'
#' @param inputLength input size along a single dimension.
#' @param filterSize kernel size along a single dimension.
#' @param stride stride length along a single dimension.
#' @param dilation dilation rate along a single dimension.
#' @param padding type of padding.  Can be "same", "valid", "full", or "causal".
#' @return the output size along a single dimension.
#' @author Tustison NJ
#'
#' @examples
#' library( ANTsRNet )
#' outputSize <- convOutputLength( 256, filterSize = 3, stride = 2, padding = "same" )
#' testthat::expect_identical( outputSize, 128 )
#' @export
convOutputLength = function( inputLength, filterSize, stride, dilation = 1,
   padding = c( "same", "valid", "full", "causal" ) )
  {
  if( is.null( inputLength ) )
    {
    return( NULL )
    }
  padding <- match.arg( padding )
  dilatedFilterSize = filterSize + ( filterSize - 1 ) * ( dilation - 1 )

  outputLength <- NULL
  if( padding == "same" || padding == "causal" )
    {
    outputLength <- inputLength
    } else if( padding == "valid" ) {
    outputLength <- inputLength - dilatedFilterSize + 1
    } else if( padding == "full" ) {
    outputLength <- inputLength + dilatedFilterSize - 1
    }
  return( floor( ( outputLength + stride - 1 ) / stride ) )
  }

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

    kernelSize = NULL,

    eps = 1e-6,

    strides = c( 1L, 1L ),

    padding = "valid",

    dataFormat = "channels_last",

    dilationRate = c( 1L, 1L ),

    activation = NULL,

    useBias = TRUE,

    kernelInitializer = "glorot_uniform",

    biasInitializer = "zeros",

    kernelRegularizer = NULL,

    biasRegularizer = NULL,

    activityRegularizer = NULL,

    initialize = function( filters = NULL,
        kernelSize = NULL, eps = 1e-6, strides = c( 1L, 1L ), padding = "valid",
        dataFormat = "channels_last", dilationRate = c( 1L, 1L ), activation = NULL,
        useBias = TRUE, kernelInitializer = "glorot_uniform", biasInitializer = "zeros",
        kernelRegularizer = NULL, biasRegularizer = NULL, activityRegularizer = NULL )
      {
      self$filters <- as.integer( filters )
      self$kernelSize <- as.integer( kernelSize )
      if( length( kernelSize ) == 1 )
        {
        self$kernelSize <- c( self$kernelSize, self$kernelSize )
        }
      self$eps <- eps
      self$strides <- as.integer( strides )
      self$padding <- padding
      self$dataFormat <- dataFormat
      self$dilationRate <- as.integer( dilationRate )
      self$activation <- activation
      self$useBias <- useBias
      self$kernelInitializer <- kernelInitializer
      self$biasInitializer <- biasInitializer
      self$kernelRegularizer <- kernelRegularizer
      self$biasRegularizer <- biasRegularizer
      self$activityRegularizer <- activityRegularizer
      },

    build = function( input_shape )
      {
      if( self$dataFormat == 'channels_first' )
        {
        self$channelAxis <- 2
        } else {
        self$channelAxis <- 4
        }

      self$dimensionality <- as.list( input_shape[[1]] )[[4]]
      # Image kernel
      kernelShape <- list( self$kernelSize[1], self$kernelSize[2],
                            self$dimensionality, self$filters )
      self$kernel <- self$add_weight( shape = kernelShape,
                                      initializer = self$kernelInitializer,
                                      name = "image_kernel",
                                      regularizer = self$kernelRegularizer,
                                      constraint = self$kernelConstraint,
                                      trainable = TRUE )
      # Mask kernel
      # self$maskKernel <- tensorflow::tf$Variable( initial_value = tensorflow::tf$ones( kernelShape ),
      #                                             trainable = FALSE )
      self$maskKernel <- self$add_weight( shape = kernelShape,
                                      initializer = tensorflow::tf$keras$initializers$Ones(),
                                      name = "mask_kernel",
                                      trainable = FALSE )

     if( self$useBias )
       {
       self$bias <- self$add_weight( shape = shape( self$filters ),
                                     initializer = self$biasInitializer,
                                     name = 'bias',
                                     regularizer = self$biasRegularizer,
                                     constraint = self$biasConstraint )
       } else {
       self$bias <- NULL
       }
     },

   call = function( inputs, mask = NULL )
     {
     # Both image and mask must be supplied
     if( ! is.list( inputs ) || length( inputs ) != 2 )
       {
       stop( "PartialConvolution2D must be called on a list of two tensors [img, mask]" )
       }

     features <- inputs[[1]]
     featureMask <- inputs[[2]]
     if( featureMask$shape[[self$channelAxis]] == 1 )
       {
       # featureMask <- tensorflow::tf$repeat( featureMask, featureMask$shape[[self$channelAxis]], axis=self$channelAxis )
       featureMask <- tensorflow::tf$tile( featureMask, list( 1L, 1L, 1L, featureMask$shape[[self$channelAxis]] ) )
       }

     K <- tensorflow::tf$keras$backend

     features <- tensorflow::tf$multiply( features, featureMask )
     features <- K$conv2d( features, self$kernel, strides = self$strides, padding = "same",
                          data_format = self$dataFormat, dilation_rate = self$dilationRate )
     norm <- K$conv2d( featureMask, self$maskKernel, strides = self$strides, padding = 'same',
                          data_format = self$dataFormat, dilation_rate = self$dilationRate )

     # See corresponding note in antspynet as to why I opted for the following workaround.

     featureMaskFanin <- self$kernelSize[1] * self$kernelSize[2]
     for( i in seq.int( 2, featureMaskFanin ) )
       {
       features <- tensorflow::tf$where( tensorflow::tf$equal( norm,
                                                           tensorflow::tf$constant( i, dtype = tensorflow::tf$float32 ) ),
                                         tensorflow::tf$math$divide( features,
                                                           tensorflow::tf$constant( i, dtype = tensorflow::tf$float32 ) ),
                                         features )
       }

     # Apply bias only to the image (if chosen to do so)
     if( self$useBias )
       {
       features <- K$bias_add( features, self$bias, data_format = self$dataFormat )
       }

     # Apply activations on the image
     if( ! is.null( self$activation ) )
       {
       features <- self$activation( features )
       }

     featureMask <- tensorflow::tf$where( tensorflow::tf$greater( norm, self$eps ), 1.0, 0.0 )

     return( list( features, featureMask ) )
     },

   compute_output_shape = function( self, inputShape )
     {
     newSpatialDims <- rep( NA, 2 )
     for( i in seq.int( length( newSpatialDims ) ) )
       {
       index <- i + 1
       if( self$dataFormat == "channels_first" )
         {
         index <- i + 2
         }
       newSpatialDims[i] <- convOutputLength( inputShape[[index]],
         self$kernelSize[i], self$strides[i], self$dilationRate[i], padding = "same" )
       }

     newShape <- NULL
     if( self$data_format == "channels_first" )
       {
       new_shape = list( inputShape[[1]], self$filters,
                         newSpatialDims[1], newSpatialDims[2] )
       } else if( self$data_format == "channels_last" ) {
       new_shape = list( inputShape[[1]],
                         newSpatialDims[1], newSpatialDims[2],
                         self$filters )
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
#' @param kernelSize kernel size
#' @param strides strides
#' @param padding padding
#' @param dataFormat format
#' @param dilationRate dilate rate
#' @param activation activation
#' @param kernelInitializer kernel initializer
#' @param biasInitializer bias initializer
#' @param kernelRegularizer kernel regularizer
#' @param biasRegularizer bias regularizer
#' @param activityRegularizer activity regularizer
#' @param useBias use bias
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @author Tustison NJ
#' @import keras
#' @export
layer_partial_conv_2d <- function( object, filters, kernelSize,
    strides = c( 1L, 1L ), padding = "valid",
    dataFormat = "channels_last", dilationRate = c( 1L, 1L ), activation = NULL,
    kernelInitializer = "glorot_uniform", biasInitializer = "zeros",
    kernelRegularizer = NULL, biasRegularizer = NULL, activityRegularizer = NULL,
    useBias = TRUE, trainable = TRUE )
    {
    create_layer( PartialConv2DLayer, object,
      list( filters = filters, kernelSize = kernelSize, strides = strides, padding = padding,
            dataFormat = dataFormat, dilationRate = dilationRate, activation = activation,
            kernelInitializer = kernelInitializer, biasInitializer = biasInitializer,
            kernelRegularizer = kernelRegularizer, biasRegularizer = biasRegularizer,
            activityRegularizer = activityRegularizer, useBias = useBias, trainable = trainable )
       )
    }

