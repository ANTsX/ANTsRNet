#' Attention layer (2-D)
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfChannels}{number of channels.}
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
#' @return output of tensor shape.
#' @name AttentionLayer2D
NULL

#' @export
AttentionLayer2D <- R6::R6Class( "AttentionLayer2D",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    numberOfChannels = NA,

    doGoogleBrainVersion = TRUE,

    initialize = function( numberOfChannels, doGoogleBrainVersion = TRUE )
      {
      self$numberOfChannels <- as.integer( numberOfChannels )

      self$doGoogleBrainVersion <- doGoogleBrainVersion

      self$numberOfFiltersFG <- as.integer( floor( self$numberOfChannels / 8 ) )
      self$numberOfFiltersH <- as.integer( self$numberOfChannels )
      if( self$doGoogleBrainVersion )
        {
        self$numberOfFiltersH <- as.integer( floor( self$numberOfChannels / 2 ) )
        }
      },

    build = function( inputShape )
      {
      kernelShapeFG <- c( 1L, 1L, self$numberOfChannels, self$numberOfFiltersFG )
      kernelShapeH <- c( 1L, 1L, self$numberOfChannels, self$numberOfFiltersH )

      self$kernelF <- self$add_weight( shape = kernelShapeFG,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelF' )
      self$kernelG <- self$add_weight( shape = kernelShapeFG,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelG' )
      self$kernelH <- self$add_weight( shape = kernelShapeH,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelH' )
      self$biasF <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                     initializer = initializer_zeros(),
                                     name = "biasF" )
      self$biasG <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                     initializer = initializer_zeros(),
                                     name = "biasG" )
      self$biasH <- self$add_weight( shape = c( self$numberOfFiltersH ),
                                     initializer = initializer_zeros(),
                                     name = "biasH" )
      self$gamma <- self$add_weight( shape = c( 1 ),
                                     initializer = initializer_zeros(),
                                     name = "gamma",
                                     trainable = TRUE )

      if( self$doGoogleBrainVersion )
        {
        kernelShapeO <- c( 1L, 1L, as.integer( floor( self$numberOfChannels / 2 ) ), self$numberOfChannels )
        self$kernelO <- self$add_weight( shape = kernelShapeO,
                                         initializer = initializer_glorot_uniform(),
                                         name = 'kernelO' )
        self$biasO <- self$add_weight( shape = c( self$numberOfChannels ),
                                       initializer = initializer_zeros(),
                                       name = "biasO" )
        }
      },

    call = function( input, mask = NULL )
      {
      flatten = function( x )
        {
        K <- keras::backend()
        inputShape <- K$shape( x )
        outputShape <- c( inputShape[1], inputShape[2] * inputShape[3], inputShape[4] )
        xFlat <- K$reshape( x, shape = outputShape )
        return( xFlat )
        }

      K <- keras::backend()
      self$inputShape <- K$shape( input )

      f <- K$conv2d( input, kernel = self$kernelF, strides = c( 1, 1 ), padding = 'same' )
      f <- K$bias_add( f, self$biasF )
      if( self$doGoogleBrainVersion )
        {
        f <- K$relu( f )
        f <- K$pool2d( f, pool_size = tuple( 2, 2 ), strides = tuple( 2, 2 ), padding = 'same' )
        # fshape <- unlist( K$int_shape( f ) )
        # f <- K$pool2d( f, pool_size = tuple( fshape[1], fshape[2] ), strides = tuple( 1, 1 ), padding = 'same' )
        }

      g <- K$conv2d( input, kernel = self$kernelG, strides = c( 1, 1 ), padding = 'same' )
      g <- K$bias_add( g, self$biasG )

      h <- K$conv2d( input, kernel = self$kernelH, strides = c( 1, 1 ), padding = 'same' )
      h <- K$bias_add( h, self$biasH )
      if( self$doGoogleBrainVersion )
        {
        h <- K$relu( h )
        h <- K$pool2d( h, pool_size = tuple( 2, 2 ), strides = tuple( 2, 2 ), padding = 'same' )
        # hshape <- unlist( K$int_shape( h ) )
        # h <- K$pool2d( h, pool_size = tuple( hshape[1], hshape[2] ), strides = tuple( 1, 1 ), padding = 'same' )
        }

      fFlat <- flatten( f )
      gFlat <- flatten( g )
      hFlat <- flatten( h )

      s <- tensorflow::tf$matmul( gFlat, fFlat, transpose_b = TRUE )
      beta <- K$softmax( s, axis = -1L )
      o <- tensorflow::tf$matmul( beta, hFlat )

      if( self$doGoogleBrainVersion )
        {
        outputShape <- K$shape( input )
        outputChannelSize <- as.integer( floor( self$numberOfChannels / 2L ) )
        outputShape <- c( outputShape[1], outputShape[2], outputShape[3], outputChannelSize )

        o <- K$reshape( o, shape = outputShape )
        o <- K$conv2d( o, self$kernelO, strides = c( 1, 1 ), padding = 'same' )
        o <- K$bias_add( o, self$biasO )
        o <- K$relu( o )
        } else {
        o <- K$reshape( o, shape = K$shape( input ) )
        }

      x <- self$gamma * o + input
      return( x )
      },

    compute_output_shape = function( inputShape )
      {
      return( inputShape )
      }
  )
)

#' Attention layer (2-D)
#'
#' Wraps the AttentionLayer2D taken from the following python implementation
#'
#' \url{https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras}
#' \url{https://github.com/taki0112/Self-Attention-GAN-Tensorflow}
#'
#' based on the following paper:
#'
#' \url{https://arxiv.org/abs/1805.08318}
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to
#' or another Layer which this layer will call.
#' @param numberOfChannels numberOfChannels
#' @param doGoogleBrainVersion boolean.  Variant described at second url.
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @export
#' @examples
#'
#' \dontrun{
#' library( keras )
#' library( ANTsRNet )
#'
#' inputShape <- c( 100, 100, 3 )
#' input <- layer_input( shape = inputShape )
#'
#' numberOfFilters <- 64
#' outputs <- input %>% layer_conv_2d( filters = numberOfFilters, kernel_size = 2 )
#' outputs <- outputs %>% layer_attention_2d( numberOfFilters )
#'
#' model <- keras_model( inputs = input, outputs = outputs )
#'}
#' @export
layer_attention_2d <- function( object, numberOfChannels,
  doGoogleBrainVersion = TRUE, trainable = TRUE ) {
create_layer( AttentionLayer2D, object,
    list( numberOfChannels = numberOfChannels,
      doGoogleBrainVersion = doGoogleBrainVersion,
      trainable = trainable )
    )
}

#' Attention layer (3-D)
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfChannels}{number of channels.}
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
#' @return output of tensor shape.
#' @name AttentionLayer3D
NULL

#' @export
AttentionLayer3D <- R6::R6Class( "AttentionLayer3D",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    numberOfChannels = NA,

    doGoogleBrainVersion = TRUE,

    initialize = function( numberOfChannels, doGoogleBrainVersion = TRUE )
      {
      self$numberOfChannels <- as.integer( numberOfChannels )

      self$doGoogleBrainVersion <- doGoogleBrainVersion

      self$numberOfFiltersFG <- as.integer( floor( self$numberOfChannels / 8 ) )
      self$numberOfFiltersH <- as.integer( self$numberOfChannels )
      if( self$doGoogleBrainVersion )
        {
        self$numberOfFiltersH <- as.integer( floor( self$numberOfChannels / 2 ) )
        }
      },

    build = function( inputShape )
      {
      kernelShapeFG <- c( 1L, 1L, 1L, self$numberOfChannels, self$numberOfFiltersFG )
      kernelShapeH <- c( 1L, 1L, 1L, self$numberOfChannels, self$numberOfFiltersH )

      self$kernelF <- self$add_weight( shape = kernelShapeFG,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelF' )
      self$kernelG <- self$add_weight( shape = kernelShapeFG,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelG' )
      self$kernelH <- self$add_weight( shape = kernelShapeH,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelH' )
      self$biasF <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                     initializer = initializer_zeros(),
                                     name = "biasF" )
      self$biasG <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                     initializer = initializer_zeros(),
                                     name = "biasG" )
      self$biasH <- self$add_weight( shape = c( self$numberOfFiltersH ),
                                     initializer = initializer_zeros(),
                                     name = "biasH" )
      self$gamma <- self$add_weight( shape = c( 1 ),
                                     initializer = initializer_zeros(),
                                     name = "gamma",
                                     trainable = TRUE )

      if( self$doGoogleBrainVersion )
        {
        kernelShapeO <- c( 1L, 1L, 1L, as.integer( floor( self$numberOfChannels / 2 ) ), self$numberOfChannels )
        self$kernelO <- self$add_weight( shape = kernelShapeO,
                                         initializer = initializer_glorot_uniform(),
                                         name = 'kernelO' )
        self$biasO <- self$add_weight( shape = c( self$numberOfChannels ),
                                       initializer = initializer_zeros(),
                                       name = "biasO" )
        }
      },

    call = function( input, mask = NULL )
      {
      flatten = function( x )
        {
        K <- keras::backend()
        inputShape <- K$shape( x )
        outputShape <- c( inputShape[1], inputShape[2] * inputShape[3] * inputShape[4], inputShape[5] )
        xFlat <- K$reshape( x, shape = outputShape )
        return( xFlat )
        }

      K <- keras::backend()
      self$inputShape <- K$shape( input )

      f <- K$conv3d( input, kernel = self$kernelF, strides = c( 1, 1, 1 ), padding = 'same' )
      f <- K$bias_add( f, self$biasF )
      if( self$doGoogleBrainVersion )
        {
        f <- K$relu( f )
        f <- K$pool3d( f, pool_size = tuple( 2, 2, 2 ), strides = tuple( 2, 2, 2 ), padding = 'same' )
        # fshape <- unlist( K$int_shape( f ) )
        # f <- K$pool3d( f, pool_size = tuple( fshape[1], fshape[2], fshape[3] ), strides = tuple( 1, 1, 1 ), padding = 'same' )
        }

      g <- K$conv3d( input, kernel = self$kernelG, strides = c( 1, 1, 1 ), padding = 'same' )
      g <- K$bias_add( g, self$biasG )

      h <- K$conv3d( input, kernel = self$kernelH, strides = c( 1, 1, 1 ), padding = 'same' )
      h <- K$bias_add( h, self$biasH )
      if( self$doGoogleBrainVersion )
        {
        h <- K$relu( h )
        h <- K$pool3d( h, pool_size = tuple( 2, 2, 2 ), strides = tuple( 2, 2, 2 ), padding = 'same' )
        # hshape <- unlist( K$int_shape( h ) )
        # h <- K$pool3d( h, pool_size = tuple( hshape[1], hshape[2], hshape[3] ), strides = tuple( 1, 1, 1 ), padding = 'same' )
        }

      fFlat <- flatten( f )
      gFlat <- flatten( g )
      hFlat <- flatten( h )

      s <- tensorflow::tf$matmul( gFlat, fFlat, transpose_b = TRUE )
      beta <- K$softmax( s, axis = -1L )
      o <- tensorflow::tf$matmul( beta, hFlat )

      if( self$doGoogleBrainVersion )
        {
        outputShape <- K$shape( input )
        outputChannelSize <- as.integer( floor( self$numberOfChannels / 2L ) )
        outputShape <- c( outputShape[1], outputShape[2], outputShape[3], outputShape[3], outputChannelSize )

        o <- K$reshape( o, shape = outputShape )
        o <- K$conv3d( o, self$kernelO, strides = c( 1, 1, 1 ), padding = 'same' )
        o <- K$bias_add( o, self$biasO )
        o <- K$relu( o )
        } else {
        o <- K$reshape( o, shape = K$shape( input ) )
        }

      x <- self$gamma * o + input
      return( x )
      },

    compute_output_shape = function( inputShape )
      {
      return( inputShape )
      }
  )
)

#' Attention layer (3-D)
#'
#' Wraps the AttentionLayer3D taken from the following python implementation
#'
#' \url{https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras}
#' \url{https://github.com/taki0112/Self-Attention-GAN-Tensorflow}
#'
#' based on the following paper:
#'
#' \url{https://arxiv.org/abs/1805.08318}
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to
#' or another Layer which this layer will call.
#' @param numberOfChannels numberOfChannels
#' @param doGoogleBrainVersion boolean.  Variant described at second url.
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @export
#' @examples
#' \dontrun{
#' library( keras )
#' library( ANTsRNet )
#'
#' inputShape <- c( 100, 100, 100, 3 )
#' input <- layer_input( shape = inputShape )
#'
#' numberOfFilters <- 64
#' outputs <- input %>% layer_conv_3d( filters = numberOfFilters, kernel_size = 2 )
#' outputs <- outputs %>% layer_attention_3d( numberOfFilters )
#'
#' model <- keras_model( inputs = input, outputs = outputs )
#'}
layer_attention_3d <- function( object, numberOfChannels,
  doGoogleBrainVersion = TRUE, trainable = TRUE ) {
create_layer( AttentionLayer3D, object,
    list( numberOfChannels = numberOfChannels,
      doGoogleBrainVersion = doGoogleBrainVersion,
      trainable = trainable )
    )
}

#' Efficient attention layer (2-D)
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfFiltersFG}{number of filters for F and G layers.}
#'  \item{numberOfFiltersH}{number of filters for H. If = NA, only
#'        use filter F for efficiency.}
#'  \item{poolSize}{pool_size in max pool layer.}
#'  \item{doUpsample}{upsample final layers before concatenation or adding.}
#'  \item{doConcatenateFinalLayers}{concatenate final layer with input.
#'        Alternatively, add.}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author BB Avants, NJ Tustison
#'
#' @return output of tensor shape.
#' @name EfficientAttentionLayer2D
NULL

#' @export
EfficientAttentionLayer2D <- R6::R6Class( "EfficientAttentionLayer2D",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    numberOfFiltersFG = 4L,

    numberOfFiltersH = 8L,

    poolSize = 2L,

    doUpsample = TRUE,

    doConcatenateFinalLayers = FALSE,

    initialize = function( numberOfFiltersFG = 4L, numberOfFiltersH = 8L, poolSize = 2L,
      doUpsample = TRUE, doConcatenateFinalLayers = FALSE )
      {
      self$numberOfFiltersFG <- as.integer( numberOfFiltersFG )
      self$numberOfFiltersH <- as.integer( numberOfFiltersH )
      self$poolSize <- as.integer( poolSize )
      self$doUpsample <- doUpsample
      self$doConcatenateFinalLayers <- doConcatenateFinalLayers
      },

    build = function( inputShape )
      {
      kernelShapeFG <- c( 1L, 1L, inputShape[4], self$numberOfFiltersFG )
      kernelShapeH <- c( 1L, 1L, inputShape[4], self$numberOfFiltersH )

      self$kernelF <- self$add_weight( shape = kernelShapeFG,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelF' )
      self$biasF <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                     initializer = initializer_zeros(),
                                     name = "biasF" )

      if( ! is.na( self$numberOfFiltersH ) )
        {
        self$kernelG <- self$add_weight( shape = kernelShapeFG,
                                         initializer = initializer_glorot_uniform(),
                                         name = 'kernelG' )
        self$biasG <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                       initializer = initializer_zeros(),
                                       name = "biasG" )
        self$kernelH <- self$add_weight( shape = kernelShapeH,
                                         initializer = initializer_glorot_uniform(),
                                         name = 'kernelH' )
        self$biasH <- self$add_weight( shape = c( self$numberOfFiltersH ),
                                       initializer = initializer_zeros(),
                                       name = "biasH" )
        }

      kernelShapeO <- c( 1L, 1L, inputShape[4], inputShape[4] )
      self$kernelO <- self$add_weight( shape = kernelShapeO,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelO' )
      self$biasO <- self$add_weight( shape = c( inputShape[4] ),
                                     initializer = initializer_zeros(),
                                     name = "biasO" )
      },

    call = function( input, mask = NULL )
      {
      flatten = function( x )
        {
        K <- keras::backend()
        inputShape <- K$shape( x )
        outputShape <- c( inputShape[1], inputShape[2] * inputShape[3], inputShape[4] )
        xFlat <- K$reshape( x, shape = outputShape )
        return( xFlat )
        }

      K <- keras::backend()
      self$inputShape <- K$shape( input )

      f <- K$conv2d( input, kernel = self$kernelF, strides = c( 1, 1 ), padding = 'same' )
      f <- K$bias_add( f, self$biasF )
      f <- K$relu( f )
      f <- K$pool2d( f, pool_size = tuple( self$poolSize, self$poolSize ), padding = 'same' )
      fFlat <- flatten( f )

      if( ! is.na( self$numberOfFiltersH ) )
        {
        g <- K$conv2d( input, kernel = self$kernelG, strides = c( 1, 1 ), padding = 'same' )
        g <- K$bias_add( g, self$biasG )
        g <- K$relu( g )
        g <- K$pool2d( g, pool_size = tuple( self$poolSize, self$poolSize ), padding = 'same' )
        gFlat <- flatten( g )

        h <- K$conv2d( input, kernel = self$kernelH, strides = c( 1, 1 ), padding = 'same' )
        h <- K$bias_add( h, self$biasH )
        h <- K$relu( h )
        h <- K$pool2d( h, pool_size = tuple( self$poolSize, self$poolSize ), padding = 'same' )
        hFlat <- flatten( h )

        s <- tensorflow::tf$matmul( gFlat, fFlat, transpose_b = TRUE )
        beta <- K$softmax( s, axis = -1L )
        o <- tensorflow::tf$matmul( beta, hFlat )
        } else {
        s <- tensorflow::tf$matmul( fFlat, fFlat, transpose_b = TRUE )
        beta <- K$softmax( s, axis = -1L )
        o <- tensorflow::tf$matmul( beta, fFlat )
        }

      if( self$doUpsample == TRUE )
        {
        outputShape <- list( self$inputShape[1],
          as.integer( floor( input$shape[1] / self$poolSize ) ),
          as.integer( floor( input$shape[2] / self$poolSize ) ),
          self$inputShape[4] )

        o <- K$reshape( o, shape = K$stack( outputShape ) )

        o <- K$resize_images( o, height_factor = self$poolSize,
          width_factor = self$poolSize, data_format = "channels_last" )
        o <- K$conv2d( o, self$kernelO, strides = c( 1, 1 ), padding = 'same' )
        o <- K$bias_add( o, self$biasO )
        o <- K$relu( o )
        }

     if( self$doConcatenateFinalLayers == TRUE )
       {
       x <- K$concatenate( list( input, o ) )
       } else {
       x <- input * 0.5 + o * 0.5
       }
      return( x )
      },

    compute_output_shape = function( inputShape )
      {
      return( inputShape )
      }
  )
)

#' Efficient attention layer (2-D)
#'
#' Wraps the EfficientAttentionLayer2D modified from the following python implementation
#'
#' \url{https://github.com/taki0112/Self-Attention-GAN-Tensorflow}
#'
#' based on the following paper:
#'
#' \url{https://arxiv.org/abs/1805.08318}
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to
#' or another Layer which this layer will call.
#' @param numberOfFiltersFG number of filters for F and G layers.
#' @param numberOfFiltersH number of filters for H. If \code{= NA}, only
#' use filter \code{F} for efficiency.
#' @param poolSize pool size in max pool layer.  Default = 2L.
#' @param doUpsample upsample final layers before concatenating or
#' adding.  Default = TRUE.
#' @param doConcatenateFinalLayers concatenate final layer with input.
#' Alternatively, add.  Default = FALSE
#' @return a keras layer tensor
#' @export
#' @examples
#'
#' \dontrun{
#' library( keras )
#' library( ANTsRNet )
#'
#' inputShape <- c( 100, 100, 3 )
#' input <- layer_input( shape = inputShape )
#'
#' numberOfFiltersFG <- 64L
#' outputs <- input %>% layer_efficient_attention_2d( numberOfFiltersFG )
#'
#' model <- keras_model( inputs = input, outputs = outputs )
#'}
#'
#' @export
layer_efficient_attention_2d <- function( object, numberOfFiltersFG = 4L,
  numberOfFiltersH = 8L, poolSize = 2L, doUpsample = TRUE,
  doConcatenateFinalLayers = FALSE, trainable = TRUE ) {
create_layer( EfficientAttentionLayer2D, object,
    list( numberOfFiltersFG = numberOfFiltersFG, numberOfFiltersH = numberOfFiltersH,
          poolSize = poolSize, doUpsample = doUpsample,
          doConcatenateFinalLayers = doConcatenateFinalLayers,
          trainable = trainable )
    )
}

#' Efficient attention layer (3-D)
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{numberOfFiltersFG}{number of filters for F and G layers.}
#'  \item{numberOfFiltersH}{number of filters for H. If = NA, only
#'        use filter F for efficiency.}
#'  \item{poolSize}{pool_size in max pool layer.}
#'  \item{doUpsample}{upsample final layers before concatenation or adding.}
#'  \item{doConcatenateFinalLayers}{concatenate final layer with input.
#'        Alternatively, add.}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author BB Avants, NJ Tustison
#'
#' @return output of tensor shape.
#' @name EfficientAttentionLayer3D
NULL

#' @export
EfficientAttentionLayer3D <- R6::R6Class( "EfficientAttentionLayer3D",

  inherit = KerasLayer,

  lock_objects = FALSE,

  public = list(

    numberOfFiltersFG = 4L,

    numberOfFiltersH = 8L,

    poolSize = 2L,

    doUpsample = TRUE,

    doConcatenateFinalLayers = FALSE,

    initialize = function( numberOfFiltersFG = 4L, numberOfFiltersH = 8L, poolSize = 2L,
      doUpsample = TRUE, doConcatenateFinalLayers = FALSE )
      {
      self$numberOfFiltersFG <- as.integer( numberOfFiltersFG )
      self$numberOfFiltersH <- as.integer( numberOfFiltersH )
      self$poolSize <- as.integer( poolSize )
      self$doUpsample <- doUpsample
      self$doConcatenateFinalLayers <- doConcatenateFinalLayers
      },

    build = function( inputShape )
      {
      kernelShapeFG <- c( 1L, 1L, 1L, inputShape[5], self$numberOfFiltersFG )
      kernelShapeH <- c( 1L, 1L, 1L, inputShape[5], self$numberOfFiltersH )

      self$kernelF <- self$add_weight( shape = kernelShapeFG,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelF' )
      self$biasF <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                     initializer = initializer_zeros(),
                                     name = "biasF" )

      if( ! is.na( self$numberOfFiltersH ) )
        {
        self$kernelG <- self$add_weight( shape = kernelShapeFG,
                                         initializer = initializer_glorot_uniform(),
                                         name = 'kernelG' )
        self$biasG <- self$add_weight( shape = c( self$numberOfFiltersFG ),
                                       initializer = initializer_zeros(),
                                       name = "biasG" )
        self$kernelH <- self$add_weight( shape = kernelShapeH,
                                         initializer = initializer_glorot_uniform(),
                                         name = 'kernelH' )
        self$biasH <- self$add_weight( shape = c( self$numberOfFiltersH ),
                                       initializer = initializer_zeros(),
                                       name = "biasH" )
        }

      kernelShapeO <- c( 1L, 1L, 1L, inputShape[5], inputShape[5] )
      self$kernelO <- self$add_weight( shape = kernelShapeO,
                                       initializer = initializer_glorot_uniform(),
                                       name = 'kernelO' )
      self$biasO <- self$add_weight( shape = c( inputShape[5] ),
                                     initializer = initializer_zeros(),
                                     name = "biasO" )
      },

    call = function( input, mask = NULL )
      {
      flatten = function( x )
        {
        K <- keras::backend()
        inputShape <- K$shape( x )
        outputShape <- c( inputShape[1], inputShape[2] * inputShape[3] * inputShape[4], inputShape[5] )
        xFlat <- K$reshape( x, shape = outputShape )
        return( xFlat )
        }

      K <- keras::backend()
      self$inputShape <- K$shape( input )

      f <- K$conv3d( input, kernel = self$kernelF, strides = c( 1, 1, 1 ), padding = 'same' )
      f <- K$bias_add( f, self$biasF )
      f <- K$relu( f )
      f <- K$pool3d( f, pool_size = tuple( self$poolSize, self$poolSize, self$poolSize ), padding = 'same' )
      fFlat <- flatten( f )

      if( ! is.na( self$numberOfFiltersH ) )
        {
        g <- K$conv3d( input, kernel = self$kernelG, strides = c( 1, 1, 1 ), padding = 'same' )
        g <- K$bias_add( g, self$biasG )
        g <- K$relu( g )
        g <- K$pool3d( g, pool_size = tuple( self$poolSize, self$poolSize, self$poolSize ), padding = 'same' )
        gFlat <- flatten( g )

        h <- K$conv3d( input, kernel = self$kernelH, strides = c( 1, 1, 1 ), padding = 'same' )
        h <- K$bias_add( h, self$biasH )
        h <- K$relu( h )
        h <- K$pool3d( h, pool_size = tuple( self$poolSize, self$poolSize, self$poolSize ), padding = 'same' )
        hFlat <- flatten( h )

        s <- tensorflow::tf$matmul( gFlat, fFlat, transpose_b = TRUE )
        beta <- K$softmax( s, axis = -1L )
        o <- tensorflow::tf$matmul( beta, hFlat )
        } else {
        s <- tensorflow::tf$matmul( fFlat, fFlat, transpose_b = TRUE )
        beta <- K$softmax( s, axis = -1L )
        o <- tensorflow::tf$matmul( beta, fFlat )
        }

      if( self$doUpsample == TRUE )
        {
        outputShape <- list( self$inputShape[1],
          as.integer( floor( input$shape[1] / self$poolSize ) ),
          as.integer( floor( input$shape[2] / self$poolSize ) ),
          as.integer( floor( input$shape[3] / self$poolSize ) ),
          self$inputShape[5] )

        o <- K$reshape( o, shape = K$stack( outputShape ) )

        o <- K$resize_volumes( o, depth_factor = self$poolSize,
          height_factor = self$poolSize, width_factor = self$poolSize,
          data_format = "channels_last" )
        o <- K$conv3d( o, self$kernelO, strides = c( 1, 1, 1 ), padding = 'same' )
        o <- K$bias_add( o, self$biasO )
        o <- K$relu( o )
        }

     if( self$doConcatenateFinalLayers == TRUE )
       {
       x <- K$concatenate( list( input, o ) )
       } else {
       x <- input * 0.5 + o * 0.5
       }
      return( x )
      },

    compute_output_shape = function( inputShape )
      {
      return( inputShape )
      }
  )
)

#' Efficient attention layer (3-D)
#'
#' Wraps the EfficientAttentionLayer3D modified from the following python implementation
#'
#' \url{https://github.com/taki0112/Self-Attention-GAN-Tensorflow}
#'
#' based on the following paper:
#'
#' \url{https://arxiv.org/abs/1805.08318}
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to
#' or another Layer which this layer will call.
#' @param numberOfFiltersFG number of filters for F and G layers.
#' @param numberOfFiltersH number of filters for H. If \code{= NA}, only
#' use filter \code{F} for efficiency.
#' @param poolSize pool size in max pool layer.  Default = 2L.
#' @param doUpsample upsample final layers before concatenating or
#' adding.  Default = TRUE.
#' @param doConcatenateFinalLayers concatenate final layer with input.
#' Alternatively, add.  Default = FALSE
#' @return a keras layer tensor
#' @export
#' @examples
#'
#' \dontrun{
#' library( keras )
#' library( ANTsRNet )
#'
#' inputShape <- c( 100, 100, 100, 3 )
#' input <- layer_input( shape = inputShape )
#'
#' numberOfFiltersFG <- 64L
#' outputs <- input %>% layer_efficient_attention_3d( numberOfFiltersFG )
#'
#' model <- keras_model( inputs = input, outputs = outputs )
#'}
#'
#' @export
layer_efficient_attention_3d <- function( object, numberOfFiltersFG = 4L,
  numberOfFiltersH = 8L, poolSize = 2L, doUpsample = TRUE,
  doConcatenateFinalLayers = FALSE, trainable = TRUE ) {
create_layer( EfficientAttentionLayer3D, object,
    list( numberOfFiltersFG = numberOfFiltersFG, numberOfFiltersH = numberOfFiltersH,
          poolSize = poolSize, doUpsample = doUpsample,
          doConcatenateFinalLayers = doConcatenateFinalLayers,
          trainable = trainable )
    )
}

#' Attention augmentation layer (2-D)
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{depthOfQueries}{number of filters for queries.}
#'  \item{depthOfValues}{number of filters for values.}
#'  \item{numberOfHeads}{number of attention heads to use. It is required
#'                       that depthOfQueries/numberOfHeads > 0.}
#'  \item{isRelative}{whether or not to use relative encodings.}
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
#' @return output of tensor shape [batchSize, height, width, depthOfQueries].
#' @name AttentionAugmentationLayer2D
NULL

#' @export
AttentionAugmentationLayer2D <- R6::R6Class(
  "AttentionAugmentationLayer2D",

  inherit = KerasLayer,

  public = list(

    depthOfQueries = NA,

    depthOfValues = NA,

    numberOfHeads = NA,

    isRelative = TRUE,

    initialize = function( depthOfQueries, depthOfValues, numberOfHeads )
    {
      if( depthOfQueries %% numberOfHeads != 0 )
        {
        stop( "Error:  depthOfQueries must be divisible by numberOfHeads." )
        }
      if( depthOfValues %% numberOfHeads != 0 )
        {
        stop( "Error:  depthOfValues must be divisible by numberOfHeads." )
        }

      if( floor( depthOfQueries / numberOfHeads ) < 1.0 )
        {
        stop( "Error:  depthOfQueries / numberOfHeads must be > 1." )
        }
      if( floor( depthOfValues / numberOfHeads ) < 1.0 )
        {
        stop( "Error:  depthOfValues / numberOfHeads must be > 1." )
        }

      self$depthOfQueries <- as.integer( depthOfQueries )
      self$depthOfValues <- as.integer( depthOfValues )
      self$numberOfHeads <- as.integer( numberOfHeads )
      self$isRelative <- isRelative

      K <- keras::backend()

      self$channelAxis <- 1L
      if( keras::backend()$image_data_format() == "channels_last" )
        {
        self$channelAxis <- -1L
        }
    },

    build = function( inputShape )
    {
      self$inputShape <- inputShape

      numberOfChannels <- self$inputShape[2]
      height <- self$inputShape[3]
      width <- self$inputShape[4]
      if( self$channelAxis == -1L )
        {
        height <- self$inputShape[2]
        width <- self$inputShape[[3]]
        numberOfChannels <- self$inputShape[4]
        }

      self$keyRelativeWidth <- NULL
      self$keyRelativeHeight <- NULL
      if( self$isRelative )
        {
        depthOfQueriesPerHead <- floor( self$depthOfQueries / self$numberOfHeads )

        self$keyRelativeWidth <- self$add_weight(
                                   name = "key_relative_width",
                                   shape = shape( 2 * width - 1, depthOfQueriesPerHead ),
                                   initializer = initializer_random_normal(
                                     stddev = depthOfQueriesPerHead ^ -0.5 ) )
        self$keyRelativeHeight <- self$add_weight(
                                   name = "key_relative_height",
                                   shape = shape( 2 * height - 1, depthOfQueriesPerHead ),
                                   initializer = initializer_random_normal(
                                     stddev = depthOfQueriesPerHead ^ -0.5 ) )
        }
    },

    call = function( inputs, mask = NULL )
    {
      K <- keras::backend()

      if( self$channelAxis == 1 )
        {
        inputs <- K$permute_dimensions( inputs, c( 0L, 2L, 3L, 1L ) )
        }

      splitTensors <- tensorflow::tf$split( inputs, c( self$depthOfQueries,
        self$depthOfQueries, self$depthOfValues, axis = -1L ) )

      qTensor <- self$splitHeads2D( splitTensors[[1]] )
      kTensor <- self$splitHeads2D( splitTensors[[2]] )
      vTensor <- self$splitHeads2D( splitTensors[[3]] )

      depthOfQueriesHeads <- self$depthOfQueries / self$numberOfHeads
      qTensor <- qTensor * ( depthOfQueriesHeads ^ -0.5 )

      qkShape <- rep( NA, 4 )
      qkShape[1] <- self$batchSize
      qkShape[2] <- self$numberOfHeads
      qkShape[3] <- as.integer( self$height * self$width )
      qkShape[4] <- as.integer( floor( self$depthOfQueries / self$numberOfHeads ) )

      vShape <- rep( NA, 4 )
      vShape[1] <- self$batchSize
      vShape[2] <- self$numberOfHeads
      vShape[3] <- as.integer( self$height * self$width )
      vShape[4] <- as.integer( floor( self$depthOfValues / self$numberOfHeads ) )

      qFlat <- K$reshape( q, K$stack( qkShape ) )
      kFlat <- K$reshape( k, K$stack( qkShape ) )
      vFlat <- K$reshape( v, K$stack( vShape ) )

      logits <- tensorflow::tf$matmul( qFlat, kFlat, transpose_b = TRUE )

      # Apply relative encodings
      if( self$isRelative == TRUE )
        {
        hwLogits <- self$relativeLogits( qTensor )
        logits <- logits + hwLogits[[1]]
        logits <- logits + hwLogits[[2]]
        }

      weights <- K$softmax( logits, axis = -1L )
      attentionOut <- tensorflow::tf$matmul( weights, vFlat )

      attentionOutShape <- rep( NA, 5 )
      attentionOutShape[1] <- self$batchSize
      attentionOutShape[2] <- self$numberOfHeads
      attentionOutShape[3] <- self$height
      attentionOutShape[4] <- self$width
      attentionOutShape[5] <- as.integer( floor( self$depthOfQueries / self$numberOfHeads ) )

      attentionOutShape <- K$stack( attentionOutShape )
      attentionOut <- K$reshape( attentionOut, attentionOutShape )
      attentionOut <- self$combineHeads2D( attentionOut )

      if( self$axis == 1 )
        {
        attentionOut <- K$permute_dimensions( attentionOut, c( 0L, 3L, 1L, 2L ) )
        }

      attentionOut$set_shape( self$compute_output_shape( self$inputShape ) )

      return( attentionOut )
    },

    compute_output_shape = function( input_shape )
    {
      numberOfChannels <- as.integer( tail( unlist( input_shape[[1]] ), 1 ) )

      return( list( NULL, as.integer( self$resampledSize[1] ),
                    as.integer( self$resampledSize[2] ), as.integer( self$resampledSize[3] ),
                    numberOfChannels ) )
    },

    splitHeads2D = function( input )
    {
      tensorShape <- K$shape( input )

      batchSize <- tensorShape[[0]]
      height <- tensorShape[[1]]
      width <- tensorShape[[2]]
      numberOfChannels <- tensorShape[[3]]

      self$batchSize <- batchSize
      self$height <- height
      self$width <- width

      returnShape <- K$stack( c( batchSize, height, width, self$numberOfHeads,
        floor( numberOfChannels / self$numberOfHeads ) ) )
      split <- K$reshape( input, returnShape )
      transposeAxes <- c( 0, 3, 1, 2, 4 )
      split <- K$premute_dimensions( split, transposeAxes )

      return( split )
    },

    relativeLogits = function( q )
    {
      K <- keras::backend()

      shape <- K$shape( q )

      height <- shape[2]
      width <- shape[3]

      relativeLogits <- list()
      relativeLogits[[1]] <- self$relativeLogits1D( q, self$key_relative_width,
        height, width, transposeMask = c( 0, 1, 2, 4, 3, 5 ) )
      qPermuted <- K$permute_dimensions( q, c( 0, 1, 3, 2, 4 ) )
      relativeLogits[[2]] <- self$relativeLogits1D( qPermuted, self$key_relative_height,
        width, height, transposeMask = c( 0, 1, 4, 2, 5, 3 ) )

      return( relativeLogits )
    },

    relativeLogits1D = function( q, relative, height, width, transposeMask )
    {
      K <- keras::backend()

      relativeLogits <- tensorflow::tf$einsum( 'bhxyd,md->bhxym', q, relative )
      relativeLogits <- K$reshape( relativeLogits, c( -1L,
        as.integer( self$numberOfHeads * height ), width, as.integer( 2 * width - 1 ) ) )
      relativeLogits <- self$relativeToAbsolute( relativeLogits )
      relativeLogits <- K$reshape( relativeLogits, c( -1L, self$numberOfHeads,
        height, width, width ) )
      relativeLogits <- K$tile( relativeLogits, c( 1L, 1L, 1L, H, 1L, 1L ) )
      relatievLogits <- K$permute_dimensions( relativeLogits, transposeMask )
      relativeLogits <- K$reshape( relativeLogits, c( -1L, self$numberOfHeads,
        height * width, height * width ) )

      return( relativeLogits )
    },

    relativeToAbsolute = function( x )
    {
      K <- keras::backend()

      shape <- K$shape( x )
      B <- shape[0]
      Nh <- shape[1]
      L <- shape[2]

      columnPad <- K$zeros( K$stack( list( B, Nh, L, 1L ) ) )
      x <- K$concatenate( list( x, columnPad ), axis = 3 )
      xFlat <- K$reshape( K$reshape( x, c( B, Nh, L + 1, 2 * L - 1 ) ) )
      padFlat <- K$zeros( K$stack( c( B, Nh, L - 1 ) ) )
      xFlatPadded <- K$concatenate( list( xFlat, padFlat ) )
      xFinal <- K$reshape( xFlatPadded, c( B, Nh, L + 1, 2 * L - 1 ) )
      xFinal <- xFinal[,,1:( L - 1 ),(L - 1):( 2 * L - 1 )]

      return( xFinal )
    },

    combineHeads2D = function( inputs )
    {
      K <- keras::backend()

      transposed <- K$permute_dimensions( inputs, c( 0L, 2L, 3L, 1L, 4L ) )
      shape <- K$shape( transposed )
      a <- shape[3]
      b <- shape[4]

      returnShape <- K$stack( list( shape[3:4], a * b ) )

      return( K$reshape( transposed, returnShape ) )
    }
  )
)

#' Attention augmentation layer (2-D)
#'
#' Wraps the AttentionAugmentation2D layer.
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param depthOfQueries number of filters for queries.
#' @param depthOfValues number of filters for values.
#' @param numberOfHeads number of attention heads to use. It is required
#'   that \code{depthOfQueries/numberOfHeads > 0}.
#' @param isRelative whether or not to use relative encodings.
#' @param trainable Whether the layer weights will be updated during training.
#' @return a keras layer tensor
#' @export
#' @examples
#' \dontrun{
#'  }
layer_attention_augmentation_2d <- function( object,
  depthOfQueries, depthOfValues, numberOfHeads, isRelative,
  trainable = TRUE ) {
create_layer( AttentionAugmentationLayer2D, object,
    list( depthOfQueries = depthOfQueries, depthOfValues = depthOfValues,
      numberOfHeads = numberOfHeads, isRelative = isRelative,
      trainable = trainable )
    )
}

#' Creates a 2-D attention augmented convolutional block
#'
#' Creates a 2-D attention augmented convolutional layer as described in the paper
#'
#'   \url{https://arxiv.org/abs/1904.09925}
#'
#' with the implementation ported from the following repository
#'
#'   \url{https://github.com/titu1994/keras-attention-augmented-convs}
#'
#' @param inputLayer input keras layer.
#' @param numberOfOutputFilters number of output filters.
#' @param kernelSize convolution kernel size.
#' @param strides convolution strides.
#' @param depthOfQueries Defines the number of filters for the queries or \code{k}.
#'   Either absolute or, if \code{< 1.0}, number of \code{k} filters =
#'   \code{depthOfQueries * numberOfOutputFilters}.
#' @param depthOfValues Defines the number of filters for the values or \code{v}.
#'   Either absolute or, if \code{< 1.0}, number of \code{v} filters =
#'   \code{depthOfValues * numberOfOutputFilters}.
#' @param numberOfAttentionHeads number of attention heads.  Note that
#' \code{as.integer(kDepth/numberOfAttentionHeads)>0} (default = 8).
#' @param useRelativeEncodings boolean for whether to use relative encodings
#' (default = TRUE).
#'
#' @return a keras tensor
#' @author Tustison NJ
#' @export
layer_attention_augmented_convolution_block_2d <- function( inputLayer,
                                     numberOfOutputFilters,
                                     kernelSize = c( 3, 3 ),
                                     strides = c( 1, 1 ),
                                     depthOfQueries = 0.2,
                                     depthOfValues = 0.2,
                                     numberOfAttentionHeads = 8,
                                     useRelativeEncodings = TRUE )
{
  stop( "Not finished yet." )
  channelAxis <- 2L
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1L
    }

  if( depthOfQueries < 1.0 )
    {
    depthOfQueries <- as.integer( depthOfQueries * numberOfOutputFilters )
    } else {
    depthOfQueries <- as.integer( depthOfQueries )
    }
  if( depthOfValues < 1.0 )
    {
    depthOfValues <- as.integer( depthOfValues * numberOfOutputFilters )
    } else {
    depthOfValues <- as.integer( depthOfValues )
    }

  localNumberOfFilters <- numberOfOutputFilters - depthOfQueries
  convolutionLayer <- inputLayer %>% layer_conv_2d( localNumberOfFilters,
    kernel_size = kernelSize, strides = strides, padding = 'same',
    use_bias = TRUE, kernel_initializer = 'he_normal' )

  # Augmented attention block

  localNumberOfFilters <- 2 * depthOfQueries + depthOfValues
  qkvConvolutionLayer <- inputLayer %>% layer_conv_2d( localNumberOfFilters,
    kernel_size = c( 1, 1 ), strides = strides, padding = 'same',
    use_bias = TRUE, kernel_initializer = 'he_normal' )

  attentionOutLayer <- qkvConvolutionLayer %>%
    layer_attention_augmentation_2d( depthOfQueries,
    depthOfValues, numberOfAttentionHeads, useRelativeEncodings )
  attentionOutLayer <- attentionOutLayer %>% layer_conv_2d( depthOfValues,
    kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same',
    use_bias = TRUE, kernel_initializer = 'he_normal' )

  output <- layer_concatenate( list( convolutionLayer, attentionOutLayer ),
    axis = channelAxis )
  output <- output %>% layer_batch_normalization()

  return( output )
}
