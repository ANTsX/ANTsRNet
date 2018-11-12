#' Bilinear interpolation layer (2-D)
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{outputs <- bilinearInterpolation( inputs, resampledSize )}
#'
#' @section Arguments:
#' \describe{
#'  \item{inputs}{list of size 2 where the first element are the images. The second
#'                element are the weights.}
#'  \item{resampledSize}{size of the resampled output images.}
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
#' @return resampled batch images.
#'
#' @name BilinearInterpolationLayer2D
NULL

#' @export
BilinearInterpolationLayer2D <- R6::R6Class( "BilinearInterpolationLayer2D",

  inherit = KerasLayer,

  public = list(

    resampledSize = NULL,

    initialize = function( resampledSize )
      {
      K <- keras::backend()
      if( K$backend() != 'tensorflow' )
        {
        stop( "Error:  tensorflow is required for this STN implementations." )
        }

      self$resampledSize <- resampledSize
      },

    call = function( inputs, mask = NULL )
      {
      image <- inputs[[1]]
      transformParameters <- inputs[[2]]

      output <- self$affineTransformImage( image, transformParameters, self$resampledSize )

      return( output )
      },

    compute_output_shape = function( input_shape )
      {
      numberOfChannels <- input_shape[[4]]
      return( reticulate::tuple( input_shape[[1]], resampledSize[1],
          resampledSize[2], numberOfChannels ) )
      },

    interpolate = function( image, sampledGrids, resampledSize )
      {
      batchSize <- K$shape( image )[1]
      height <- K$shape( image )[2]
      width <- K$shape( image )[3]
      numberOfChannels <- K$shape( image )[4]

      x <- K$cast( K$flatten( sampledGrids[, 1:2,] ), dtype = 'float32' )
      y <- K$cast( K$flatten( sampledGrids[, 3:4,] ), dtype = 'float32' )
      x <- 0.5 * ( x + 1.0 ) * K$cast( width, dtype = 'float32' )
      y <- 0.5 * ( y + 1.0 ) * K$cast( height, dtype = 'float32' )

      x0 <- K$cast( x, dtype = 'int32' )
      x1 <- x0 + 1L
      y0 <- K$cast( y, dtype = 'int32' )
      y1 <- y0 + 1L

      xMax <- as.integer( ( K$int_shape( image )[2] - 1 ) )
      yMax <- as.integer( ( K$int_shape( image )[1] - 1 ) )

      x0 <- K$clip( x0, 0, xMax )
      x1 <- K$clip( x1, 0, xMax )
      y0 <- K$clip( y0, 0, yMax )
      y1 <- K$clip( y1, 0, yMax )

      batchPixels <- K$arange( 0, batchSize ) * ( height * width )
      batchPixels <- K$expand_dims( batchPixels, axis = -1 )
      base <- K$repeat_elements(
        batchPixels, rep = as.integer( prod( resampledSize ) ), axis = 1 )
      base <- K$flatten( base )

      baseY0 <- y0 * width
      baseY0 <- base + baseY0
      baseY1 <- y1 * width
      baseY1 <- baseY1 + base

      indicesA <- baseY0 + x0
      indicesB <- baseY1 + x0
      indicesC <- baseY0 + x1
      indicesD <- baseY1 + x1

      flatImage <- K$reshape( image, shape = c( -1L, numberOfChannels ) )
      flatImage <- K$cast( flatImage, dtype = 'float32' )
      pixelValuesA <- K$gather( flatImage, indicesA )
      pixelValuesB <- K$gather( flatImage, indicesB )
      pixelValuesC <- K$gather( flatImage, indicesC )
      pixelValuesD <- K$gather( flatImage, indicesD )

      x0 <- K$cast( x0, dtype = 'float32' )
      x1 <- K$cast( x1, dtype = 'float32' )
      y0 <- K$cast( y0, dtype = 'float32' )
      y1 <- K$cast( y1, dtype = 'float32' )

      areaA <- K$expand_dims( ( ( x1 - x ) * ( y1 - y ) ), axis = 1 )
      areaB <- K$expand_dims( ( ( x1 - x ) * ( y - y0 ) ), axis = 1 )
      areaC <- K$expand_dims( ( ( x - x0 ) * ( y1 - y ) ), axis = 1 )
      areaD <- K$expand_dims( ( ( x - x0 ) * ( y - y0 ) ), axis = 1 )

      interpolatedValuesA <- areaA * pixelValuesA
      interpolatedValuesB <- areaB * pixelValuesB
      interpolatedValuesC <- areaC * pixelValuesC
      interpolatedValuesD <- areaD * pixelValuesD

      interpolatedValues <- interpolatedValuesA + interpolatedValuesB +
        interpolatedValuesC + interpolatedValuesD

      return( interpolatedValues )
      },

    makeRegularGrids = function( batchSize, resampledSize )
      {
      K <- keras::backend()

      xLinearSpace <- tensorflow::tf$linspace( -1.0, 1.0, as.integer( resampledSize[2] ) )
      yLinearSpace <- tensorflow::tf$linspace( -1.0, 1.0, as.integer( resampledSize[1] ) )

      coords <- tensorflow::tf$meshgrid( xLinearSpace, yLinearSpace )
      coords[[1]] <- K$flatten( coords[[1]] )
      coords[[2]] <- K$flatten( coords[[2]] )

      ones <- K$ones_like( coords[[1]] )
      regularGrid <- K$concatenate( list( coords[[1]], coords[[2]], ones ), 0L )
      regularGrid <- K$flatten( regularGrid )

      regularGrids <- K$tile( regularGrid, K$stack( list( batchSize ) ) )
      regularGrids <- K$reshape( regularGrids,
        reticulate::tuple( batchSize, 3L, as.integer( prod( resampledSize ) ) ) )

      return( regularGrids )
      },

    affineTransformImage = function( image, affineTransformParameters, resampledSize )
      {
      K <- keras::backend()

      batchSize <- K$shape( image )[1]
      numberOfChannels <- K$shape( image )[4]
      transformParameters <- K$reshape( affineTransformParameters,
        shape = reticulate::tuple( batchSize, 2, 3 ) )

      regularGrids <- makeRegularGrids( batchSize, resampledSize )
      sampledGrids <- K$batch_dot( transformParameters, regularGrids )
      interpolatedImage <- interpolate( image, sampledGrids, resampledSize )
      newOutputShape <- reticulate::tuple( batchSize, resampledSize[1],
        resampledSize[2], numberOfChannels )
      interpolatedImage <- K$reshape( interpolatedImage, shape = newOutputShape )

      return( interpolatedImage )
      }
  )
)

bilinear_interpolation_2d <- function( objects, resampledSize ) {
create_layer( BilinearInterpolationLayer2D, objects,
    list( resampledSize = resampledSize )
    )
}

