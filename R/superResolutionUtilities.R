#' Mean square error of a single image or between two images.
#'
#' @param x input image.
#' @param y input image.
#'
#' @return the mean squared error
#' @author Avants BB (from redr)
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' mseValue <- MSE( r16, r85 )
#'
#' @export
MSE <- function( x, y = NULL )
{
  if( is.null( y ) )
    {
    return( mean( x^2 ) )
    } else {
    return( mean ( ( x - y )^2 ) )
    }
}

#' Mean absolute error of a single image or between two images.
#'
#' @param x input image.
#' @param y input image.
#'
#' @return the mean absolute error
#' @author Avants BB (from redr)
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' maeValue <- MAE( r16, r85 )
#'
#' @export
MAE <- function( x, y = NULL )
{
  if( is.null( y ) )
    {
    return( mean( abs( x ) ) )
    } else {
    return( mean ( abs( x - y ) ) )
    }
}

#' Peak signal-to-noise ratio between two images.
#'
#' @param x input image.
#' @param y input image.
#'
#' @return the peak signal-to-noise ratio
#' @author Avants BB
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' psnrValue <- PSNR( r16, r85 )
#'
#' @export
PSNR <- function( x, y )
{
  return( 20 * log10( max( x ) ) - 10 * log10( MSE( x, y ) ) )
}

#' Structural similarity index (SSI) between two images.
#'
#' Implementation of the SSI quantity for two images proposed in
#'
#' Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli. "Image quality
#' assessment: from error visibility to structural similarity". IEEE TIP.
#' 13 (4): 600–612.
#'
#' @param x input image.
#' @param y input image.
#' @param K vector of length 2 which contain SSI parameters meant to stabilize
#' the formula in case of weak denominators.
#'
#' @return the structural similarity index
#' @author Avants BB
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' ssimValue <- SSIM( r16, r85 )
#'
#' @export
SSIM <- function( x, y, K = c( 0.01, 0.03 ) )
{
  globalMax <- max( max( x ), max( y ) )
  globalMin <- abs( min( min( x ), min( y ) ) )
  L <- globalMax - globalMin

  C1 <- ( K[1] * L )^2
  C2 <- ( K[2] * L )^2
  C3 <- C2 / 2

  mu_x <- mean( x )
  mu_y <- mean( y )

  mu_x_sq <- mu_x * mu_x
  mu_y_sq <- mu_y * mu_y
  mu_xy <- mu_x * mu_y

  sigma_x_sq <- mean( x * x ) - mu_x_sq
  sigma_y_sq <- mean( y * y ) - mu_y_sq
  sigma_xy <- mean( x * y ) - mu_xy

  numerator <- ( 2 * mu_xy + C1 ) * ( 2 * sigma_xy + C2 )
  denominator <- ( mu_x_sq + mu_y_sq + C1 ) * ( sigma_x_sq + sigma_y_sq + C2 )

  SSI <- numerator / denominator

  return( SSI )
}

#' Gradient Magnitude Similarity Deviation
#'
#' A fast and simple metric that correlates to perceptual quality
#'
#' @param x input image.
#' @param y input image.
#'
#' @return scalar
#' @author Avants BB
#' @examples
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( 'r16' ) )
#' r85 <- antsImageRead( getANTsRData( 'r85' ) )
#' value <- GMSD( r16, r85 )
#'
#' @export
GMSD <- function( x, y )
{
  gx <- iMath( x, "Grad" )
  gy <- iMath( y, "Grad" )

  # see eqn 4 - 6 in https://arxiv.org/pdf/1308.3052.pdf

  constant <- 0.0026
  gmsd_numerator <- 2.0 * gx * gy + constant
  gmsd_denominator <- gx^2 + gy^2 + constant
  gmsd <- gmsd_numerator / gmsd_denominator

  prefactor <-  1.0 / prod( dim( x ) )

  return( sqrt( prefactor * sum( ( gmsd - mean( gmsd )  )^2 ) ) )
}

#' Apply a pretrained model for super resolution.
#'
#' Helper function for applying a pretrained super resolution model.
#' Apply a patch-wise trained network to perform super-resolution. Can be applied
#' to variable sized inputs. Warning: This function may be better used on CPU
#' unless the GPU can accommodate the full image size. Warning 2: The global
#' intensity range (min to max) of the output will match the input where the
#' range is taken over all channels.
#'
#' @param image input image.
#' @param model pretrained model or filename (cf \code{getPretrainedNetwork}).
#' @param targetRange a vector defining the \code{c(min, max)} of each input
#'                    image (e.g., -127.5, 127.5).  Output images will be scaled
#'                    back to original intensity. This range should match the
#'                    mapping used in the training of the network.
#' @param batchSize batch size used for the prediction call.
#' @param regressionOrder if specified, then apply the function
#'                        \code{regressionMatchImage} with
#'                        \code{polyOrder = regressionOrder}.
#' @param verbose If \code{TRUE}, show status messages.
#' @return super-resolution image upscaled to resolution specified by the network.
#' @author Avants BB
#' @examples
#' \dontrun{
#' image <- applySuperResolutionModelToImage( ri( 1 ), getPretrainedNetwork( "dbpn4x" ) )
#' }
#' @export
applySuperResolutionModelToImage <- function( image, model,
  targetRange = c( -127.5, 127.5 ), batchSize = 32, regressionOrder = NA,
  verbose = FALSE )
{
  channelAxis <- 1L
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1L
    }

  shapeLength <- length( model$input_shape )

  if( shapeLength < 4 || shapeLength > 5 )
    {
    stop( "Unexpected input shape." )
    } else if( shapeLength == 5 && image@dimension != 3 )
    {
    stop( "Expecting 3D input for this model." )
    } else if( shapeLength == 4 && image@dimension != 2 ) {
    stop( "Expecting 2D input for this model." )
    }

  if( channelAxis == -1L )
    {
    channelAxis <- shapeLength
    }
  channelSize <- model$input_shape[[channelAxis]]

  if( channelSize != image@components )
    {
    stop( paste( "Channel size of model", channelSize,
      'does not match ncomponents=', image@components, 'of the input image.') )
    }

  if( targetRange[1] > targetRange[2] )
    {
    targetRange = rev( targetRange )
    }

  if( ! is.object( model ) && is.character( model ) )
    {
    if( file.exists( model ) )
      {
      startTime <- Sys.time()
      if( verbose )
        {
        cat( "Load model." )
        }
      model <- load_model_hdf5( model )
      if( verbose )
        {
        elapsedTime <- Sys.time() - startTime
        cat( "  (elapsed time: ", elapsedTime, ")" )
        }
      } else {
      stop( "Model not found." )
      }
    }

  imagePatches <- extractImagePatches(
    image, dim( image ), maxNumberOfPatches = 1,
    strideLength = dim( image ), returnAsArray = TRUE )
  imagePatches <- array( imagePatches, dim = c( 1,  dim( image ),
    image@components ) )
  imagePatches <- imagePatches - min( imagePatches )
  imagePatches <- imagePatches / max( imagePatches ) *
    ( targetRange[2] - targetRange[1] ) + targetRange[1]

  if( verbose )
    {
    cat( "Prediction\n" )
    }
  startTime <- Sys.time()
  prediction <- predict( model, imagePatches, batch_size = batchSize )

  if( verbose )
    {
    elapsedTime <- Sys.time() - startTime
    cat( " (elapsed time: ", elapsedTime, ")" )
    }

  if( verbose )
    {
    cat( "Reconstruct intensities." )
    }

  intensityRange <- range( image )
  prediction <- prediction - min( prediction )
  prediction <- prediction / max( prediction ) *
    ( intensityRange[2] - intensityRange[1] ) + intensityRange[1]

  sliceArrayChannel <- function( inputArray, slice )
    {
    if( channelAxis == 1 )
      {
      if( shapeLength == 4 )
        {
        return( inputArray[slice,,,] )
        } else {
        return( inputArray[slice,,,,] )
        }
      } else {
      if( shapeLength == 4 )
        {
        return( inputArray[,,,slice] )
        } else {
        return( inputArray[,,,,slice] )
        }
      }
    }

  expansionFactor <- ( dim( prediction ) / dim( imagePatches ) )[-1][1:image@dimension]
  if( channelAxis == 1 )
    {
    expansionFactor <- ( dim( prediction ) / dim( imagePatches ) )[2:( 1 + image@dimension )]
    }
  if ( verbose )
    {
    cat( "ExpansionFactor:", paste( expansionFactor, collapse = 'x' ) )
    }

  if( image@components == 1 )
    {
    imageArray <- sliceArrayChannel( prediction, 1 )
    predictionImage = makeImage( dim( image ) * expansionFactor, imageArray )
    if( ! is.na( regressionOrder ) )
      {
      referenceImage <- resampleImageToTarget( image, predictionImage )
      predictionImage <- regressionMatchImage( predictionImage, referenceImage,
        polyOrder = regressionOrder  )
      }
    } else {
    imageComponentList <- list()
    for( k in seq_len( image@components ) )
      {
      imageArray <- sliceArrayChannel( prediction, k )
      imageComponentList[[k]] <- makeImage(
        dim( image ) * expansionFactor, imageArray )
      }
    predictionImage <- mergeChannels( imageComponentList )
    }

  predictionImage <- antsCopyImageInfo( image, predictionImage )
  antsSetSpacing( predictionImage, antsGetSpacing( image ) / expansionFactor )
  return( predictionImage )
  }

