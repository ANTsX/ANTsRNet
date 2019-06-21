#' applySuperResolutionModel
#'
#' Apply pretrained super-resolution network
#'
#' Apply a patch-wise trained network to perform super-resolution. Can be applied
#' to variable sized inputs. Warning: This function may be better used on CPU
#' unless the GPU can accommodate the full image size. Warning 2: The global
#' intensity range (min to max) of the output will match the input where the
#' range is taken over all channels.
#'
#' @param image input image
#' @param model model object or filename see \code{getPretrainedNetwork}
#' @param targetRange a vector defining min max of each the input image,
#' eg -127.5, 127.5.  Output images will be scaled back to original intensity.
#' This range should match the mapping used in the training of the network.
#' @param batch_size for prediction call
#' @param verbose If \code{TRUE}, show status messages
#' @return image upscaled to resolution provided by network
#' @author Avants BB
#' @examples
#' \dontrun{
#' simg <- applySuperResolutionModel( ri( 1 ), getPretrainedNetwork( "dbpn4x" ) )
#' }
#' @export applySuperResolutionModel
applySuperResolutionModel <- function(
  image,
  model,
  targetRange,
  batch_size = 32,
  verbose = FALSE )
{
linMatchIntensity <- function( fromImg, toImg ) {
  mdl = lm(  as.numeric( toImg ) ~ as.numeric( fromImg ) )
  pp = predict( mdl )
  pp[ pp < min( toImg ) ] = min( toImg )
  pp[ pp > max( toImg ) ] = max( toImg )
  newImg = makeImage( dim( fromImg ), pp )
  temp = antsCopyImageInfo( fromImg,  newImg )
  return( newImg )
  }
if ( ! missing( targetRange ) )
  if ( targetRange[1] > targetRange[2] )
    targetRange = rev( targetRange )

if ( verbose ) print( "1. load model" )
tl1 = Sys.time()
if ( ! is.object( model )  )
  if ( is.character( model ) ) {
    if ( file.exists( model ) ) {
      model = load_model_hdf5( model )
      } else stop("Model not found")
    }
if ( verbose ) print( paste("load model in : ", Sys.time() - tl1 ) )
shapeLength = length( model$input_shape )
if ( shapeLength == 5 ) { # image dimension is 3
  if ( image@dimension != 3 ) stop("Expecting 3D input for this model")
  channelSize = model$input_shape[[5]]
  channelSizeOut = model$output_shape[[5]]
}
if ( shapeLength == 4 ) { # image dimension is 2
  if ( image@dimension != 2 ) stop("Expecting 2D input for this model")
  channelSize = model$input_shape[[4]]
  channelSizeOut = model$output_shape[[4]]
}
# check image components matches nchannels, otherwise replicate the input
ncomponents = image@components
if ( channelSize != ncomponents ) {
  stop( paste(
    "ChannelSize of model", channelSize,
    'does not match ncomponents', ncomponents, 'of image') )
}
###############
X_test <- extractImagePatches(
    image,
    dim( image ), maxNumberOfPatches = 1,
    strideLength = dim( image ), returnAsArray = TRUE )
if ( ! missing( targetRange ) ) {
  X_test = X_test - min( X_test )
  X_test = X_test / max( X_test ) * ( targetRange[2] - targetRange[1] ) + targetRange[1]
}
#################################################
if ( verbose ) print( "##### prediction" )
t1 = Sys.time()
pred = predict( model, X_test, batch_size = batch_size )
if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
# below is a fast simple linear regression model to map patch intensities
if ( verbose ) print( "4. reconstruct intensity" )
if ( ! missing( targetRange ) ) {
  temp = range( image )
  pred = pred - min( pred )
  pred = pred / max( pred ) * ( temp[2] - temp[1] ) + temp[1]
  }
sliceArray <- function(  myArr, j ) {
  if ( shapeLength == 3 ) {
    return( myArr[1,,j] )
  }
  if ( shapeLength == 4 ) {
    return( myArr[1,,,j] )
  }
  if ( shapeLength == 5 ) {
    return( myArr[1,,,,j] )
  }
}

expansionFactor = ( dim( pred ) / dim( X_test ) )[-1][1:image@dimension]
if ( verbose )
  print( paste( "expansionFactor: ", paste( expansionFactor, collapse= 'x' ) ) )

if ( tail(dim(pred),1) == 1 ) {
  ivec = sliceArray( pred, 1 )
  predImg = makeImage( dim( image ) * expansionFactor, ivec )
}
if ( tail(dim(pred),1) > 1 ) {
  mcList = list()
  for ( k in 1:tail(dim(pred),1) ) {
    ivec = sliceArray( pred, k )
    mcList[[k]] = makeImage( dim( image ) * expansionFactor, ivec )
    }
  predImg = mergeChannels( mcList )
}
predImg = antsCopyImageInfo( image, predImg )
antsSetSpacing( predImg, antsGetSpacing( image ) / expansionFactor )
return( predImg )
}
