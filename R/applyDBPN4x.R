#' applySuperResolutionModel
#'
#' Apply pretrained super-resolution network
#'
#' This function does patch-wise pre-processing and image reconstruction given
#' the input super-resolution model.
#'
#' @param image input image
#' @param model model object or filename see \code{getPretrainedNetwork}
#' @param strideLength stride length should be less than patch size
#' @param targetRange a vector defining min max of each patch, eg -127.5, 127.5
#' Output images will be scaled back to original intensity.  This range should
#' match the mapping used in the training of the network.
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
  strideLength = 20,
  targetRange = c( -127.5, 127.5 ),
  batch_size = 32,
  verbose = FALSE )
{
if ( length( strideLength ) == 1 )
  strl = rep( strideLength, image@dimension ) else strl = strideLength
###############################################
makeNChannelArray <-function( img, nchan, inRange, noizSD = 1 ) {
  X = array( dim = c( dim( img ), nchan ) )
  idiff = inRange[2] - inRange[1]
  img = iMath( img, "Normalize") * idiff + inRange[1]
  mynoise = makeImage( dim( img ), rnorm( prod(dim(img)), 0, noizSD ) )
  mynoise = antsCopyImageInfo( img, mynoise )
  temp = img + mynoise
  temp[ temp < inRange[1] ] = inRange[1]
  temp[ temp > inRange[2] ] = inRange[2]
  for ( k in 1:nchan ) {
    X[,,k] = as.array( temp )
  }
  return( X )
}

if ( verbose ) print( "1. load model" )
tl1 = Sys.time()
if ( is.character( model ) ) {
  if ( file.exists( model ) ) {
    model = load_model_hdf5( model )
    } else stop("Model not found")
  } else stop("Model not found")
if ( verbose ) print( paste("     load --- in : ", Sys.time() - tl1 ) )
shapeLength = length( model$input_shape )
if ( shapeLength == 5 ) { # image dimension is 3
  if ( image@dimension != 3 ) stop("Expecting 3D input for this model")
  lowResolutionPatchSize = c(
    model$input_shape[[2]],
    model$input_shape[[3]],
    model$input_shape[[4]] )
  channelSize = model$input_shape[[5]]
  highResolutionPatchSize = c(
    model$output_shape[[2]],
    model$output_shape[[3]],
    model$output_shape[[4]] )
  channelSizeOut = model$output_shape[[5]]
}
if ( shapeLength == 4 ) { # image dimension is 2
  if ( image@dimension != 2 ) stop("Expecting 2D input for this model")
  lowResolutionPatchSize = c(
    model$input_shape[[2]],
    model$input_shape[[3]] )
  channelSize = model$input_shape[[4]]
  highResolutionPatchSize = c(
    model$output_shape[[2]],
    model$output_shape[[3]] )
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
t0 = Sys.time()
if ( verbose ) print( paste("2. extract patches:", channelSize ) )
if ( channelSize == 1 ) {
  X_test <- extractImagePatches(
    image,
    lowResolutionPatchSize, maxNumberOfPatches = 'all',
    strideLength = strl, returnAsArray = TRUE )
  }

if ( channelSize > 1 ) {
  simg = splitChannels( image )
  X_test <- extractImagePatches(
    simg[[1]],
    lowResolutionPatchSize, maxNumberOfPatches = 'all',
    strideLength = strl, returnAsArray = TRUE )
  for ( k in 2:length( simg ) ) {
    temp <- extractImagePatches(
      simg[[k]],
      lowResolutionPatchSize, maxNumberOfPatches = 'all',
      strideLength = strl, returnAsArray = TRUE )
    X_test = abind::abind( X_test, temp, along = shapeLength )
  }
}
#################################################
numberOfPatches <- dim( X_test )[1]
sliceArray <- function(  myArr, j ) {
  if ( shapeLength == 3 ) {
    return( myArr[j,,] )
  }
  if ( shapeLength == 4 ) {
    return( myArr[j,,,] )
  }
  if ( shapeLength == 5 ) {
    return( myArr[j,,,,] )
  }
}

# this should work for all cases - pretty sure
X_test <- array( data = X_test,
  dim = c( numberOfPatches, lowResolutionPatchSize, channelSize ) )
xvec = as.numeric( sliceArray( X_test, 1 ) )
tempMat = matrix( nrow = numberOfPatches, ncol = length( xvec ))
idiff = targetRange[2] - targetRange[1]
if ( idiff < 0 ) stop("targetRange[2] - targetRange[1] must be positive")

for( j in 1:numberOfPatches )
  {
  temp = as.numeric( sliceArray( X_test, j ) )
  xmax = max( temp, na.rm = T )
  xmin = min( temp, na.rm = T )
  if ( xmax == 0 ) xmax = 1
  temp = ( temp - xmin ) / xmax
  temp = temp * idiff + targetRange[1]
  tempMat[j,] = temp
  }
X_test <- array( data = tempMat,
  dim = c( numberOfPatches, lowResolutionPatchSize, channelSize ) )
rm( tempMat )
gc()
########
expansionFactor = highResolutionPatchSize/lowResolutionPatchSize
bigImg = resampleImage( image,
  dim( image ) * expansionFactor, useVoxels = T )
if ( channelSizeOut != channelSize ) {
  if ( bigImg@components > 1 )
    bigImgSplit = splitChannels( bigImg ) else bigImgSplit=list( bigImg )
  bigavg = antsAverageImages( bigImgSplit )
  blist = list()
  for ( k in 1:channelSizeOut ) {
    blist[[k]] = bigavg
    }
  bigImg = mergeChannels( blist )
  }
t1=Sys.time()
if ( verbose ) print( paste( "     - Extract:", numberOfPatches, "in:", t1-t0 ) )
if ( verbose ) print( "3. ##### prediction" )
pred = predict( model, X_test, batch_size = batch_size )
if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
bigStrides = strl * expansionFactor
if ( channelSize == 1 ) {
  Y_test <- extractImagePatches(
    bigImg,
    highResolutionPatchSize, maxNumberOfPatches = 'all',
    strideLength = bigStrides, returnAsArray = FALSE )
  }
if ( channelSize > 1 ) {
  simg = splitChannels( bigImg )
  Y_test <- extractImagePatches(
    simg[[1]],
    highResolutionPatchSize, maxNumberOfPatches = 'all',
    strideLength = bigStrides, returnAsArray = FALSE )
  for ( k in 2:length( simg ) ) {
    temp <- extractImagePatches(
      simg[[k]],
      highResolutionPatchSize, maxNumberOfPatches = 'all',
      strideLength = bigStrides, returnAsArray = FALSE )
    binddim = length( dim( temp[[1]] ))
    for ( kk in 1:length( Y_test ) )
      Y_test[[kk]] = abind::abind( Y_test[[kk]], temp[[kk]], along = binddim + 1 )
  }
}
# below is a fast simple linear regression model to map patch intensities
# back to the original space defined by the upsampled image
if ( verbose ) print( "4. reconstruct intensity" )
plist = list()
for( j in seq_len( numberOfPatches ) ) {
  temp = sliceArray( pred, j )
  ivec1 = as.numeric( temp )
  ivec2 = as.numeric( Y_test[[j]])
  mybeta = cov(ivec1,ivec2)/var(ivec1)
  ivec1t = mybeta * ivec1
  ivec1t = ivec1t + mean( ivec2 ) - mean( ivec1t )
  plist[[j]] = array( ivec1t, dim = dim( temp ) )
  }
if ( verbose ) print( "5. reconstruct full image" )
predImg = reconstructImageFromPatches( plist,
  bigImg,   strideLength = bigStrides )
return( predImg )
}
