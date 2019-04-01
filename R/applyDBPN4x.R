#' applyDBPNsr
#'
#' Apply pretrained DBPN super-resolution network
#'
#' @param image input image
#' @param modelFilename see \code{getPretrainedNetwork}
#' @param expansionFactor size to upscale, should match network pretraining
#' @param strideLength stride length should be less than patch size
#' @param lowResolutionPatchSize patch size at low resolution that will be
#' expanded by expansionFactor - needs to be consistent with trained model.
#' @param standardDeviationOffset an offset added to the normalization of patches to prevent divisions by zero
#' @param verbose If \code{TRUE}, show status messages
#' @return image upscaled to resolution provided by network
#' @author Avants BB
#' @examples
#' \dontrun{
#' simg <- applyDBPNsr( ri( 1 ), getPretrainedNetwork( "dbpn4x" ) )
#' }
#' @export applyDBPNsr
applyDBPNsr <- function(
  image,
  modelFilename,
  expansionFactor = 4,
  strideLength = 20,
  lowResolutionPatchSize = c( 48, 48 ),
  standardDeviationOffset = 0.01,
  verbose = FALSE )
{
if ( verbose ) print( paste( "perform-*x-sr" ) )
channelSize = 1
highResolutionPatchSize <- round( lowResolutionPatchSize * expansionFactor )
strl = strideLength
###############################################
inputImageSize = c( lowResolutionPatchSize, 1 )
if ( verbose ) print( "1. load model" )
tl1 = Sys.time()
if ( file.exists( modelFilename ) )
  srModel = load_model_hdf5( modelFilename ) else stop( paste( modelFilename, 'does not exist' ) )
if ( verbose ) print( paste("      --- in : ", Sys.time() - tl1 ) )
###############
t0 = Sys.time()
if ( verbose ) print("2. extract patches")
X_test <- extractImagePatches(
  iMath( image, "Normalize" ) * 255,
  lowResolutionPatchSize, maxNumberOfPatches = 'all',
  strideLength = strl, returnAsArray = TRUE )
#################################################
numberOfPatches <- dim( X_test )[1]
X_test <- array( data = X_test,
  dim = c( numberOfPatches, lowResolutionPatchSize, channelSize ) )
xmean = xsd = rep( NA, numberOfPatches )
for( j in seq_len( numberOfPatches ) )
  {
  xmean[ j ] = mean( X_test[j,,,1], na.rm = T )
  xsd[ j ] = sd( X_test[j,,,1], na.rm = T )
  X_test[j,,,1] <- ( X_test[j,,,1] - xmean[j] ) / ( xsd[j] + standardDeviationOffset )
  }
########
bigImg = resampleImage( iMath( image, "Normalize" ) * 255,
  antsGetSpacing( image )/expansionFactor, useVoxels = F )
t1=Sys.time()
if ( verbose ) print( paste( "     - Extract:", numberOfPatches, "in:", t1-t0 ) )
if ( verbose ) print( "3. ##### prediction" )
pred = predict( srModel, X_test, batch_size = 32 )
if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
plist = list()
for( j in seq_len( numberOfPatches ) )
  plist[[j]] = ( pred[j,,,1] ) * ( xsd[j] + standardDeviationOffset) + xmean[j]
if ( verbose ) print( "4. reconstruct" )
predImg = reconstructImageFromPatches( plist,
  bigImg,   strideLength = strl * expansionFactor )

return( predImg )
}
