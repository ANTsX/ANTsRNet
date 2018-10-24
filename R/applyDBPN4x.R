#' applyDBPNsr
#'
#' Apply pretrained DBPN super-resolution network
#'
#' @param image input image
#' @param modelFilename see \code{getPretrainedNetwork}
#' @param expansionFactor size to upscale, should match network pretraining
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
  expansionFactor,
  verbose = FALSE )
{
if ( verbose ) print( paste( "perform-*x-sr" ) )
channelSize = 1
lowResolutionPatchSize <- c( 48, 48 ) # from EDSR paper
if ( missing( expansionFactor ) ) expansionFactor = 4
highResolutionPatchSize <- round( lowResolutionPatchSize * expansionFactor )
strl = 20
###############################################
inputImageSize = c( lowResolutionPatchSize, 1 )
if ( verbose ) print( "1. load model" )
tl1 = Sys.time()
if ( ! exists( "srModel" ) )
  srModel = load_model_hdf5( modelFilename )
if ( verbose ) print( paste("      --- in : ", Sys.time() - tl1 ) )
###############
t0 = Sys.time()
if ( verbose ) print("2. extract patches")
X_test <- extractImagePatches( image,
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
  X_test[j,,,1] <- ( X_test[j,,,1] - xmean[j] ) / ( xsd[j] + 1 )
  }
########
bigimage = resampleImage( image, antsGetSpacing( image )/expansionFactor, useVoxels = F )
t1=Sys.time()
if ( verbose ) print( paste( "     - Extract:", numberOfPatches, "in:", t1-t0 ) )
if ( verbose ) print( "3. ##### prediction" )
pred = predict( srModel, X_test, batch_size = 32 )
if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
plist = list()
for( j in seq_len( numberOfPatches ) )
  plist[[j]] = ( pred[j,,,1] ) * ( xsd[j] + 1) + xmean[j]
if ( verbose ) print( "4. reconstruct" )
predimage = reconstructImageFromPatches( plist,
  bigimage,   strideLength = strl * expansionFactor )
return( predimage )
}
