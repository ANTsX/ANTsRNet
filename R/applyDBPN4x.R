#' applySuperResolutionModel
#'
#' Apply pretrained super-resolution network
#'
#' This function makes several assumptions about how the network was trained.
#' It assumes input data was scaled to the range of 0 to 255.  Optionally, 
#' additional patch-wise mean subtraction and normalization by standard deviation
#' is applied.  Otherwie, a single additive constant may be subtracted from each
#' patch.  See the code if questions arise.
#'
#' @param image input image
#' @param model model object or filename see \code{getPretrainedNetwork}
#' @param expansionFactor size to upscale, should match network pretraining
#' @param strideLength stride length should be less than patch size
#' @param lowResolutionPatchSize patch size at low resolution that will be
#' expanded by expansionFactor - needs to be consistent with trained model.
#' @param additiveOffset a single number to add to the patch intensity.  if this is missing, then it is not applied.  instead, the mean will be subtracted from the patch.
#' @param standardDeviationOffset an offset added to the normalization of patches 
#' to prevent divisions by zero.  if this is missing, then it is not applied.
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
  expansionFactor = 4,
  strideLength = 20,
  lowResolutionPatchSize = c( 48, 48 ),
  additiveOffset,
  standardDeviationOffset,
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
if ( is.character( model ) )
  if ( file.exists( model ) )
    model = load_model_hdf5( model ) 
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
if ( missing( additiveOffset ) & !missing( standardDeviationOffset ) )
  for( j in seq_len( numberOfPatches ) )
    {
    xmean[ j ] = mean( X_test[j,,,1], na.rm = T )
    xsd[ j ] = sd( X_test[j,,,1], na.rm = T )
    X_test[j,,,1] <- ( X_test[j,,,1] - xmean[j] ) / ( xsd[j] + standardDeviationOffset )
    }

if ( ! missing( additiveOffset ) ) X_test = X_test - additiveOffset
########
bigImg = resampleImage( iMath( image, "Normalize" ) * 255,
  antsGetSpacing( image )/expansionFactor, useVoxels = F )
t1=Sys.time()
if ( verbose ) print( paste( "     - Extract:", numberOfPatches, "in:", t1-t0 ) )
if ( verbose ) print( "3. ##### prediction" )
pred = predict( model, X_test, batch_size = 32 )
if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
plist = list()
if ( missing( additiveOffset ) & !missing( standardDeviationOffset ) )
  for( j in seq_len( numberOfPatches ) )
    plist[[j]] = ( pred[j,,,1] ) * ( xsd[j] + standardDeviationOffset) + xmean[j]

if ( ! missing( additiveOffset ) )
  for( j in seq_len( numberOfPatches ) )
    plist[[j]] = ( pred[j,,,1] ) + additiveOffset
if ( verbose ) print( "4. reconstruct" )
predImg = reconstructImageFromPatches( plist,
  bigImg,   strideLength = strl * expansionFactor )

return( predImg )
}
