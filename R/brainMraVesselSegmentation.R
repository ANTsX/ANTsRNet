#' MRA-TOF vessel segmentation.
#'
#' Perform MRA-TOF vessel segmentation.  Training data taken from the 
#' (https://data.kitware.com/#item/58a372e48d777f0721a64dc9). 
#'
#' @param mra input mra image.
#' @param mask input binary mask which defines the patch extraction.  
#' If not supplied, one is estimated.
#' @param predictionBatchSize Control memory usage for prediction.  More consequential 
#' for GPU-usage.
#' @param patchStrideLength  3-D vector or int.   Dictates the stride length for 
#' accumulating predicting patches.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return Probability image.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' }
#' @export
brainMraVesselSegmentation <- function( mra, mask = NULL, 
  predictionBatchSize = 16, patchStrideLength = 32,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  ################################
  #
  # Preprocess images
  #
  ################################

  if( is.null( mask ) )
    {
    mask <- brainExtraction( mra, modality = "mra", 
                              antsxnetCacheDirectory = antsxnetCacheDirectory,
                              verbose = verbose )
    mask <- thresholdImage( mask, 0.5, 1.1, 1, 0 )
    }
  
  template <- antsImageRead( getANTsXNetData( "mraTemplate" ) )
  templateBrainMask <- antsImageRead( getANTsXNetData( "mraTemplateBrainMask" ) )

  mraPreprocessed <- antsImageClone( mra )
  mraPreprocessed[mask == 1] <- ( mraPreprocessed[mask == 1] - min( mraPreprocessed[mask == 1] ) ) / 
                                ( max( mraPreprocessed[mask == 1] ) - min( mraPreprocessed[mask == 1] ) )
  reg <- antsRegistration( template * templateBrainMask, mraPreprocessed * mask,
                           typeofTransform = "antsRegistrationSyNQuick[a]",
                           verbose = verbose )
  mraPreprocessed <- antsImageClone( reg$warpedmovout )

  patchSize <- c( 160, 160, 160 )

  if( any( dim( mra ) < patchSize ) )
    {
    stop( "Images must be > 160 voxels per dimension." )
    }

  templateMraPrior <- antsImageRead( getANTsXNetData( "mraTemplateVesselPrior" ) )
  templateMraPrior <- ( ( templateMraPrior - ANTsR::min( templateMraPrior ) ) /
                        ( ANTsR::max( templateMraPrior ) - ANTsR::min( templateMraPrior ) ) )

  ################################
  #
  # Build model and load weights
  #
  ################################

  if( verbose )
    {
    message( "Load model and weights.\n" )
    }

  if( is.double( patchStrideLength ) || is.integer( patchStrideLength ) )
    {
    patchStrideLength <- rep( as.integer( patchStrideLength ), 3 )
    }
  channelSize <- 2

  model <- createUnetModel3D( c( patchSize, channelSize ),
               numberOfOutputs = 1, mode = "sigmoid",
               numberOfFilters = c( 32, 64, 128, 256, 512 ),
               convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
               dropoutRate = 0.0, weightDecay = 0.0 )
  weightsFileName <- getPretrainedNetwork( "mraVesselWeights_160", 
                                           antsxnetCacheDirectory = antsxnetCacheDirectory )
  load_model_weights_hdf5( model, filepath = weightsFileName )

  ################################
  #
  # Extract patches
  #
  ################################

  if( verbose )
    {
    message( "Extract patches." )
    }

  mraPatches <- extractImagePatches( mraPreprocessed,
                                     patchSize = patchSize,
                                     maxNumberOfPatches = "all",
                                     strideLength = patchStrideLength,
                                     maskImage = lungMask,
                                     randomSeed = NULL,
                                     returnAsArray = TRUE )
  mraPriorPatches <- extractImagePatches( templateMraPrior,
                                     patchSize = patchSize,
                                     maxNumberOfPatches = "all",
                                     strideLength = patchStrideLength,
                                     maskImage = lungMask,
                                     randomSeed = NULL,
                                     returnAsArray = TRUE )
  totalNumberOfPatches <- dim( mraPatches )[1]
 
  ################################
  #
  # Do prediction and then restock into the image
  #
  ################################

  numberOfBatches <- floor( totalNumberOfPatches / predictionBatchSize )
  residualNumberOfPatches <- totalNumberOfPatches - numberOfBatches * predictionBatchSize
  if( residualNumberOfPatches > 0 )
    {
    numberOfBatches <- numberOfBatches + 1 
    }

  if( verbose )
    {
    message( "  Total number of patches: ", totalNumberOfPatches )
    message( "  Prediction batch size: ", predictionBatchSize )
    message( "  Number of batches: ", numberOfBatches )
    }
 
  prediction <- array( data = 0, dim = c( totalNumberOfPatches, patchSize, 1 ) )
  for( b in seq.int( numberOfBatches ) )
    {
    batchX <- NULL
    if( b < numberOfBatches || residualNumberOfPatches == 0 )
      {
      batchX <- array( data = 0, dim = c( predictionBatchSize, patchSize, channelSize ) ) 
      } else {
      batchX <- array( data = 0, dim = c( residualNumberOfPatches, patchSize, channelSize ) ) 
      }

    indices <- ( ( b - 1 ) * predictionBatchSize + 1):( ( b - 1 ) * predictionBatchSize + dim( batchX )[1] )
    batchX[,,,,1] <- mraPatches[indices,,,]
    batchX[,,,,2] <- mraPriorPatches[indices,,,]

    if( verbose )
      {
      message( "  Predicting batch ", b, " of ", numberOfBatches )
      }
    prediction[indices,,,,] <- model %>% predict( batchX, verbose = verbose )
    }  

  if( verbose )
    {
    message( "Predict patches and reconstruct." )
    }
  probabilityImageWarped <- reconstructImageFromPatches( drop( prediction[,,,,1] ),
                                                   strideLength = patchStrideLength,
                                                   domainImage = templateBrainMask,
                                                   domainImageIsMask = TRUE )
  probabilityImage <- antsApplyTransforms( mra, probabilityImageWarped,
                                           transformlist = reg$invtransforms,
                                           whichtoinvert = c( TRUE ),
                                           verbose = verbose )
 
  return( probabilityImage )
  }

