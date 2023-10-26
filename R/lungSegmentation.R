#' Functional lung segmentation.
#'
#' Perform functional lung segmentation using hyperpolarized gases.
#'
#' \url{https://pubmed.ncbi.nlm.nih.gov/30195415/}
#'
#' @param ventilationImage input ventilation image
#' @param mask input mask image
#' @param useCoarseSlicesOnly if \code{TRUE}, apply network only in the
#' dimension of greatest slice thickness.  If \code{FALSE}, apply to all
#' dimensions and average the results.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return ventilation segmentation and corresponding probability images
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "flair.nii.gz" )
#' probabilityMask <-sysuMediaWmhSegmentation( image )
#' }
#' @export
elBicho <- function( ventilationImage, mask,
   useCoarseSlicesOnly = TRUE, antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( ventilationImage@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( any( dim( ventilationImage ) != dim( mask ) ) )
    {
    stop( "Ventilation image and mask size are not the same size." )
    }

  ################################
  #
  # Preprocess image
  #
  ################################

  templateSize <- c( 256L, 256L )
  classes <- c( 0, 1, 2, 3, 4 )
  numberOfClassificationLabels <- length( classes )

  imageModalities <- c( "Ventilation", "Mask" )
  channelSize <- length( imageModalities )

  preprocessedImage <- ( ventilationImage - mean( ventilationImage ) ) /
    sd( ventilationImage )

  ################################
  #
  # Build models and load weights
  #
  ################################

  unetModel <- createUnetModel2D( c( templateSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
    weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

  if( verbose == TRUE )
    {
    cat( "El Bicho:  retrieving model weights.\n" )
    }
  weightsFileName <- getPretrainedNetwork( "elBicho", antsxnetCacheDirectory = antsxnetCacheDirectory )
  unetModel$load_weights( weightsFileName )

  ################################
  #
  # Extract slices
  #
  ################################

  dimensionsToPredict <- c( which.max( antsGetSpacing( preprocessedImage ) )[1] )

  if( useCoarseSlicesOnly == FALSE )
    {
    dimensionsToPredict <- 1:3
    }

  batchX <- array( data = 0,
    c( sum( dim( preprocessedImage )[dimensionsToPredict]), templateSize, channelSize ) )

  sliceCount <- 1
  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    numberOfSlices <- dim( preprocessedImage )[dimensionsToPredict[d]]

    if( verbose == TRUE )
      {
      cat( "Extracting slices for dimension", dimensionsToPredict[d], "\n" )
      pb <- txtProgressBar( min = 1, max = numberOfSlices, style = 3 )
      }

    for( i in seq.int( numberOfSlices ) )
      {
      if( verbose )
        {
        setTxtProgressBar( pb, i )
        }

      ventilationSlice <- padOrCropImageToSize( extractSlice( preprocessedImage, i, dimensionsToPredict[d] ), templateSize )
      batchX[sliceCount,,,1] <- as.array( ventilationSlice )

      maskSlice <- padOrCropImageToSize( extractSlice( mask, i, dimensionsToPredict[d] ), templateSize )
      batchX[sliceCount,,,2] <- as.array( maskSlice )

      sliceCount <- sliceCount + 1
      }
    if( verbose == TRUE )
      {
      cat( "\n" )
      }
    }

  ################################
  #
  # Do prediction and then restack into the image
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Prediction.\n" )
    }

  prediction <- predict( unetModel, batchX, verbose = verbose )

  permutations <- list()
  permutations[[1]] <- c( 1, 2, 3 )
  permutations[[2]] <- c( 2, 1, 3 )
  permutations[[3]] <- c( 2, 3, 1 )

  probabilityImages <- list()
  for( l in seq.int( numberOfClassificationLabels ) )
    {
    probabilityImages[[l]] <- antsImageClone( mask ) * 0
    }

  currentStartSlice <- 1
  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    currentEndSlice <- currentStartSlice - 1 + dim( preprocessedImage )[dimensionsToPredict[d]]
    whichBatchSlices <- currentStartSlice:currentEndSlice
    for( l in seq.int( numberOfClassificationLabels ) )
      {
      predictionPerDimension <- prediction[whichBatchSlices,,,l]
      predictionArray <- aperm( drop( predictionPerDimension ), permutations[[dimensionsToPredict[d]]] )
      predictionImage <- antsCopyImageInfo( preprocessedImage,
        padOrCropImageToSize( as.antsImage( predictionArray ), dim( preprocessedImage ) ) )
      probabilityImages[[l]] <- probabilityImages[[l]] + ( predictionImage - probabilityImages[[l]] ) / d
      }
    currentStartSlice <- currentEndSlice + 1
    }

  ################################
  #
  # Convert probability images to segmentation
  #
  ################################

  imageMatrix <- imageListToMatrix( probabilityImages[2:length( probabilityImages )], mask * 0 + 1 )
  backgroundForegroundMatrix <- rbind( imageListToMatrix( list( probabilityImages[[1]] ), mask * 0 + 1 ),
                                      colSums( imageMatrix ) )
  foregroundMatrix <- matrix( apply( backgroundForegroundMatrix, 2, which.max ), nrow = 1 ) - 1
  segmentationMatrix <- ( matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) ) * foregroundMatrix
  segmentationImage <- matrixToImages( segmentationMatrix, mask * 0 + 1 )[[1]]

  return( list( segmentationImage = segmentationImage,
                probabilityImages = probabilityImages ) )
}

#' Pulmonary artery segmentation.
#'
#' Perform pulmonary artery segmentation.  Training data taken from the 
#' PARSE2022 challenge (Luo, Gongning, et al. "Efficient automatic segmentation 
#' for multi-level pulmonary arteries: The PARSE challenge." 
#' https://arxiv.org/abs/2304.03708).
#'
#' @param ct input 3-D ct image.
#' @param lungMask input binary lung mask which defines the patch extraction.  
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
lungPulmonaryArterySegmentation <- function( ct, lungMask = NULL, 
  predictionBatchSize = 16, patchStrideLength = 32,
  doPreprocessing = TRUE, antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  patchSize <- c( 160, 160, 160 )

  if( any( dim( ct ) < patchSize ) )
    {
    stop( "Images must be > 160 voxels per dimension." )
    }

  ################################
  #
  # Preprocess images
  #
  ################################

  if( is.null( lungMask ) )
    {
    lungMask <- thresholdImage( lungExtraction( ct, modality = "ct", verbose = verbose ), 1, 3, 1, 0 )
    }
  ctPreprocessed <- antsImageClone( ct )
  ctPreprocessed <- ( ctPreprocessed + 800 ) / ( 500 + 800 )
  ctPreprocessed[ctPreprocessed > 1.0] <- 1.0
  ctPreprocessed[ctPreprocessed < 0.0] <- 0.0

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
  numberOfClassificationLabels <- 2
  channelSize <- 1

  model <- createUnetModel3D( c( patchSize, channelSize ),
               numberOfOutputs = numberOfClassificationLabels, mode = "sigmoid",
               numberOfFilters = c( 32, 64, 128, 256, 512 ),
               convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
               dropoutRate = 0.0, weightDecay = 0.0 )
  weightsFileName <- getPretrainedNetwork( "pulmonaryArteryWeights", antsxnetCacheDirectory = antsxnetCacheDirectory )
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

  ctPatches <- extractImagePatches( ct,
                                    patchSize = patchSize,
                                    maxNumberOfPatches = "all",
                                    strideLength = patchStrideLength,
                                    maskImage = lungMask,
                                    randomSeed = NULL,
                                    returnAsArray = TRUE )
  totalNumberOfPatches <- dim( ctPatches )[1]
 
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
    batchX[,,,,1] <- ctPatches[indices,,,]

    if( verbose )
      {
      message( "  Predicting batch ", b, " of ", numberOfBatches )
      }
    p <- model %>% predict( batchX, verbose = verbose )
    prediction[indices,,,,] <- p[,,,,2, drop=FALSE]
    }  

  if( verbose )
    {
    message( "Predict patches and reconstruct." )
    }
  probabilityImage <- reconstructImageFromPatches( drop( prediction[,,,,2] ),
                                                   strideLength = patchStrideLength,
                                                   domainImage = lungMask,
                                                   domainImageIsMask = TRUE )
 
  return( probabilityImage )
  }



