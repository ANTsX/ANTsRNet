#' Brain tumor probabilistic segmentation
#'
#' Perform brain tumor probabilistic segmentation given pre-aligned 
#' FLAIR, T1, T1 contrast, and T2 images.  Note that the underlying
#' model is 3-D and requires images to be of > 64 voxels in each
#' dimension.
#'
#' @param flair input 3-D FLAIR brain image (not skull-stripped).
#' @param t1 input 3-D T1-weighted brain image (not skull-stripped).
#' @param t1Contrast input 3-D T1-weighted contrast brain image (not skull-stripped).
#' @param t2 input 3-D T2-weighted brain image (not skull-stripped).
#' @param predictionBatchSize Control memory usage for prediction.  More consequential 
#' for GPU-usage.
#' @param patchStrideLength  3-D vector or int.   Dictates the stride length for 
#' accumulating predicting patches.
#' @param doPreprocessing perform n4 bias correction, intensity truncation, brain
#' extraction.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return Brain tumor segmentation probability images (4 tumor tissue types).
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' t1 <- antsImageRead( "t1.nii.gz" )
#' flair <- antsImageRead( "flair.nii.gz" )
#' }
#' @export
brainTumorSegmentation <- function( flair, t1, t1Contrast, t2, 
  predictionBatchSize = 16, patchStrideLength = 32,
  doPreprocessing = TRUE, antsxnetCacheDirectory = NULL, verbose = FALSE )
{

  if( any( dim( t1 ) < c( 64, 64, 64 ) ) )
    {
    stop( "Images must be > 64 voxels per dimension." )
    }

  ################################
  #
  # Preprocess images
  #
  ################################

  flairPreprocessed <- NULL
  t1Preprocessed <- NULL
  t1ContrastPreprocessed <- NULL
  t2Preprocessed <- NULL

  if( doPreprocessing )
    {
    if( verbose )
      {
      message( "Preprocess FLAIR, T1, T1 contrast, and T2 images.\n" )
      }

    doBiasCorrection <- FALSE 

    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = NULL,
        brainExtractionModality = "t1",
        doBiasCorrection = doBiasCorrection,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    brainMask <- thresholdImage( t1Preprocessing$brainMask, 0.5, 1, 1, 0 )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * brainMask

    flairPreprocessing <- preprocessBrainImage( flair,
        truncateIntensity = NULL,
        brainExtractionModality = NULL,
        doBiasCorrection = doBiasCorrection,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    flairPreprocessed <- flairPreprocessing$preprocessedImage * brainMask

    t1ContrastPreprocessing <- preprocessBrainImage( t1Contrast,
        truncateIntensity = NULL,
        brainExtractionModality = NULL,
        doBiasCorrection = doBiasCorrection,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t1ContrastPreprocessed <- t1ContrastPreprocessing$preprocessedImage * brainMask

    t2Preprocessing <- preprocessBrainImage( t2,
        truncateIntensity = NULL,
        brainExtractionModality = NULL,
        doBiasCorrection = doBiasCorrection,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t2Preprocessed <- t2Preprocessing$preprocessedImage * brainMask

    } else {
    flairPreprocessed <- antsImageClone( flair )
    t1Preprocessed <- antsImageClone( t1 )
    t1ContrastPreprocessed <- antsImageClone( t1Contrast )
    t2Preprocessed <- antsImageClone( t2 )
    }

  images <- list()
  images[[1]] <- flairPreprocessed
  images[[2]] <- t1Preprocessed
  images[[3]] <- t1ContrastPreprocessed
  images[[4]] <- t2Preprocessed

  for( i in seq.int( length( images ) ) )
    {
    images[[i]] <- ( ( images[[i]] - min( images[[i]][brainMask > 0] ) ) /
                     ( max( images[[i]][brainMask > 0] ) - min( images[[i]][brainMask > 0] ) ) )
    }            

    ################################################################################################
    #
    #                        Stage 1:  Whole tumor segmentation
    #
    ################################################################################################

    ################################
    #
    # Build model and load weights
    #
    ################################

  if( verbose )
    {
    message( "Stage 1:  Load model and weights.\n" )
    }

  patchSize <- c( 64L, 64L, 64L )
  if( is.double( patchStrideLength ) || is.integer( patchStrideLength ) )
    {
    patchStrideLength <- rep( as.integer( patchStrideLength ), 3 )
    }
  numberOfFilters <- c( 64, 96, 128, 256, 512 )
  channelSize <- 4

  model <- createSysuMediaUnetModel3D( c( patchSize, channelSize ),
                                       numberOfFilters = numberOfFilters )
  weightsFileName <- getPretrainedNetwork( "bratsStage1", antsxnetCacheDirectory = antsxnetCacheDirectory )
  load_model_weights_hdf5( model, filepath = weightsFileName )

  ################################
  #
  # Extract patches
  #
  ################################

  if( verbose )
    {
    message( "Stage 1:  Extract patches." )
    }

  imagePatches <- list()
  for( i in seq.int( length( images ) ) )
    {
    imagePatches[[i]] <- extractImagePatches( images[[i]],
                                              patchSize = patchSize,
                                              maxNumberOfPatches = "all",
                                              strideLength = patchStrideLength,
                                              maskImage = brainMask,
                                              randomSeed = NULL,
                                              returnAsArray = TRUE )
    }
  totalNumberOfPatches <- dim( imagePatches[[1]] )[1]
 
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
    for( i in seq.int( length( imagePatches ) ) )
      {
      batchX[,,,,i] <- imagePatches[[i]][indices,,,]
      }

    if( verbose )
      {
      message( "  Predicting batch ", b, " of ", numberOfBatches )
      }
    prediction[indices,,,,] <- model %>% predict( batchX, verbose = verbose )
    }  

  if( verbose )
    {
    message( "Stage 1:  Predict patches and reconstruct." )
    }
  tumorProbabilityImage <- reconstructImageFromPatches( drop( prediction ),
                                                        strideLength = patchStrideLength,
                                                        domainImage = brainMask,
                                                        domainImageIsMask = TRUE )

  tumorMask <- thresholdImage( tumorProbabilityImage, 0.5, 1.0, 1, 0)

    ################################################################################################
    #
    #                        Stage 2:  Tumor component segmentation
    #
    ################################################################################################

    ################################
    #
    # Build model and load weights
    #
    ################################

  if( verbose )
    {
    message( "Stage 2:  Load model and weights.\n" )
    }

  patchSize <- c( 64L, 64L, 64L )
  numberOfFilters <- c( 64, 96, 128, 256, 512 )
  channelSize <- 5 # [FLAIR, T1, T1GD, T2, MASK]
  numberOfClassificationLabels <- 5

  model <- createUnetModel3D( c( patchSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = "sigmoid",
      numberOfFilters = c( 32, 64, 128, 256, 512 ),
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      dropoutRate = 0.0, weightDecay = 0 )

  weightsFileName <- getPretrainedNetwork( "bratsStage2", antsxnetCacheDirectory = antsxnetCacheDirectory )
  load_model_weights_hdf5( model, filepath = weightsFileName )

  ################################
  #
  # Extract patches
  #
  ################################

  if( verbose )
    {
    message( "Stage 2:  Extract patches." )
    }

  images[[5]] <- tumorMask 

  imagePatches <- list()
  for( i in seq.int( length( images ) ) )
    {
    imagePatches[[i]] <- extractImagePatches( images[[i]],
                                              patchSize = patchSize,
                                              maxNumberOfPatches = "all",
                                              strideLength = patchStrideLength,
                                              maskImage = tumorMask,
                                              randomSeed = NULL,
                                              returnAsArray = TRUE )
    }
  totalNumberOfPatches <- dim( imagePatches[[1]] )[1]
 
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
    for( i in seq.int( length( imagePatches ) ) )
      {
      batchX[,,,,i] <- imagePatches[[i]][indices,,,]
      }

    if( verbose )
      {
      message( "Predicting batch ", b, " of ", numberOfBatches )
      }
    prediction[indices,,,,] <- model %>% predict( batchX, verbose = verbose )
    }  

  if( verbose )
    {
    message( "Stage 2:  Predict patches and reconstruct." )
    }
  probabilityImages <- list()
  for( c in seq.int( numberOfClassificationLabels ) ) 
    {
    probabilityImages[[c]] <- reconstructImageFromPatches( drop( prediction[,,,,c] ),
                                                            strideLength = patchStrideLength,
                                                            domainImage = tumorMask,
                                                            domainImageIsMask = TRUE )
    }

  imageMatrix <- imageListToMatrix( probabilityImages, tumorMask * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]] - 1

  results <- list( segmentationImage = segmentationImage,
                   probabilityImages = probabilityImages )
  }



