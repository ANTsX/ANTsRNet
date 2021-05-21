#' Six tissue segmentation
#'
#' Perform Atropos-style six tissue segmentation using deep learning
#'
#' The labeling is as follows:
#' \itemize{
#'   \item{Label 0:}{background}
#'   \item{Label 1:}{CSF}
#'   \item{Label 2:}{gray matter}
#'   \item{Label 3:}{white matter}
#'   \item{Label 4:}{deep gray matter}
#'   \item{Label 5:}{brain stem}
#'   \item{Label 6:}{cerebellum}
#' }
#'
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * denoising,
#'    * brain extraction, and
#'    * affine registration to MNI.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param t1 raw or preprocessed 3-D T1-weighted brain image.
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param useSpatialPriors use MNI spatial tissue priors (0, 1, or 2).  0 is no priors.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @param debug return feature images in the last layer of the u-net model.
#' @return list consisting of the segmentation image and probability images for
#' each label.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' results <- deepAtropos( image )
#' }
#' @export
deepAtropos <- function( t1, doPreprocessing = TRUE, useSpatialPriors = 0,
  antsxnetCacheDirectory = NULL, verbose = FALSE, debug = FALSE )
{

  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  ################################
  #
  # Preprocess image
  #
  ################################

  t1Preprocessed <- t1
  if( doPreprocessing == TRUE )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.01, 0.99 ),
        brainExtractionModality = "t1",
        template = "croppedMni152",
        templateTransformType = "antsRegistrationSyNQuickRepro[a]",
        doBiasCorrection = TRUE,
        doDenoising = TRUE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    }

  ################################
  #
  # Build model and load weights
  #
  ################################

  patchSize <- c( 112L, 112L, 112L )
  strideLength <- dim( t1Preprocessed ) - patchSize

  classes <- c( "background", "csf", "gray matter", "white matter",
    "deep gray matter", "brain stem", "cerebellum" )
  labels <- c( 0:6 )

  mniPriors <- NULL
  channelSize <- 1
  if( useSpatialPriors != 0 )
    {
    mniPriors <- splitNDImageToList( antsImageRead( getANTsXNetData( "croppedMni152Priors" ) ) )
    for( i in seq.int( length( mniPriors ) ) )
      {
      mniPriors[[i]] <- antsCopyImageInfo( t1Preprocessed, mniPriors[[i]] )
      }
    # channelSize <- length( mniPriors ) + 1
    channelSize <- 2  # T1 and cerebellum
    }

  unetModel <- createUnetModel3D( c( patchSize, channelSize ),
    numberOfOutputs = length( labels ), mode = 'classification',
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

  if( verbose == TRUE )
    {
    cat( "DeepAtropos:  retrieving model weights.\n" )
    }
  weightsFileName <- ''
  if( useSpatialPriors == 0 )
    {
    weightsFileName <- getPretrainedNetwork( "sixTissueOctantBrainSegmentation", antsxnetCacheDirectory = antsxnetCacheDirectory )
    } else if( useSpatialPriors == 1 ) {
    weightsFileName <- getPretrainedNetwork( "sixTissueOctantBrainSegmentationWithPriors1", antsxnetCacheDirectory = antsxnetCacheDirectory )
    } else if( useSpatialPriors == 2 ) {
    weightsFileName <- getPretrainedNetwork( "sixTissueOctantBrainSegmentationWithPriors2", antsxnetCacheDirectory = antsxnetCacheDirectory )
    }
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )

  ################################
  #
  # Do prediction and normalize to native space
  #
  ################################

  if( verbose == TRUE )
    {
    message( "Prediction.\n" )
    }

  t1Preprocessed <- ( t1Preprocessed - mean( t1Preprocessed ) ) / sd( t1Preprocessed )
  imagePatches <- extractImagePatches( t1Preprocessed, patchSize, maxNumberOfPatches = "all",
                                       strideLength = strideLength, returnAsArray = TRUE )
  batchX <- array( data = 0, dim = c( dim( imagePatches ), channelSize ) )
  batchX[,,,,1] <- imagePatches
  if( channelSize > 1 )
    {
    # for( i in seq.int( 1, channelSize-1 ) )
    #   {
      priorPatches <- extractImagePatches( mniPriors[[7]], patchSize, maxNumberOfPatches = "all",
                        strideLength = strideLength, returnAsArray = TRUE )
      batchX[,,,,2] <- priorPatches
      # }
    }
  predictedData <- unetModel %>% predict( batchX, verbose = verbose )

  probabilityImages <- list()
  for( i in seq.int( dim( predictedData )[5] ) )
    {
    message( "Reconstructing image ", classes[i], "\n" )
    reconstructedImage <- reconstructImageFromPatches( predictedData[,,,,i],
        domainImage = t1Preprocessed, strideLength = strideLength )
    if( doPreprocessing == TRUE )
      {
      probabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = reconstructedImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      } else {
      probabilityImages[[i]] <- reconstructedImage
      }
    }

  imageMatrix <- imageListToMatrix( probabilityImages, t1 * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]]

  relabeledImage <- antsImageClone( segmentationImage )

  for( i in seq.int( length( labels ) ) )
    {
    relabeledImage[( segmentationImage == i )] <- labels[i]
    }

  results <- list( segmentationImage = relabeledImage, probabilityImages = probabilityImages )

  # debugging

  if( debug == TRUE )
    {
    inputImage <- unetModel$input
    featureLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]
    featureFunction <- keras::backend()$`function`( list( inputImage ), list( featureLayer$output ) )
    featureBatch <- featureFunction( list( batchX[1,,,,,drop = FALSE] ) )

    featureImagesList <- decodeUnet( featureBatch[[1]], croppedImage )

    featureImages <- list()
    for( i in seq.int( length( featureImagesList[[1]] ) ) )
      {
      decroppedImage <- decropImage( featureImagesList[[1]][[i]], t1Preprocessed * 0 )
      if( doPreprocessing == TRUE )
        {
        featureImages[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        } else {
        featureImages[[i]] <- decroppedImage
        }
      }
    results[['featureImagesLastLayer']] <- featureImages
    }

  return( results )
}

