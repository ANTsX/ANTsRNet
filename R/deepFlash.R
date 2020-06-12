#' Hippocampal/Enthorhinal segmentation using "Deep Flash"
#'
#' Perform hippocampal/entorhinal segmentation in T1 images using
#' labels from Mike Yassa's lab
#'
#' \url{https://faculty.sites.uci.edu/myassa/}
#'
#' The labeling is as follows:
#'   Label 0 :  background
#'   Label 5 :  left aLEC
#'   Label 6 :  right aLEC
#'   Label 7 :  left pMEC
#'   Label 8 :  right pMEC
#'   Label 9 :  left perirhinal
#'   Label 10:  right perirhinal
#'   Label 11:  left parahippocampal
#'   Label 12:  right parahippocampal
#'   Label 13:  left DG/CA3
#'   Label 14:  right DG/CA3
#'   Label 15:  left CA1
#'   Label 16:  right CA1
#'   Label 17:  left subiculum
#'   Label 18:  right subiculum
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
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
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
#' results <- deepFlash( image )
#' }
#' @export
deepFlash <- function( t1, doPreprocessing = TRUE,
  outputDirectory = NULL, verbose = FALSE, debug = FALSE )
{

  padOrCropImageToSize <- function( image, size )
    {
    imageSize <- dim( image )
    delta <- imageSize - size

    if( any( delta < 0 ) )
      {
      padSize <- abs( min( delta ) )
      image <- iMath( image, "PadImage", padSize )
      }
    croppedImage <- cropImageCenter( image, size )
    return( croppedImage )
    }

  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
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
        doBrainExtraction = TRUE,
        template = "croppedMni152",
        templateTransformType = "AffineFast",
        doBiasCorrection = TRUE,
        doDenoising = TRUE,
        outputDirectory = outputDirectory,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    }

  ################################
  #
  # Build model and load weights
  #
  ################################

  templateSize <- c( 160L, 192L, 160L )
  labels <- c( 0, 5:18 )

  unetModel <- createUnetModel3D( c( templateSize, 1 ),
    numberOfOutputs = length( labels ), mode = 'classification',
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5, addAttentionGating = TRUE )

  weightsFileName <- paste0( outputDirectory, "deepFlashWeights.h5" )
  if( ! file.exists( weightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "DeepFlash:  downloading model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "deepFlash", weightsFileName )
    }
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )

  unetModel %>% compile(
    optimizer = optimizer_adam(),
    loss = categorical_focal_loss( alpha = 0.25, gamma = 2.0 ),
    metrics = 'accuracy' )

  ################################
  #
  # Do prediction and normalize to native space
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Prediction.\n" )
    }

  croppedImage <- padOrCropImageToSize( t1Preprocessed, templateSize )
  imageArray <- as.array( croppedImage )

  batchX <- array( data = imageArray, dim = c( 1, templateSize, 1 ) )
  batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  probabilityImagesList <- decodeUnet( predictedData, croppedImage )

  probabilityImages <- list()
  for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
    {
    if( i > 1 )
      {
      decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 )
      } else {
      decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 + 1 )
      }
    if( doPreprocessing == TRUE )
      {
      probabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      } else {
      probabilityImages[[i]] <- decroppedImage
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

