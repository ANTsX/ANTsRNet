#' wholeHeadInpainting
#'
#' Perform in-painting for whole-head MRI
#'
#' @param t1 input 3-D T1 brain image (not skull-stripped).
#' @param doPreprocessing  Perform n4 bias correction, intensity truncation, brain 
#' extraction.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return inpainted image
#' @author Tustison NJ
#' @export
lesionSegmentation <- function( t1, doPreprocessing = TRUE,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  ################################
  #
  # Preprocess images
  #
  ################################

  t1Preprocessed <- NULL
  brainMask <- NULL

  if( doPreprocessing )
    {
    if( verbose )
      {
      message( "Preprocess T1 image.\n" )
      }
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = NULL,
        brainExtractionModality = "t1",
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    brainMask <- t1Preprocessing$brainMask
    t1Preprocessed <- t1Preprocessing$preprocessedImage * brainMask
    } else {
    t1Preprocessed <- antsImageClone( t1 )
    brainMask <- thresholdImage( t1Preprocessed, 0, 0, 0, 1 )
    }
   
  templateSize <- c( 192, 208, 192 ) 
  template <- antsImageRead( getANTsXNetData( "mni152" ) )
  template <- padOrCropImageToSize( template, templateSize )
  templateMask <- brainExtraction( template, modality = "t1", verbose = verbose )
  template <- template * templateMask 
   
  if( verbose )
    {
    cat( "Load u-net models and weights." )
    }

   numberOfClassificationLabels = 1
   channelSize = 1
   unetWeightsFileName <- getPretrainedNetwork( "lesion_whole_brain", 
                                                antsxnetCacheDirectory = antsxnetCacheDirectory )
   unetModel <- createUnetModel3D( c( templateSize, channelSize ),
       numberOfOutputs = numberOfClassificationLabels,
       mode = 'sigmoid',
       numberOfFilters = c(16, 32, 64, 128, 256 ), dropoutRate = 0.0,
       convolutionKernelSize = 3, deconvolutionKernelSize = 2,
       weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )
   unetModel$load_weights( unetWeightsFileName )

  if( verbose )
    {
    cat( "Alignment to template." )
    }

  imageMin <- min( t1Preprocessed[brainMask != 0] )
  imageMax <- max( t1Preprocessed[brainMask != 0] )

  registration <- antsRegistration( template, t1Preprocessed, typeofTransform = "antsRegistrationSyNQuick[a]", 
                                    verbose = verbose )
  image <- registration$warpedmovout
  image <- ( image - imageMin ) / ( imageMax - imageMin )

  batchX <- array( data = 0, c( 1, dim( image ), channelSize ) )
  batchX[1,,,,1] <- as.array( image )

  lesionMaskArray <- drop( unetModel$predict( batchX, verbose = verbose ) )
  lesionMask <- antsCopyImageInfo( template, as.antsImage( lesionMaskArray ) )

  probabilityImage <- antsApplyTransforms( t1Preprocessed, lesionMask, registration$invtransforms, 
                                           whichtoinvert = c( TRUE ), verbose = verbose )
  
  return( probabilityImage )
  }