#' Brain extraction
#'
#' Perform T1, FA, or bold brain extraction using a U-net architecture
#' training data.  "NoBrainer" is also possible where
#' brain extraction uses U-net and FreeSurfer
#' training data ported from the
#'
#'  \url{https://github.com/neuronets/nobrainer-models}
#'
#' @param image input 3-D brain image (or list of images for multi-modal scenarios).
#' @param modality image type.  Options include:
#' \itemize{
#'   \item{"t1": }{T1-weighted MRI---ANTs-trained.  Update from "t1v0"}
#'   \item{"t1v0": }{T1-weighted MRI---ANTs-trained.}
#'   \item{"t1nobrainer": }{T1-weighted MRI---FreeSurfer-trained: h/t Satra Ghosh and Jakub Kaczmarzyk.}
#'   \item{"t1combined": }{Brian's combination of "t1" and "t1nobrainer".  One can also specify
#'                         "t1combined[X]" where X is the morphological radius.  X = 12 by default.}
#'   \item{"flair": }{FLAIR MRI.}
#'   \item{"t2": }{T2-w MRI.}
#'   \item{"bold": }{3-D BOLD MRI.}
#'   \item{"fa": }{Fractional anisotropy.}
#'   \item{"t1t2infant": }{Combined T1-w/T2-w infant MRI h/t Martin Styner.}
#'   \item{"t1infant": }{T1-w infant MRI h/t Martin Styner.}
#'   \item{"t2infant": }{T2-w infant MRI h/t Martin Styner.}
#' }
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return brain probability mask (ANTsR image)
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' probabilityMask <- brainExtraction( image, modality = "t1" )
#' }
#' @export
brainExtraction <- function( image,
  modality = c( "t1", "t1v0", "t1nobrainer", "t1combined", "t2", "flair", "bold", "fa", "t1t2infant", "t1infant", "t2infant" ),
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  classes <- c( "background", "brain" )
  numberOfClassificationLabels <- length( classes )
  channelSize <- length( image )

  inputImages <- list()
  if( channelSize == 1 )
    {
    inputImages[[1]] <- image
    } else {
    inputImages <- image
    }

  if( inputImages[[1]]@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  if( substr( modality, 1, 10 ) == "t1combined" )
    {

    # Need to change with voxel resolution
    morphologicalRadius <- 12
    if( grepl( '\\[', modality ) && grepl( '\\]', modality ) )
      {
      morphologicalRadius <- as.numeric( strsplit( strsplit( modality, "\\[" )[[1]][2], "\\]" )[[1]][1] )
      }

    brainExtraction_t1 <- brainExtraction( image, modality = "t1",
      antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )
    brainMask <- thresholdImage( brainExtraction_t1, 0.5, Inf ) %>%
      morphology("close",morphologicalRadius) %>%
      iMath("FillHoles") %>%
      iMath( "GetLargestComponent" )

    brainExtraction_t1nobrainer <- brainExtraction( image * iMath( brainMask, "MD", morphologicalRadius ),
      modality = "t1nobrainer", antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )
    brainExtraction_combined <- iMath( brainExtraction_t1nobrainer * brainMask, "GetLargestComponent" ) %>% iMath( "FillHoles" )

    brainExtraction_combined <- brainExtraction_combined + iMath( brainMask, "ME", morphologicalRadius ) + brainMask

    return( brainExtraction_combined )
    }

  if( modality != "t1nobrainer" )
    {

    #####################
    #
    # ANTs-based
    #
    #####################

    weightsFilePrefix <- ''
    if( modality == "t1v0" )
      {
      weightsFilePrefix <- "brainExtraction"
      } else if( modality == "t1" ) {
      weightsFilePrefix <- "brainExtractionT1v1"
      } else if( modality == "t2" ) {
      weightsFilePrefix <- "brainExtractionT2"
      } else if( modality == "flair" ) {
      weightsFilePrefix <- "brainExtractionFLAIR"
      } else if( modality == "bold" ) {
      weightsFilePrefix <- "brainExtractionBOLD"
      } else if( modality == "fa" ) {
      weightsFilePrefix <- "brainExtractionFA"
      } else if( modality == "t1t2infant" ) {
      weightsFilePrefix <- "brainExtractionInfantT1T2"
      } else if( modality == "t1infant" ) {
      weightsFilePrefix <- "brainExtractionInfantT1"
      } else if( modality == "t2infant" ) {
      weightsFilePrefix <- "brainExtractionInfantT2"
      } else {
      stop( "Unknown modality type." )
      }

    if( verbose == TRUE )
      {
      cat( "Brain extraction:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( weightsFilePrefix, antsxnetCacheDirectory = antsxnetCacheDirectory )

    reorientTemplateFileName <- "S_template3_resampled.nii.gz"
    if( verbose == TRUE )
      {
      cat( "Brain extraction:  retrieving template.\n" )
      }
    reorientTemplateFileNamePath <- getANTsXNetData( "S_template3",
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    reorientTemplate <- antsImageRead( reorientTemplateFileNamePath )
    resampledImageSize <- dim( reorientTemplate )

    numberOfFilters <- c( 8, 16, 32, 64 )
    mode <- "classification"
    if( modality == "t1" )
      {
      numberOfFilters <- c( 16, 32, 64, 128 )
      numberOfClassificationLabels <- 1
      mode <- "sigmoid"
      }

    unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = mode,
      numberOfFilters = numberOfFilters, dropoutRate = 0.0,
      convolutionKernelSize = 3, deconvolutionKernelSize = 2,
      weightDecay = 1e-5 )

    unetModel$load_weights( weightsFileName )

    if( verbose == TRUE )
      {
      cat( "Brain extraction:  normalizing image to the template.\n" )
      }

    centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
    centerOfMassImage <- getCenterOfMass( inputImages[[1]] )
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )

    batchX <- array( data = 0, dim = c( 1, resampledImageSize, channelSize ) )

    for( i in seq.int( channelSize ) )
      {
      warpedImage <- applyAntsrTransformToImage( xfrm, inputImages[[i]], reorientTemplate )
      warpedArray <- as.array( warpedImage )
      batchX[1,,,,i] <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )
      }

    if( verbose == TRUE )
      {
      cat( "Brain extraction:  prediction and decoding.\n" )
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

    if( verbose == TRUE )
      {
      cat( "Brain extraction:  renormalize probability mask to native space.\n" )
      }
    probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      probabilityImagesArray[[1]][[numberOfClassificationLabels]], inputImages[[1]] )

    return( probabilityImage )
    } else {

    #####################
    #
    # NoBrainer
    #
    #####################

    if( verbose == TRUE )
      {
      cat( "NoBrainer:  generating network.\n")
      }
    model <- createNoBrainerUnetModel3D( list( NULL, NULL, NULL, 1 ) )

    if( verbose == TRUE )
      {
      cat( "NoBrainer:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "brainExtractionNoBrainer",
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    model$load_weights( weightsFileName )

    if( verbose == TRUE )
      {
      cat( "NoBrainer:  preprocessing (intensity truncation and resampling).\n" )
      }
    imageArray <- as.array( image )
    imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
    thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
    thresholdedMask <- thresholdImage( image, -10000, thresholdValue, 0, 1 )
    thresholdedImage <- image * thresholdedMask

    imageResampled <- resampleImage( image, rep( 256, 3 ), useVoxels = TRUE )
    imageArray <- array( as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )

    if( verbose == TRUE )
      {
      cat( "NoBrainer:  predicting mask.\n" )
      }
    brainMaskArray <- predict( model, imageArray )
    brainMaskResampled <- as.antsImage( brainMaskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
    brainMaskImage = resampleImage( brainMaskResampled, dim( image ),
      useVoxels = TRUE, interpType = "nearestneighbor" )
    minimumBrainVolume <- round( 649933.7 / prod( antsGetSpacing( image ) ) )
    brainMaskLabeled = labelClusters( brainMaskImage, minimumBrainVolume )

    return( brainMaskLabeled )
    }
  }
