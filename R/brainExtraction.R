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
#' \enumerate{
#'   \item "t1"
#'   \item "t1nobrainer"
#'   \item "bold"
#'   \item "fa"
#'   \item "t1t2infant"
#'   \item "t1infant"
#'   \item "t2infant"
#' }
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
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
  modality = c( "t1", "t1nobrainer", "bold", "fa", "t1t2infant", "t1infant", "t2infant" ), 
  outputDirectory = NULL, verbose = FALSE )
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

  modality <- match.arg( modality )

  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  if( modality != "t1nobrainer" )
    {

    #####################
    #
    # ANTs-based
    #
    ##################### 

    weightsFileName <- ''
    weightsFilePrefix <- ''
    if( modality == "t1" )
      {
      weightsFileName <- paste0( outputDirectory, "/brainExtractionWeights.h5" )
      weightsFilePrefix <- "brainExtraction"
      } else if( modality == "bold" ) {
      weightsFileName <- paste0( outputDirectory, "/brainExtractionBoldWeights.h5" )
      weightsFilePrefix <- "brainExtractionBOLD"
      } else if( modality == "fa" ) {
      weightsFileName <- paste0( outputDirectory, "/brainExtractionFaWeights.h5" )
      weightsFilePrefix <- "brainExtractionFA"
      } else if( modality == "t1t2infant" ) {
      weightsFileName <- paste0( outputDirectory, "/brainExtractionInfantT1T2Weights.h5" )
      weightsFilePrefix <- "brainExtractionInfantT1T2"
      } else if( modality == "t1infant" ) {
      weightsFileName <- paste0( outputDirectory, "/brainExtractionInfantT1Weights.h5" )
      weightsFilePrefix <- "brainExtractionInfantT1"
      } else if( modality == "t2infant" ) {
      weightsFileName <- paste0( outputDirectory, "/brainExtractionInfantT2Weights.h5" )
      weightsFilePrefix <- "brainExtractionInfantT2"
      } else {
      stop( "Unknown modality type." )  
      }

    if( ! file.exists( weightsFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "Brain extraction:  downloading model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( weightsFilePrefix, weightsFileName )
      }

    reorientTemplateFileName <- paste0( outputDirectory, "/S_template3_resampled.nii.gz" )
    if( ! file.exists( reorientTemplateFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "Brain extraction:  downloading template.\n" )
        }
      reorientTemplateUrl <- "https://ndownloader.figshare.com/files/22597175"
      download.file( reorientTemplateUrl, reorientTemplateFileName, quiet = !verbose )
      }
    reorientTemplate <- antsImageRead( reorientTemplateFileName )
    resampledImageSize <- dim( reorientTemplate )

    unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels,
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
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
      probabilityImagesArray[[1]][[2]], inputImages[[1]] )

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

    weightsFileName <- paste0( outputDirectory, "/noBrainerWeights.h5" )
    if( ! file.exists( weightsFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "NoBrainer:  downloading model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( "brainExtractionNoBrainer", weightsFileName )
      }
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
