#' Lung extraction
#'
#' Perform proton (H1) or CT lung extraction using a U-net architecture.  
#'
#' @param image input 3-D lung image.
#' @param modality image type.  Options include "proton" or "ct".
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return left/right probability masks (list of ANTsR image)
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "lung.nii.gz" )
#' output <- lungExtraction( image, modality = "proton" )
#' }
#' @export
lungExtraction <- function( image, 
  modality = c( "proton", "ct" ), 
  outputDirectory = NULL, verbose = FALSE )
  {

  if( image@dimension != 3 )
    {
    stop( "Image dimension must be 3." )  
    }

  modality <- match.arg( modality )

  imageModalities <- c( modality )
  channelSize <- length( imageModalities )

  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  weightsFileName <- ''
  unetModel <- NULL
  if( modality == "proton" )
    {
    weightsFileName <- paste0( outputDirectory, "/protonLungSegmentationWeights.h5" )
    if( ! file.exists( weightsFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "Lung extraction:  downloading model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( "protonLungMri", weightsFileName )
      }

    classes <- c( "Background", "LeftLung", "RightLung" )
    numberOfClassificationLabels <- length( classes )

    reorientTemplateFileName <- paste0( outputDirectory, "/protonLungTemplate.nii.gz" )
    if( ! file.exists( reorientTemplateFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "Lung extraction:  downloading template.\n" )
        }
      reorientTemplateUrl <- "https://ndownloader.figshare.com/files/22707338"
      download.file( reorientTemplateUrl, reorientTemplateFileName, quiet = !verbose )
      }
    reorientTemplate <- antsImageRead( reorientTemplateFileName )
    resampledImageSize <- dim( reorientTemplate )

    unetModel <- createUnetModel3D( c( resampledImageSize, 1 ),
      numberOfOutputs = numberOfClassificationLabels,
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
      convolutionKernelSize = c( 7, 7, 5 ), deconvolutionKernelSize = c( 7, 7, 5 ) )
    unetModel$load_weights( weightsFileName )

    if( verbose == TRUE )
      {
      cat( "Lung extraction:  normalizing image to the template.\n" )
      }
    reorientTemplateOnes <- antsImageClone( reorientTemplate ) ^ 0
    centerOfMassTemplate <- getCenterOfMass( reorientTemplateOnes )
    imageOnes <- antsImageClone( image ) ^ 0
    centerOfMassImage <- getCenterOfMass( imageOnes )
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )
    warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
    warpedArray <- as.array( warpedImage )
    warpedArray[which( warpedArray < -1000 )] <- -1000

    batchX <- array( data = as.array( warpedImage ),
      dim = c( 1, resampledImageSize, channelSize ) )
    batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

    if( verbose == TRUE )
      {
      cat( "Lung extraction:  prediction and decoding.\n" )
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

    if( verbose == TRUE )
      {
      cat( "Lung extraction:  renormalize probability mask to native space.\n" )
      }

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels - 1 ) )
      {
      probabilityImageTmp <- probabilityImagesArray[[1]][[i+1]]
      probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImageTmp, image )
      }
    return( list( leftLung = probabilityImages[[1]],
                  rightLung = probabilityImages[[2]] ) )

    } else if( modality == "ct" ) {
    weightsFileName <- paste0( outputDirectory, "/ctLungSegmentationWeights.h5" )
    if( ! file.exists( weightsFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "Lung extraction:  downloading model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( "ctHumanLung", weightsFileName )
      }

    classes <- c( "Background", "LeftLung", "RightLung", "Trachea" )
    numberOfClassificationLabels <- length( classes )

    reorientTemplateFileName <- paste0( outputDirectory, "/ctLungTemplate.nii.gz" )
    if( ! file.exists( reorientTemplateFileName ) )
      {
      if( verbose == TRUE )
        {
        cat( "Lung extraction:  downloading template.\n" )
        }
      reorientTemplateUrl <- "https://ndownloader.figshare.com/files/22707335"
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
      cat( "Lung extraction:  normalizing image to the template.\n" )
      }
    reorientTemplateOnes <- antsImageClone( reorientTemplate ) ^ 0
    centerOfMassTemplate <- getCenterOfMass( reorientTemplateOnes )
    imageOnes <- antsImageClone( image ) ^ 0
    centerOfMassImage <- getCenterOfMass( imageOnes )
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )
    warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
    warpedArray <- as.array( warpedImage )
    warpedArray[which( warpedArray < -1000 )] <- -1000

    batchX <- array( data = as.array( warpedImage ),
      dim = c( 1, resampledImageSize, channelSize ) )
    batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

    if( verbose == TRUE )
      {
      cat( "Lung extraction:  prediction and decoding.\n" )
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

    if( verbose == TRUE )
      {
      cat( "Lung extraction:  renormalize probability mask to native space.\n" )
      }

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels - 1 ) )
      {
      probabilityImageTmp <- probabilityImagesArray[[1]][[i+1]]
      probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImageTmp, image )
      }

    return( list( leftLung = probabilityImages[[1]],
                  rightLung = probabilityImages[[2]],
                  trachea = probabilityImages[[3]] ) )

    } else {
    stop( "Unknown modality type." )  
    }

  }