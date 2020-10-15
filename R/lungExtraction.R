#' Lung extraction
#'
#' Perform proton (H1) or CT lung extraction using a U-net architecture.
#'
#' @param image input 3-D lung image.
#' @param modality image type.  Options include "proton" or "ct".
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return segmentation and probability images
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
    outputDirectory <- "ANTsXNet"
    }

  if( modality == "proton" )
    {
    if( verbose == TRUE )
      {
      cat( "Lung extraction:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "protonLungMri", outputDirectory = outputDirectory )

    classes <- c( "Background", "LeftLung", "RightLung" )
    numberOfClassificationLabels <- length( classes )

    reorientTemplateFileName <- "protonLungTemplate.nii.gz"
    if( verbose == TRUE )
      {
      cat( "Lung extraction:  retrieving template.\n" )
      }
    reorientTemplateUrl <- "https://ndownloader.figshare.com/files/22707338"
    reorientTemplateFileNamePath <- tensorflow::tf$keras$utils$get_file(
      reorientTemplateFileName, reorientTemplateUrl, cache_subdir = outputDirectory )
    reorientTemplate <- antsImageRead( reorientTemplateFileNamePath )
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
    centerOfMassTemplate <- getCenterOfMass( reorientTemplate * 0 + 1 )
    centerOfMassImage <- getCenterOfMass( image  * 0 + 1 )
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
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      probabilityImageTmp <- probabilityImagesArray[[1]][[i]]
      probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImageTmp, image )
      }

    imageMatrix <- imageListToMatrix( probabilityImages, image * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) - 1
    segmentationImage <- matrixToImages( segmentationMatrix, image * 0 + 1 )[[1]]

    return( list( segmentationImage = segmentationImage,
                  probabilityImages = probabilityImages ) )

    } else if( modality == "ct" ) {
    if( verbose == TRUE )
      {
      cat( "Lung extraction:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "ctHumanLung", outputDirectory = outputDirectory )

    classes <- c( "Background", "LeftLung", "RightLung", "Trachea" )
    numberOfClassificationLabels <- length( classes )

    reorientTemplateFileName <- "ctLungTemplate.nii.gz"
    if( verbose == TRUE )
      {
      cat( "Lung extraction:  retrieving template.\n" )
      }
    reorientTemplateUrl <- "https://ndownloader.figshare.com/files/22707335"
    reorientTemplateFileNamePath <- tensorflow::tf$keras$utils$get_file(
      reorientTemplateFileName, reorientTemplateUrl, cache_subdir = outputDirectory )
    reorientTemplate <- antsImageRead( reorientTemplateFileNamePath )
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
    centerOfMassTemplate <- getCenterOfMass( reorientTemplate * 0 + 1 )
    centerOfMassImage <- getCenterOfMass( image  * 0 + 1 )
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
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      probabilityImageTmp <- probabilityImagesArray[[1]][[i+1]]
      probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImageTmp, image )
      }
    imageMatrix <- imageListToMatrix( probabilityImages, image * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) - 1
    segmentationImage <- matrixToImages( segmentationMatrix, image * 0 + 1 )[[1]]

    return( list( segmentationImage = segmentationImage,
                  probabilityImages = probabilityImages ) )

    } else {
    stop( "Unknown modality type." )
    }

  }