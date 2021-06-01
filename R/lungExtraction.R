#' Lung extraction
#'
#' Perform proton (H1) or CT lung extraction using a U-net architecture.
#'
#' @param image input 3-D lung image.
#' @param modality image type.  Options include "proton" or "ct".
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
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
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( image@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  modality <- match.arg( modality )

  imageModalities <- c( modality )
  channelSize <- length( imageModalities )

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  if( modality == "proton" )
    {
    if( verbose == TRUE )
      {
      cat( "Lung extraction:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "protonLungMri", antsxnetCacheDirectory = antsxnetCacheDirectory )

    classes <- c( "Background", "LeftLung", "RightLung" )
    numberOfClassificationLabels <- length( classes )

    reorientTemplateFileName <- "protonLungTemplate.nii.gz"
    if( verbose == TRUE )
      {
      cat( "Lung extraction:  retrieving template.\n" )
      }
    reorientTemplateFileNamePath <- getANTsXNetData( "protonLungTemplate",
      antsxnetCacheDirectory = antsxnetCacheDirectory)
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

    # } else if( modality == "ctOld" ) {
    # if( verbose == TRUE )
    #   {
    #   cat( "Lung extraction:  retrieving model weights.\n" )
    #   }
    # weightsFileName <- getPretrainedNetwork( "ctHumanLung",
    #   antsxnetCacheDirectory = antsxnetCacheDirectory )

    # classes <- c( "Background", "LeftLung", "RightLung", "Trachea" )
    # numberOfClassificationLabels <- length( classes )

    # if( verbose == TRUE )
    #   {
    #   cat( "Lung extraction:  retrieving template.\n" )
    #   }
    # reorientTemplateFileNamePath <- getANTsXNetData( "ctLungTemplate",
    #   antsxnetCacheDirectory = antsxnetCacheDirectory )
    # reorientTemplate <- antsImageRead( reorientTemplateFileNamePath )
    # resampledImageSize <- dim( reorientTemplate )

    # unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
    #   numberOfOutputs = numberOfClassificationLabels,
    #   numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
    #   convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    #   weightDecay = 1e-5 )
    # unetModel$load_weights( weightsFileName )

    # if( verbose == TRUE )
    #   {
    #   cat( "Lung extraction:  normalizing image to the template.\n" )
    #   }
    # centerOfMassTemplate <- getCenterOfMass( reorientTemplate * 0 + 1 )
    # centerOfMassImage <- getCenterOfMass( image  * 0 + 1 )
    # xfrm <- createAntsrTransform( type = "Euler3DTransform",
    #   center = centerOfMassTemplate,
    #   translation = centerOfMassImage - centerOfMassTemplate )
    # warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
    # warpedArray <- as.array( warpedImage )
    # warpedArray[which( warpedArray < -1000 )] <- -1000

    # batchX <- array( data = as.array( warpedImage ),
    #   dim = c( 1, resampledImageSize, channelSize ) )
    # batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

    # if( verbose == TRUE )
    #   {
    #   cat( "Lung extraction:  prediction and decoding.\n" )
    #   }
    # predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    # probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

    # if( verbose == TRUE )
    #   {
    #   cat( "Lung extraction:  renormalize probability mask to native space.\n" )
    #   }

    # probabilityImages <- list()
    # for( i in seq_len( numberOfClassificationLabels ) )
    #   {
    #   probabilityImageTmp <- probabilityImagesArray[[1]][[i]]
    #   probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    #     probabilityImageTmp, image )
    #   }
    # imageMatrix <- imageListToMatrix( probabilityImages, image * 0 + 1 )
    # segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) - 1
    # segmentationImage <- matrixToImages( segmentationMatrix, image * 0 + 1 )[[1]]

    # return( list( segmentationImage = segmentationImage,
    #               probabilityImages = probabilityImages ) )

    } else if( modality == "ct" ) {

    ################################
    #
    # Preprocess image
    #
    ################################

    if( verbose == TRUE )
      {
      cat("Preprocess CT image.\n")
      }

    closestSimplifiedDirectionMatrix <- function( direction )
      {
      closest = floor( abs( direction + 0.5 ) )
      closest[direction < 0] <- closest[direction < 0] * -1.0
      return( direction )
      }

    simplifiedDirection <- closestSimplifiedDirectionMatrix( antsGetDirection( image ) )

    referenceImageSize <- c( 200, 200, 200 )

    ctPreprocessed <- resampleImage( image, referenceImageSize, useVoxels = TRUE, interpType = 0 )
    ctPreprocessed[ctPreprocessed < -1000] <- -1000
    ctPreprocessed[ctPreprocessed > 400] <- 400
    antsSetDirection( ctPreprocessed, simplifiedDirection )
    antsSetOrigin( ctPreprocessed, c( 0, 0, 0 ) )
    antsSetSpacing( ctPreprocessed, c( 1, 1, 1 ) )

    ################################
    #
    # Reorient image
    #
    ################################

    referenceImage <- makeImage(referenceImageSize, voxval = 0, spacing = c( 1, 1, 1 ),
      origin = c( 0, 0, 0 ), direction = diag( 3 ) )

    centerOfMassReference <- floor( getCenterOfMass( referenceImage * 0 + 1 ) )
    centerOfMassImage <- floor( getCenterOfMass( ctPreprocessed * 0 + 1 ) )
    translation <- centerOfMassImage - centerOfMassReference
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassReference, translation = translation )
    ctPreprocessed <- ( ( ctPreprocessed - min( ctPreprocessed ) ) /
        ( max( ctPreprocessed ) - min( ctPreprocessed ) ) ) - 0.5
    ctPreprocessedWarped = applyAntsrTransformToImage(
        xfrm, ctPreprocessed, referenceImage, interpolation = "nearestneighbor" )

    ################################
    #
    # Build models and load weights
    #
    ################################

    if( verbose == TRUE )
      {
      cat( "Build model and load weights.\n" )
      }

    patchSize <- c( 128L, 128L, 128L )
    strideLength <- dim( ctPreprocessed ) - patchSize

    weightsFileName <- getPretrainedNetwork( "lungCtOctantWithPriorsSegmentationWeights", antsxnetCacheDirectory = antsxnetCacheDirectory )

    classes <- c( "background", "left lung", "right lung", "airways" )
    numberOfClassificationLabels <- length( classes )

    luna16Priors <- splitNDImageToList( antsImageRead( getANTsXNetData( "luna16LungPriors" ) ) )
    for( i in seq.int( length( luna16Priors ) ) )
      {
      luna16Priors[[i]] <- antsCopyImageInfo( ctPreprocessedWarped, luna16Priors[[i]] )
      }
    channelSize <- length( luna16Priors ) + 1

    unetModel <- createUnetModel3D( c( patchSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 1e-5 )

    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose == TRUE )
      {
      cat( "Prediction.\n" )
      }

    imagePatches <- extractImagePatches( ctPreprocessedWarped, patchSize, maxNumberOfPatches = "all",
                                        strideLength = strideLength, returnAsArray = TRUE )
    batchX <- array( data = 0, dim = c( dim( imagePatches ), channelSize ) )
    batchX[,,,,1] <- imagePatches
    for( i in seq.int( length( luna16Priors ) ) )
      {
      priorPatches <- extractImagePatches( luna16Priors[[i]], patchSize, maxNumberOfPatches = "all",
                        strideLength = strideLength, returnAsArray = TRUE )
      batchX[,,,,i+1] <- priorPatches
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      if( verbose == TRUE )
        cat( "Reconstructing image", classes[i], "\n" )
      reconstructedImage <- reconstructImageFromPatches( predictedData[,,,,i],
          domainImage = ctPreprocessedWarped, strideLength = strideLength )
      probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        reconstructedImage, ctPreprocessed )
      probabilityImage <- resampleImage( probabilityImage, resampleParams = dim( image ),
        useVoxels = TRUE, interpType = 0 )
      probabilityImage <- antsCopyImageInfo( image, probabilityImage )
      probabilityImages[[i]] <- probabilityImage
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