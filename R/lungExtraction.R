#' Lung extraction
#'
#' Perform lung extraction.
#'
#' @param image input 3-D lung image.
#' @param modality image type.  Options include "proton", "ct", or "ventilation".
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
#' @import keras
#' @export
lungExtraction <- function( image,
  modality = c( "proton", "protonLobes", "maskLobes", "ct", "ventilation", "xray" ),
  verbose = FALSE )
  {

  if( image@dimension != 3 && modality != "xray" )
    {
    stop( "Image dimension must be 3." )
    } else if( image@dimension != 2 && modality == "xray" ) {
    stop( "Image dimension must be 2." )
    }

  modality <- match.arg( modality )

  imageModalities <- c( modality )
  channelSize <- length( imageModalities )

  if( modality == "proton" )
    {
    weightsFileName <- getPretrainedNetwork( "protonLungMri" )

    classes <- c( "Background", "LeftLung", "RightLung" )
    numberOfClassificationLabels <- length( classes )

    reorientTemplateFileName <- "protonLungTemplate.nii.gz"
    if( verbose )
      {
      cat( "Lung extraction:  retrieving template.\n" )
      }
    reorientTemplateFileNamePath <- getANTsXNetData( "protonLungTemplate" )
    reorientTemplate <- antsImageRead( reorientTemplateFileNamePath )
    resampledImageSize <- dim( reorientTemplate )

    unetModel <- createUnetModel3D( c( resampledImageSize, 1 ),
      numberOfOutputs = numberOfClassificationLabels,
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
      convolutionKernelSize = c( 7, 7, 5 ), deconvolutionKernelSize = c( 7, 7, 5 ) )
    unetModel$load_weights( weightsFileName )

    if( verbose )
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

    if( verbose )
      {
      cat( "Lung extraction:  prediction and decoding.\n" )
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    predictedData <- array( predictedData[1,,,,], dim = c( dim( reorientTemplate ), numberOfClassificationLabels ) )
    probabilityImagesList <- oneHotToSegmentation( predictedData, reorientTemplate )

    if( verbose )
      {
      cat( "Lung extraction:  renormalize probability mask to native space.\n" )
      }

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      probabilityImageTmp <- probabilityImagesList[[i]]
      probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImageTmp, image )
      }

    imageMatrix <- imageListToMatrix( probabilityImages, image * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) - 1
    segmentationImage <- matrixToImages( segmentationMatrix, image * 0 + 1 )[[1]]

    return( list( segmentationImage = segmentationImage,
                  probabilityImages = probabilityImages ) )

    } else if( modality == "protonLobes" || modality == "maskLobes" ) {

    reorientTemplateFileNamePath <- getANTsXNetData( "protonLungTemplate" )
    reorientTemplate <- antsImageRead( reorientTemplateFileNamePath )

    resampledImageSize <- dim( reorientTemplate )

    spatialPriorsFileNamePath <- getANTsXNetData( "protonLobePriors" )
    spatialPriors <- antsImageRead( spatialPriorsFileNamePath )
    priorsImageList <- splitNDImageToList( spatialPriors )

    channelSize <- 1 + length( priorsImageList )
    numberOfClassificationLabels <- 1 + length( priorsImageList )

    unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels,
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      additionalOptions = c( "attentionGating" ) )

    if( modality == "protonLobes" )
      {
      penultimateLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]$output
      outputs2 <- penultimateLayer %>% layer_conv_3d( filters = 1,
        kernel_size = c( 1L, 1L, 1L ), activation = 'sigmoid',
        kernel_regularizer = regularizer_l2( l = 0.0 ) )
      unetModel = keras_model( inputs = unetModel$input,
        outputs = list( unetModel$output, outputs2 ) )
      weightsFileName <- getPretrainedNetwork( "protonLobes" )
      } else {
      weightsFileName <- getPretrainedNetwork( "maskLobes" )
      }
    unetModel$load_weights( weightsFileName )

    if( verbose )
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
    if( modality == "protonLobes" )
      {
      warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )
      } else {
      warpedArray[warpedArray != 0] = 1
      }

    batchX <- array( data = 0, dim = c( 1, resampledImageSize, channelSize ) )
    batchX[1,,,,1] <- warpedArray
    for( i in seq.int(length( priorsImageList ) ) )
      {
      batchX[1,,,,i+1] <- as.array( priorsImageList[[i]] )
      }

    predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    if( modality == "protonLobes" )
      {
      predictedDataBatch <- array( predictedData[[1]][1,,,,], dim = c( dim( reorientTemplate ), numberOfClassificationLabels ) )
      probabilityImagesList <- oneHotToSegmentation( predictedDataBatch, reorientTemplate )
      } else {
      predictedDataBatch <- array( predictedData[1,,,,], dim = c( dim( reorientTemplate ), numberOfClassificationLabels ) )
      probabilityImagesList <- oneHotToSegmentation( predictedDataBatch, reorientTemplate )
      }

    if( verbose )
      {
      cat( "Lung extraction:  renormalize probability mask to native space.\n" )
      }

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      probabilityImageTmp <- probabilityImagesList[[i]]
      probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImageTmp, image )
      }

    imageMatrix <- imageListToMatrix( probabilityImages, image * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) - 1
    segmentationImage <- matrixToImages( segmentationMatrix, image * 0 + 1 )[[1]]

    if( modality == "protonLobes" )
      {
      predictedDataBatch <- array( predictedData[[2]][1,,,,], dim = c( dim( reorientTemplate ), numberOfClassificationLabels ) )
      wholeLungMask <- oneHotToSegmentation( predictedDataBatch, reorientTemplate )[[1]]

      wholeLungMask <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        wholeLungMask, image )
      return( list( segmentationImage = segmentationImage,
                    probabilityImages = probabilityImages,
                    wholeLungMaskImage = wholeLungMask ) )
      } else {
      return( list( segmentationImage = segmentationImage,
                    probabilityImages = probabilityImages ) )

      }

    } else if( modality == "ct" ) {

    ################################
    #
    # Preprocess image
    #
    ################################

    if( verbose )
      {
      cat("Preprocess CT image.\n")
      }

    closestSimplifiedDirectionMatrix <- function( direction )
      {
      closest = floor( abs( direction ) + 0.5 )
      closest[direction < 0] <- closest[direction < 0] * -1.0
      return( closest )
      }

    simplifiedDirection <- closestSimplifiedDirectionMatrix( antsGetDirection( image ) )

    referenceImageSize <- c( 128, 128, 128 )

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
    ctPreprocessed <- iMath( ctPreprocessed, "Normalize" )
    ctPreprocessedWarped = applyAntsrTransformToImage(
        xfrm, ctPreprocessed, referenceImage, interpolation = "nearestneighbor" )
    ctPreprocessedWarped <- iMath( ctPreprocessedWarped, "Normalize" ) - 0.5

    ################################
    #
    # Build models and load weights
    #
    ################################

    if( verbose )
      {
      cat( "Build model and load weights.\n" )
      }

    weightsFileName <- getPretrainedNetwork( "lungCtWithPriorsSegmentationWeights" )

    classes <- c( "background", "left lung", "right lung", "airways" )
    numberOfClassificationLabels <- length( classes )

    luna16Priors <- splitNDImageToList( antsImageRead( getANTsXNetData( "luna16LungPriors" ) ) )
    for( i in seq.int( length( luna16Priors ) ) )
      {
      luna16Priors[[i]] <- resampleImage( luna16Priors[[i]], referenceImageSize, useVoxels = TRUE )
      }
    channelSize <- length( luna16Priors ) + 1

    unetModel <- createUnetModel3D( c( referenceImageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose )
      {
      cat( "Prediction.\n" )
      }

    batchX <- array( data = 0, dim = c( 1, referenceImageSize, channelSize ) )
    batchX[,,,,1] <- as.array( ctPreprocessedWarped )
    for( i in seq.int( length( luna16Priors ) ) )
      {
      batchX[,,,,i+1] <- as.array( luna16Priors[[i]] ) - 0.5
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      if( verbose )
        {
        cat( "Reconstructing image", classes[i], "\n" )
        }
      probabilityImage <- as.antsImage( drop( predictedData[,,,,i] ), reference = ctPreprocessedWarped )
      probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
        probabilityImage, ctPreprocessed )
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

    } else if( modality == "ventilation" ) {

      templateSize <- c( 256L, 256L )

      imageModalities <- c( "Ventilation" )
      channelSize <- length( imageModalities )

      preprocessedImage <- ( image - mean( image ) ) / sd( image )

      ################################
      #
      # Build models and load weights
      #
      ################################

      unetModel <- createUnetModel2D( c( templateSize, channelSize ),
        numberOfOutputs = 1, mode = 'sigmoid',
        numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.0,
        convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
        weightDecay = 0 )

      if( verbose )
        {
        cat( "Whole lung mask: retrieving model weights.\n" )
        }
      weightsFileName <- getPretrainedNetwork( "wholeLungMaskFromVentilation" )
      unetModel$load_weights( weightsFileName )

      ################################
      #
      # Extract slices
      #
      ################################

      dimensionsToPredict <- c( which.max( antsGetSpacing( preprocessedImage ) )[1] )

      batchX <- array( data = 0,
        c( sum( dim( preprocessedImage )[dimensionsToPredict]), templateSize, channelSize ) )

      sliceCount <- 1
      for( d in seq.int( length( dimensionsToPredict ) ) )
        {
        numberOfSlices <- dim( preprocessedImage )[dimensionsToPredict[d]]

        if( verbose )
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

          ventilationSlice <- padOrCropImageToSize(
             extractSlice( preprocessedImage, i, dimensionsToPredict[d], collapseStrategy = 1 ), templateSize )
          batchX[sliceCount,,,1] <- as.array( ventilationSlice )

          sliceCount <- sliceCount + 1
          }
        if( verbose )
          {
          cat( "\n" )
          }
        }

      ################################
      #
      # Do prediction and then restack into the image
      #
      ################################

      if( verbose )
        {
        cat( "Prediction.\n" )
        }

      prediction <- predict( unetModel, batchX, verbose = verbose )

      permutations <- list()
      permutations[[1]] <- c( 1, 2, 3 )
      permutations[[2]] <- c( 2, 1, 3 )
      permutations[[3]] <- c( 2, 3, 1 )

      probabilityImage <- antsImageClone( image ) * 0

      currentStartSlice <- 1
      for( d in seq.int( length( dimensionsToPredict ) ) )
        {
        currentEndSlice <- currentStartSlice - 1 + dim( preprocessedImage )[dimensionsToPredict[d]]
        whichBatchSlices <- currentStartSlice:currentEndSlice

        predictionPerDimension <- prediction[whichBatchSlices,,,1]
        predictionArray <- aperm( drop( predictionPerDimension ), permutations[[dimensionsToPredict[d]]] )
        predictionImage <- antsCopyImageInfo( preprocessedImage,
          padOrCropImageToSize( as.antsImage( predictionArray ), dim( preprocessedImage ) ) )
        probabilityImage <- probabilityImage + ( predictionImage - probabilityImage ) / d
        currentStartSlice <- currentEndSlice + 1
        }

      return( probabilityImage )

    } else if( modality == "xray" ) {

    ################################
    #
    # Preprocess image
    #
    ################################

    if( verbose )
      {
      cat("Preprocess Xray image.\n")
      }

    classes <- c( "background", "leftLung", "rightLung" )
    numberOfClassificationLabels <- length( classes )
    resampledImageSize <- c( 256, 256 )
    channelSize <- 3

    resampledImage <- resampleImage( image, resampledImageSize, useVoxels = TRUE, interpType = 0 )
    xrayLungPriors <- splitNDImageToList( antsImageRead( getANTsXNetData( "xrayLungPriors" ) ) )

    ################################
    #
    # Build models and load weights
    #
    ################################

    if( verbose )
      {
      cat( "Build model and load weights.\n" )
      }

    weightsFileName <- getPretrainedNetwork( "xrayLungExtraction" )

    unetModel <- createUnetModel2D( c( resampledImageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
      weightDecay = 0, additionalOptions = NULL )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose )
      {
      cat( "Prediction.\n" )
      }

    batchX <- array( data = 0, dim = c( 1, resampledImageSize, channelSize ) )
    resampledArray <- as.array( resampledImage )
    batchX[,,,1] <- ( resampledArray - min( resampledArray ) ) / ( max( resampledArray ) - min( resampledArray ) )
    batchX[,,,2] <- as.array( xrayLungPriors[[1]] )
    batchX[,,,3] <- as.array( xrayLungPriors[[2]] )

    predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    origin <- antsGetOrigin( resampledImage )
    spacing <- antsGetSpacing( resampledImage )
    direction <- antsGetDirection( resampledImage )

    probabilityImages <- list()
    for( i in seq_len( numberOfClassificationLabels ) )
      {
      if( verbose )
        {
        cat( "Reconstructing image", classes[i], "\n" )
        }
      probabilityImage <- as.antsImage( drop( predictedData[,,,i] ), reference = resampledImage )
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