#' wholeHeadInpainting
#'
#' Perform in-painting for whole-head MRI
#'
#' @param image input 3-D MR image.
#' @param roiMask binary mask image
#' @param modality Modality image type.  Options include: "t1": T1-weighted MRI.
#' "flair": FLAIR MRI.
#' @param slicewise Two models per modality are available for processing the data.  One model
#' is based on training/prediction using 2-D axial slice data whereas the
#' other uses 64x64x64 patches.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return inpainted image
#' @author Tustison NJ
#' @export
wholeHeadInpainting <- function( image, roiMask, modality = "t1", slicewise = TRUE,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( image@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  if( slicewise )
    {
    imageSize <- c( 256, 256 )
    channelSize <- 1

    if( verbose )
      {
      cat( "Preprocessing:  Reorientation.\n" )
      }

    reorientTemplate <- antsImageRead( getANTsXNetData( "oasis" ) )

    centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
    centerOfMassImage <- getCenterOfMass( image )
    xfrm <- createAntsrTransform( type = "Euler3DTransform", center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )

    imageReoriented <- applyAntsrTransformToImage( xfrm, image, reorientTemplate, interpolation = "linear" )
    roiMaskReoriented <- applyAntsrTransformToImage( xfrm, roiMask, reorientTemplate, interpolation = "nearestNeighbor" )
    roiMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 0, 1 )
    roiInvertedMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 1, 0 )

    inpaintingUnet <- createPartialConvolutionUnetModel2D( c( imageSize, channelSize ),
                   numberOfPriors = 0, numberOfFilters = c( 32, 64, 128, 256, 512, 512 ),
                   kernelSize = 3 )

    weightsName <- ''
    if( modality == "T1" || modality == "t1" )
      {
      weightsName <- "wholeHeadInpaintingT1"
      } else if( modality == "FLAIR" || modality == "FLAIR" ) {
      weightsName <- "wholeHeadInpaintingFLAIR"
      } else {
      stop( paste0( "Unavailable modality given: ", modality ) )
      }

    weightsFileName <- getPretrainedNetwork( weightsName,
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    inpaintingUnet$load_weights( weightsFileName )

    geoms <- labelGeometryMeasures( roiMaskReoriented )
    if( dim( geoms )[1] != 1 )
      {
      stop( "ROI is not specified correctly." )
      }
    lowerSlice <- floor( geoms$BoundingBoxLower_y )
    upperSlice <- floor( geoms$BoundingBoxUpper_y )
    numberOfSlices <- upperSlice - lowerSlice + 1

    if( verbose )
      {
      cat( "Preprocessing:  Slicing data.\n" )
      }

    batchX <- array( data = 0, dim = c( numberOfSlices, imageSize, channelSize ) )
    batchXMask <- array( data = 0, dim = c( numberOfSlices, imageSize, channelSize ) )

    for( i in seq_len( numberOfSlices ) )
      {
      index <- lowerSlice + i

      maskSlice <- extractSlice( roiInvertedMaskReoriented, index, 2, collapseStrategy = 1 )
      maskSlice <- padOrCropImageToSize( maskSlice, imageSize )
      maskSliceArray <- as.array( maskSlice )

      slice <- extractSlice( imageReoriented, index, 2, collapseStrategy = 1 )
      slice <- padOrCropImageToSize( slice, imageSize )
      slice <- maskSlice * ( slice - min( slice ) ) / ( max( slice ) - min( slice ) )

      slice[maskSlice == 0] <- 1
      sliceArray <- as.array( slice )

      for( j in seq_len( channelSize ) )
        {
        batchX[i,,,j] <- sliceArray
        batchXMask[i,,,j] <- maskSliceArray
        }
      }

    if( verbose )
      {
      cat( "Prediction.\n" )
      }

    predictedData <- inpaintingUnet$predict( list( batchX, batchXMask ), verbose = verbose )
    predictedData[batchXMask == 1] <- batchX[batchXMask == 1]

    if( verbose )
      {
      cat( "Post-processing:  Slicing data.\n" )
      }

    imageReorientedArray <- as.array( imageReoriented )
    for( i in seq_len( numberOfSlices ) )
      {
      index <- lowerSlice + i

      slice <- extractSlice( imageReoriented, index, 2, collapseStrategy = 1 )
      maskSlice <- extractSlice( roiInvertedMaskReoriented, index, 2, collapseStrategy = 1 )
      predictedSlice <- as.antsImage( predictedData[i,,,1], reference = slice )
      predictedSlice <- padOrCropImageToSize( predictedSlice, dim( slice ) )
      predictedSlice <- regressionMatchImage( predictedSlice, slice, mask = maskSlice )

      imageReorientedArray[,index,] <- as.array( predictedSlice )
      }

    inpaintedImage <- as.antsImage( imageReorientedArray, reference = imageReoriented )

    if( verbose )
      {
      cat( "Post-processing:  reorienting to original space.\n" )
      }

    xfrmInv <- invertAntsrTransform( xfrm )
    inpaintedImage <- applyAntsrTransformToImage( xfrmInv, inpaintedImage, image, interpolation = "linear" )
    inpaintedImage <- antsCopyImageInfo( image, inpaintedImage )
    inpaintedImage[roiMask == 0] <- image[roiMask == 0]

    return( inpaintedImage )

    } else {

    imageSize <- c( 256, 256, 256 )
    patchSize <- c( 64, 64, 64 )
    strideLength <- c( 32, 32, 32 )
    channelSize <- 1

    reorientTemplate <- antsImageRead( getANTsXNetData( "oasis" ) )
    reorientTemplate <- padOrCropImageToSize( reorientTemplate, imageSize )

    centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
    centerOfMassImage <- getCenterOfMass( image )
    xfrm <- createAntsrTransform( type = "Euler3DTransform", center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )

    imageReoriented <- applyAntsrTransformToImage( xfrm, image, reorientTemplate, interpolation = "linear" )
    roiMaskReoriented <- applyAntsrTransformToImage( xfrm, roiMask, reorientTemplate, interpolation = "nearestNeighbor" )
    roiMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 0, 1 )
    roiInvertedMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 1, 0 )

    inpaintingUnet <- createPartialConvolutionUnetModel3D( c( patchSize, channelSize ),
                   numberOfPriors = 0, numberOfFilters = c( 32, 64, 128, 256, 256 ),
                   kernelSize = 3 )

    weightsName <- ''
    if( modality == "T1" || modality == "t1" )
      {
      weightsName <- "wholeHeadInpaintingPatchBasedT1"
      } else if( modality == "FLAIR" || modality == "FLAIR" ) {
      weightsName <- "wholeHeadInpaintingPatchBasedFLAIR"
      } else {
      stop( paste0( "Unavailable modality given: ", modality ) )
      }

    weightsFileName <- getPretrainedNetwork( weightsName,
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    inpaintingUnet$load_weights( weightsFileName )

    if( verbose )
      {
      cat( "Preprocessing:  Extracting patches.\n" )
      }

    imagePatches <- extractImagePatches( imageReoriented, patchSize, maxNumberOfPatches = "all",
        strideLength = strideLength, randomSeed = NULL, returnAsArray = TRUE )

    minImageVal <- min( imageReoriented[roiInvertedMaskReoriented == 1] )
    maxImageVal <- max( imageReoriented[roiInvertedMaskReoriented == 1] )
    imageReoriented <- ( imageReoriented - minImageVal ) / ( maxImageVal - minImageVal )

    imagePatchesRescaled <- extractImagePatches( imageReoriented, patchSize, maxNumberOfPatches = "all",
        strideLength = strideLength, randomSeed = NULL, returnAsArray = TRUE )
    maskPatches <- extractImagePatches( roiInvertedMaskReoriented, patchSize, maxNumberOfPatches = "all",
        strideLength = strideLength, randomSeed = NULL, returnAsArray = TRUE )

    batchX <- array( data = imagePatchesRescaled, dim = c( dim( imagePatchesRescaled ), 1 ) )
    batchXMask <- array( data = maskPatches, dim = c( dim( imagePatchesRescaled ), 1 ) )

    batchX[batchXMask == 0] <- 1

    predictedData <- array( data = 0, dim = dim( batchX ) )

    for( i in seq_len( dim( batchX )[1] ) )
      {
      if( any( batchXMask[i,,,,] == 0 ) )
        {
        if( verbose )
          {
          cat( "  Predicting patch ", i, " (of", dim( batchX )[1], ")\n", sep = '' )
          }
        predictedPatch <- inpaintingUnet$predict( list( batchX[i,,,,, drop = FALSE],
                                                        batchXMask[i,,,,, drop = FALSE] ),
                                                  verbose = verbose )
        predictedPatchImage <- regressionMatchImage( as.antsImage( drop( predictedPatch ) ),
                                                     as.antsImage( drop( imagePatches[i,,,] ) ),
                                                     mask = as.antsImage( drop( batchXMask[i,,,,] ) ) )
        predictedData[i,,,,1] <- as.array( predictedPatchImage )
        } else {
        predictedData[i,,,,1] <- batchX[i,,,,]
        }
      }

    inpaintedImage <- reconstructImageFromPatches( drop( predictedData ),
        imageReoriented, strideLength = strideLength )

    if( verbose )
      {
      cat( "Post-processing:  reorienting to original space.\n" )
      }

    xfrmInv <- invertAntsrTransform( xfrm )
    inpaintedImage <- applyAntsrTransformToImage( xfrmInv, inpaintedImage, image, interpolation = "linear" )
    inpaintedImage <- antsCopyImageInfo( image, inpaintedImage )
    inpaintedImage[roiMask == 0] <- image[roiMask == 0]

    return( inpaintedImage )
    }
  }
