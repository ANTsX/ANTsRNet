#' wholeHeadInpainting
#'
#' Perform in-painting for whole-head MRI
#'
#' @param image input 3-D MR image.
#' @param roiMask binary mask image
#' @param modality Modality image type.  Options include: "t1": T1-weighted MRI.
#' "flair": FLAIR MRI.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return inpainted image
#' @author Tustison NJ
#' @export
wholeHeadInpainting <- function( image, roiMask, modality = "t1",
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

  imageSize <- c( 256, 256 )
  channelSize <- 1

  reorientTemplate <- antsImageRead( getANTsXNetData( "oasis" ) )
  templatePriors <- list()

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

  if( verbose )
    {
    cat( "Preprocessing:  Reorientation.\n" )
    }

  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass( image )
  xfrm <- createAntsrTransform( type = "Euler3DTransform", center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate )

  imageReoriented <- applyAntsrTransformToImage( xfrm, image, reorientTemplate, interpolation = "linear" )
  roiMaskReoriented <- applyAntsrTransformToImage( xfrm, roiMask, reorientTemplate, interpolation = "nearestNeighbor" )
  roiMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 0, 1 )
  roiInvertedMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 1, 0 )

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
  if( length( templatePriors ) > 0 )
    {
    batchXPriors <- array( data = 0, dim = c( numberOfSlices, imageSize, length( templatePriors ) ) )
    }

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

    for( j in seq_len( length( templatePriors ) ) )
      {
      priorSlice <- extractSlice( templatePriors[j], index, 2, collapseStrategy = 1 )
      priorSlice <- padOrCropImageToSize( priorSlice, imageSize )
      batchXPriors <- as.array( priorSlice )
      }
    }

  if( verbose )
    {
    cat( "Prediction.\n" )
    }

  predictedData <- inpaintingUnet$predict( list( batchX, batchXMask ), verbose = verbose )
  # predictedData <- inpaintingUnet$predict( list( batchX, batchXMask, batchXPriors ), verbose = verbose )
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
  }
