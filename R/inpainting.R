#' wholeHeadInpainting
#'
#' Perform in-painting for whole-head MRI
#'
#' @param image input 3-D MR image.
#' @param roiMask binary mask image
#' @param modality Modality image type.  Options include: "t1": T1-weighted MRI,
#' "flair": FLAIR MRI.
#' @param mode Options include:  "sagittal": sagittal view network, "coronal": 
#' coronal view network, "axial": axial view network, "average": average of all 
#' canonical views, "meg": morphological erosion, greedy, iterative.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return inpainted image
#' @author Tustison NJ
#' @export
wholeHeadInpainting <- function( image, roiMask, modality = "t1", mode = "axial",
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( image@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  if( mode == "sagittal" || mode == "coronal" || mode == "axial" )  
    {
    reorientTemplate <- antsImageRead( getANTsXNetData( "nki" ) )
    reorientTemplate <- padOrCropImageToSize( reorientTemplate, c( 256, 256, 256 ) )

    centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
    centerOfMassImage <- getCenterOfMass( image )
    xfrm <- createAntsrTransform( type = "Euler3DTransform", center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )

    imageReoriented <- applyAntsrTransformToImage( xfrm, image, reorientTemplate, interpolation = "linear" )
    roiMaskReoriented <- applyAntsrTransformToImage( xfrm, roiMask, reorientTemplate, interpolation = "nearestNeighbor" )
    roiMaskReoriented <- thresholdImage( roiMaskReoriented, 0, 0, 0, 1 )

    geoms = labelGeometryMeasures( roiMaskReoriented )    
    if( dim( geoms )[1] != 1 )
      {
      stop( "ROI is not specified correctly." )
      }
 
    lowerSlice <- NULL
    upperSlice <- NULL
    weightsFile <- NULL
    direction <- -1
    if( mode == "sagittal" )
      {
      lowerSlice <- floor( geoms$BoundingBoxLower_x )
      upperSlice <- floor( geoms$BoundingBoxUpper_x )
      if( modality == "t1" )
        { 
        weightsFile <- getPretrainedNetwork( "inpainting_sagittal_rmnet_weights",
                                            antsxnetCacheDirectory = antsxnetCacheDirectory )
        # } else if( modality == "flair" ) {
        # weightsFile <- getPretrainedNetwork( "inpainting_sagittal_rmnet_flair_weights",
        #                                     antsxnetCacheDirectory = antsxnetCacheDirectory )
        } else {
        stop( "Unrecognized modality." )
        }
      direction <- 1
      } else if( mode == "coronal" ) {
      lowerSlice <- floor( geoms$BoundingBoxLower_y )
      upperSlice <- floor( geoms$BoundingBoxUpper_y )
      if( modality == "t1" )
        { 
        weightsFile <- getPretrainedNetwork( "inpainting_coronal_rmnet_weights",
                                            antsxnetCacheDirectory = antsxnetCacheDirectory )
        # } else if( modality == "flair" ) {
        # weightsFile <- getPretrainedNetwork( "inpainting_coronal_rmnet_flair_weights",
        #                                     antsxnetCacheDirectory = antsxnetCacheDirectory )
        } else {
        stop( "Unrecognized modality." )
        }
      direction <- 2
      } else if( mode == "axial" ) {
      lowerSlice <- floor( geoms$BoundingBoxLower_z )
      upperSlice <- floor( geoms$BoundingBoxUpper_z )
      if( modality == "t1" )
        { 
        weightsFile <- getPretrainedNetwork( "inpainting_axial_rmnet_weights",
                                            antsxnetCacheDirectory = antsxnetCacheDirectory )
        } else if( modality == "flair" ) {
        weightsFile <- getPretrainedNetwork( "inpainting_axial_rmnet_flair_weights",
                                            antsxnetCacheDirectory = antsxnetCacheDirectory )
        } else {
        stop( "Unrecognized modality." )
        }
      direction <- 3
      }

    model <- createRmnetGenerator()
    model$load_weights( weightsFile )
    
    numberOfSlices <- upperSlice - lowerSlice + 1

    imageSize <- c( 256, 256 )      
    channelSize <- 3
    batchX <- array( data = 0, dim = c( numberOfSlices, imageSize, channelSize ) )
    batchXMask <- array( data = 0, dim = c( numberOfSlices, imageSize, 1 ) )
    batchXMaxValues <- rep( 0, numberOfSlices )
    
    for( i in seq_len( numberOfSlices ) )
      {
      sliceIndex <- i + lowerSlice
      slice <- extractSlice( imageReoriented, slice = sliceIndex, 
                             direction = direction, collapseStrategy = 1 )
      batchX[i,,,1] <- as.array( slice )
      batchXMaxValues[i] <- max( batchX[i,,,1] )
      batchX[i,,,1] <- batchX[i,,,1] / ( 0.5 * batchXMaxValues[i] ) - 1.
      for( j in seq.int( 2, channelSize ) )
        {
        batchX[i,,,j] <- batchX[i,,,1]
        }
      maskSlice <- extractSlice( roiMaskReoriented, slice = sliceIndex, 
                                 direction = direction, collapseStrategy = 1 )
      batchXMask[i,,,1] <- as.array( maskSlice )
      }
   
    batchY <- model$predict( list( batchX, batchXMask ), verbose = verbose )[,,,1:3, drop = FALSE]

    inpaintedImageReorientedArray <- as.array( imageReoriented )   
    for( i in seq_len( numberOfSlices ) )
      {
      sliceIndex <- i + lowerSlice
      inpaintedValues = ( apply( batchY[i,,,], c( 1, 2 ), mean ) + 1 ) * ( 0.5 * batchXMaxValues[i] )
      if( direction == 1 )
        {
        inpaintedImageReorientedArray[sliceIndex,,] <- inpaintedValues
        } else if( direction == 2 ) {
        inpaintedImageReorientedArray[,sliceIndex,] <- inpaintedValues
        } else if( direction == 3 ) {
        inpaintedImageReorientedArray[,,sliceIndex] <- inpaintedValues
        }
      } 
    inpaintedImageReoriented <- as.antsImage( inpaintedImageReorientedArray ) 
    inpaintedImageReoriented <- antsCopyImageInfo( imageReoriented, inpaintedImageReoriented ) 

    xfrmInv <- invertAntsrTransform( xfrm )
    inpaintedImage <- applyAntsrTransformToImage( xfrmInv, inpaintedImageReoriented, image, interpolation = "linear" )
    inpaintedImage <- antsCopyImageInfo( image, inpaintedImage )
    inpaintedImage[roiMask == 0] <- image[roiMask == 0]

    return( inpaintedImage )
    
    } else if( mode == "average" ) {
    
    sagittal <- wholeHeadInpainting( image, roiMask = roiMask, modality = modality, 
                                     mode = "sagittal", verbose = verbose )
    coronal <- wholeHeadInpainting( image, roiMask = roiMask, modality = modality, 
                                    mode = "coronal", verbose = verbose )
    axial <- wholeHeadInpainting( image, roiMask = roiMask, modality = modality, 
                                  mode = "axial", verbose = verbose )

    return( ( sagittal + coronal + axial ) / 3 )
  
    } else if( mode == "meg" ) {

    currentImage <- antsImageClone( image )
    currentRoiMask <- thresholdImage( roiMask, 0, 0, 0, 1 )
    roiMaskVolume <- sum( currentRoiMask )

    iteration <- 0
    while( roiMaskVolume > 0 )
      {
      if( verbose )
        {
        cat( "roiMaskVolume (", iteration, "): ", roiMaskVolume, "\n" )
        }
      currentImage <- wholeHeadInpainting( currentImage, roiMask = roiMask, modality = modality,
                                           mode = "average", verbose = verbose )
      currentRoiMask <- iMath( currentRoiMask, "ME", 1 )
      iteration <- iteration + 1
      roiMaskVolume <- sum( currentRoiMask )
      }

    return( currentImage )
    }
  }
