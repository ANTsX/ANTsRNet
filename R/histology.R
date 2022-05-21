#' Arterial lesion segmentation
#'
#' Perform arterial lesion segmentation using U-net.
#'
#' @param image input histology image.
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
#' image <- antsImageRead( "image.nii.gz" )
#' output <- arterialLesionSegmentation( image )
#' }
#' @import keras
#' @export
arterialLesionSegmentation <- function( image,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( image@dimension != 2 )
    {
    stop( "Image dimension must be 2." )
    }

  channelSize <- 1

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  weightsFileName <- getPretrainedNetwork( "arterialLesionWeibinShi",
    antsxnetCacheDirectory = antsxnetCacheDirectory )

  resampledImageSize <- c( 512, 512 )

  unetModel <- createUnetModel2D( c( resampledImageSize, 1 ),
    numberOfOutputs = 1, mode = "sigmoid",
    numberOfFilters = c( 64, 96, 128, 256, 512 ),
    convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
    dropoutRate = 0.0, weightDecay = 0,
    additionalOptions = c( "initialConvolutionKernelSize[5]", "attentionGating" ) )
  unetModel$load_weights( weightsFileName )

  if( verbose == TRUE )
    {
    cat( "Preprocessing:  Resampling and N4 bias correction.\n" )
    }
  preprocessedImage <- antsImageClone( image )
  preprocessedImage <- preprocessedImage / max( preprocessedImage )
  preprocessedImage <- resampleImage( preprocessedImage, resampledImageSize,
      useVoxels = TRUE, interpType = 0 )
  preprocessedImage <- n4BiasFieldCorrection( preprocessedImage,
      shrinkFactor = 2, returnBiasField = FALSE, verbose = verbose )

  batchX <- array( data = as.array( preprocessedImage ),
    dim = c( 1, resampledImageSize, channelSize ) )
  batchX <- ( batchX - min( batchX ) ) / ( max( batchX ) - min( batchX ) )

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  foregroundProbabilityImage <- as.antsImage( drop( predictedData ), reference = preprocessedImage )

  if( verbose == TRUE )
    {
    cat( "Post-processing:  resampling to original space.\n" )
    }

  foregroundProbabilityImage <- resampleImageToTarget( foregroundProbabilityImage, image )

  return( foregroundProbabilityImage )
  }

#' Perform brain extraction of Allen's E13.5 mouse embroyonic data.
#'
#' @param image input image
#' @param view Two trained networks are available:  "coronal" or "sagittal".
#' @param whichAxis If 3-D image, which_axis specifies the direction of the "view".
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
#' image <- antsImageRead( "image.nii.gz" )
#' output <- e13x5BrainExtraction( image )
#' }
#' @import keras
#' @export
e13x5BrainExtraction <- function( image, view = c( "coronal", "sagittal" ), 
  whichAxis = 3, antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  if( whichAxis < 1 || whichAxis > 3 )
    {
    stop( "Chosen axis not supported." )
    }

  weightsFileName <- ""
  if( tolower( view ) == "coronal" )
    {
    weightsFileName <- getPretrainedNetwork( "e13x5_coronal_weights",
        antsxnetCacheDirectory = antsxnetCacheDirectory )
    } else if( tolower( view ) == "sagittal" ) {
    weightsFileName <- getPretrainedNetwork( "e13x5_sagittal_weights",
        antsxnetCacheDirectory = antsxnetCacheDirectory )
    } else {
    stop( "Valid view options are coronal and sagittal.")
    }

  resampledImageSize <- c( 512, 512 )
  originalSliceShape <- dim( image )
  if( image@dimension > 2 )
    {
    originalSliceShape <- originalSliceShape[-c( whichAxis )] 
    }

  unetModel <- createUnetModel2D( c( resampledImageSize, 1 ),
    numberOfOutputs = 2, mode = "classification",
    numberOfFilters = c( 64, 96, 128, 256, 512 ),
    convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
    dropoutRate = 0.0, weightDecay = 0,
    additionalOptions = c( "initialConvolutionKernelSize[5]", "attentionGating" ) )
  unetModel$load_weights( weightsFileName )

  if( verbose )
    {
    cat( "Preprocessing:  Resampling.\n" )
    }

  numberOfChannels <- image@components
  numberOfSlices <- 1
  if( image@dimension > 2 )
    {
    numberOfSlices <- dim( image )[whichAxis]
    }

  imageChannels <- list()
  if( numberOfChannels == 1 )
    {
    imageChannels[[1]] <- image
    } else {
    imageChannels <- splitChannels( image )
    }

  batchX <- array( data = 0, dim = c( numberOfChannels * numberOfSlices, resampledImageSize, 1 ) )

  count <- 1
  for( i in seq.int( numberOfChannels ) )
    {
    imageChannelArray = as.array( imageChannels[[i]] )
    for( j in seq.int( numberOfSlices ) )
      {
      slice <- NULL
      if( image@dimension > 2 ) 
        {
        if( whichAxis == 1 ) 
          {
          imageChannelSliceArray <- imageChannelArray[j,,,drop = TRUE]
          } else if( whichAxis == 2 ) {
          imageChannelSliceArray <- imageChannelArray[,j,,drop = TRUE]
          } else {
          imageChannelSliceArray <- imageChannelArray[,,j,drop = TRUE]
          }
        slice <- as.antsImage( imageChannelSliceArray )  
        } else {
        slice <- imageChannels[[i]] 
        }
      sliceResampled <- resampleImage( slice, resampledImageSize, useVoxels = TRUE, interpType = 0 )
      sliceArray <- as.array( sliceResampled )
      if( max( sliceArray ) > min( sliceArray ) )
        {
        sliceArray <- ( sliceArray - min( sliceArray ) ) / ( max( sliceArray ) - min( sliceArray ) ) 
        }
      batchX[count,,,1] <- sliceArray
      count <- count + 1
      }
    }  

  if( verbose )
    {
    cat( "Prediction: " )
    }

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )

  if( numberOfChannels > 1 )
    {
    if( verbose )
       {
       cat( "Averaging across channels.\n" )
       }
    predictedDataTmp <- list()
    for( i in seq.int( numberOfChannels ) ) 
      {
      startIndex <- ( i - 1 ) * numberOfSlices + 1
      endIndex <- startIndex + numberOfSlices - 1
      predictedDataTmp[[i]] <- predictedData[startIndex:endIndex,,,] 
      }  
    predictedData <- predictedDataTmp[[1]]
    for( i in seq.int( 2, numberOfChannels ) ) 
      {
      predictedData <- ( predictedData * ( i - 1 ) + predictedDataTmp[[i]] ) / i
      }
    }

  if( verbose )
    {
    cat( "Post-processing:  resampling to original space.\n" )
    }

  foregroundProbabilityArray <- array( data = 0, dim = dim( image ) ) 
  for( j in seq.int( numberOfSlices ) )
    {
    sliceResampled <- as.antsImage( predictedData[j,,,2, drop = TRUE] )
    slice <- resampleImage( sliceResampled, originalSliceShape, useVoxels = TRUE, interpType = 0 )
    if( image@dimension == 2 )
      {
      foregroundProbabilityArray[,] <- as.array( slice )
      } else {
      if( whichAxis == 1 ) 
        {
        foregroundProbabilityArray[j,,] <- as.array( slice )
        } else if( whichAxis == 2 ) {
        foregroundProbabilityArray[,j,] <- as.array( slice )
        } else {
        foregroundProbabilityArray[,,j] <- as.array( slice )
        }
      }
    }

  origin <- antsGetOrigin( image )
  spacing <- antsGetSpacing( image )
  direction <- antsGetDirection( image )

  foregroundProbabilityImage <- as.antsImage( foregroundProbabilityArray, 
     origin = origin, spacing = spacing, direction = direction )

  return( foregroundProbabilityImage )
  }