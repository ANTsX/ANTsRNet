#' Arterial lesion segmentation
#'
#' Perform arterial lesion segmentation using U-net.
#'
#' @param image input 3-D lung image.
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
#' output <- lungExtraction( image )
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

  classes <- c( "background", "foreground" )
  numberOfClassificationLabels <- length( classes )

  resampledImageSize <- c( 512, 512 )

  unetModel <- createUnetModel2D( c( resampledImageSize, 1 ),
    numberOfOutputs = numberOfClassificationLabels, mode = "classification",
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
  probabilityImagesArray <- decodeUnet( predictedData, preprocessedImage )

  if( verbose == TRUE )
    {
    cat( "Post-processing:  resampling to original space.\n" )
    }

  probabilityImages <- list()
  for( i in seq_len( numberOfClassificationLabels ) )
    {
    probabilityImageTmp <- probabilityImagesArray[[1]][[i]]
    probabilityImages[[i]] <- resampleImageToTarget( probabilityImageTmp, image )
    }

  imageMatrix <- imageListToMatrix( probabilityImages, image * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) - 1
  segmentationImage <- matrixToImages( segmentationMatrix, image * 0 + 1 )[[1]]

  return( list( segmentationImage = segmentationImage,
                probabilityImages = probabilityImages ) )


  }