#' White matter hyperintensity segmentation 
#'
#' Perform WMH segmentation using the winning submission in the MICCAI
#' 2017 challenge by the sysu_media team using FLAIR or T1/FLAIR.  The 
#' MICCAI challenge is discussed in 
#' 
#' \url{https://pubmed.ncbi.nlm.nih.gov/30908194/} 
#'
#' with the sysu_media's team entry is discussed in 
#'
#'  \url{https://pubmed.ncbi.nlm.nih.gov/30125711/}
#'
#' with the original implementation available here:
#'
#'     \url{https://github.com/hongweilibran/wmh_ibbmTum}
#'
#' @param flair input 3-D FLAIR brain image.
#' @param t1 input 3-D T1-weighted brain image (assumed to be aligned to 
#' the flair, if specified).
#' @param brainMask input 3-D brain mask image.  If not specified, \code{getMask}
#' is used to estimate a mask which is not recommended.
#' @param doPreprocessing perform n4 bias correction and normalize to template?
#' @param useEnsemble boolean to check whether to use all 3 sets of weights.
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return WMH segmentation image
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "flair.nii.gz" )
#' probabilityMask <-sysuMediaWmhSegmentation( image )
#' }
#' @export
sysuMediaWmhSegmentation <- function( flair, t1 = NULL, brainMask = NULL, 
  doPreprocessing = TRUE, useEnsemble = TRUE, outputDirectory = NULL, 
  verbose = FALSE )
{

  padOrCropImageToSize <- function( image, size )
    {
    imageSize <- dim( image )
    delta <- imageSize - size

    if( any( delta < 0 ) )
      {
      padSize <- abs( min( delta ) )
      image <- iMath( image, "PadImage", padSize )
      }
    croppedImage <- cropImageCenter( image, size )  
    return( croppedImage )
    }

  if( flair@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )  
    }

  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }

  ################################
  #
  # Preprocess images
  #
  ################################  

  flairPreprocessed <- flair
  if( doPreprocessing == TRUE )
    {
    flairPreprocessing <- preprocessBrainImage( flair, 
        truncateIntensity = c( 0.01, 0.99 ),
        doBrainExtraction = FALSE,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        outputDirectory = outputDirectory,
        verbose = verbose )
    flairPreprocessed <- flairPreprocessing$preprocessedImage
    }    

  numberOfChannels <- 1
  if( ! is.null( t1 ) ) 
    {
    t1Preprocessed <- t1
    if( doPreprocessing == TRUE )
      {
      t1Preprocessing <- preprocessBrainImage( t1, 
          truncateIntensity = c( 0.01, 0.99 ),
          doBrainExtraction = FALSE,
          templateTransformType = NULL,
          doBiasCorrection = TRUE,
          doDenoising = FALSE,
          outputDirectory = outputDirectory,
          verbose = verbose )
      t1Preprocessed <- t1Preprocessing$preprocessedImage
      }  
    numberOfChannels <- 2
    }

  ################################
  #
  # Estimate mask (if not specified)
  #
  ################################  

  if( is.null( brainMask ) )
    {
    if( verbose == TRUE )    
      {
      cat( "Estimating brain mask." )    
      }
    if( ! is.null( t1 ) )  
      {
      brainMask <- getMask( t1, cleanup = 2 )    
      } else {
      brainMask <- getMask( flair, cleanup = 2 )    
      }
    }

  referenceImage <- makeImage( c( 200, 200, 200 ),
                               voxval = 1,
                               spacing = c( 1, 1, 1 ),
                               origin = c( 0, 0, 0 ), 
                               direction = diag( 3 ) )
  centerOfMassReference <- getCenterOfMass( referenceImage )
  centerOfMassImage <- getCenterOfMass( brainMask )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassReference,
        translation = centerOfMassImage - centerOfMassReference )
  flairPreprocessedWarped <- applyAntsrTransformToImage( xfrm, flairPreprocessed, referenceImage )
  brainMaskWarped <- thresholdImage( 
        applyAntsrTransformToImage( xfrm, brainMask, referenceImage ), 0.5, 1.1, 1, 0 )
  if( ! is.null( t1 ) )
    {
    t1PreprocessedWarped <- applyAntsrTransformToImage( xfrm, t1Preprocessed, referenceImage )
    }

  ################################
  #
  # Gaussian normalize intensity based on brain mask
  #
  ################################  

  meanFlair <- mean( flairPreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
  sdFlair <- sd( flairPreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
  flairPreprocessedWarped <- ( flairPreprocessedWarped - meanFlair ) / sdFlair

  if( numberOfChannels == 2 )
    {
    meanT1 <- mean( t1PreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
    sdT1 <- sd( t1PreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
    t1PreprocessedWarped <- ( t1PreprocessedWarped - meanT1 ) / sdT1
    }

  ################################
  #
  # Build models and load weights
  #
  ################################  

  numberOfModels <- 1
  if( useEnsemble == TRUE )
    {
    numberOfModels <- 3    
    }

  unetModels <- list()
  for( i in seq.int( numberOfModels ) ) 
    {
    weightsFileName <- ''
    if( numberOfChannels == 1 )
      {      
      weightsFileName <- paste0( outputDirectory, "sysuMediaWmhFlairOnlyModel", i - 1, ".h5" )
      if( ! file.exists( weightsFileName ) )
        {
        if( verbose == TRUE )
          {
          cat( "White matter hyperintensity:  downloading model weights.\n" )
          }
        weightsFileName <- getPretrainedNetwork( paste0( "sysuMediaWmhFlairOnlyModel", i - 1 ), weightsFileName )
        }
      } else {
      weightsFileName <- paste0( outputDirectory, "sysuMediaWmhFlairT1Model", i - 1, ".h5" )
      if( ! file.exists( weightsFileName ) )
        {
        if( verbose == TRUE )
          {
          cat( "White matter hyperintensity:  downloading model weights.\n" )
          }
        weightsFileName <- getPretrainedNetwork( paste0( "sysuMediaWmhFlairT1Model", i - 1 ), weightsFileName )
        }
      }

    unetModels[[i]] <- createSysuMediaUnetModel2D( c( 200, 200, numberOfChannels ) )
    unetModels[[i]]$load_weights( weightsFileName )
    }

  ################################
  #
  # Extract slices
  #
  ################################  

  numberOfAxialSlices <- dim( flairPreprocessedWarped )[3]

  if( verbose == TRUE )
    {
    cat( "Extracting slices." )    
    pb <- txtProgressBar( min = 1, max = numberOfAxialSlices, style = 3 )
    }

  batchX <- array( data = 0, c( numberOfAxialSlices, 200, 200, numberOfChannels ) ) 
  for( i in seq.int( numberOfAxialSlices ) )
    {
    if( verbose )    
      {
      setTxtProgressBar( pb, i )          
      }

    flairSlice <- padOrCropImageToSize( extractSlice( flairPreprocessedWarped, i, 3 ), c( 200, 200 ) )
    batchX[i,,,1] <- as.array( flairSlice )
    if( numberOfChannels == 2 )
      {
      t1Slice <- padOrCropImageToSize( extractSlice( t1PreprocessedWarped, i, 3 ), c( 200, 200 ) )
      batchX[i,,,2] <- as.array( t1Slice )
      }
    }
  cat( "\n" )  

  ################################
  #
  # Do prediction and then restack into the image
  #
  ################################  

  if( verbose == TRUE )
    {
    cat( "Prediction." )    
    }
  
  prediction <- predict( unetModels[[1]], batchX, verbose = verbose )
  if( numberOfModels > 1 )
    {
    for( i in seq.int( from = 2, to = numberOfModels ) )
      {
      prediction <- prediction + predict( unetModels[[i]], batchX, verbose = verbose )
      }
    } 
  prediction <- prediction / numberOfModels

  predictionArray <- aperm( drop( prediction ), c( 2, 3, 1 ) )
  predictionImage <- padOrCropImageToSize( as.antsImage( predictionArray ), dim( flairPreprocessedWarped ) )
  probabilityImageWarped <- antsCopyImageInfo( flairPreprocessedWarped, predictionImage )
  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      probabilityImageWarped, flair )

  return( probabilityImage )
}  


