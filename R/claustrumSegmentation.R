#' Claustrum segmentation
#'
#' Described here:
#'
#'     \url{https://pubmed.ncbi.nlm.nih.gov/34520080/}
#'
#' with the implementation available at:
#'
#'     \url{https://github.com/hongweilibran/claustrum_multi_view}
#'
#' @param t1 input 3-D T1-weighted brain image.
#' @param doPreprocessing perform n4 bias correction, denoising?
#' @param useEnsemble boolean to check whether to use all 3 sets of weights.
#' @param verbose print progress.
#' @return claustrum probability image
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' probabilityMask <- claustrumSegmentation( image )
#' }
#' @export
claustrumSegmentation <- function( t1, doPreprocessing = TRUE,
  useEnsemble = TRUE, verbose = FALSE )
{

  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  imageSize <- c( 180, 180 )

  ################################
  #
  # Preprocess images
  #
  ################################

  numberOfChannels <- 1
  t1Preprocessed <- antsImageClone(t1)
  brainMask <- thresholdImage( t1, 0, 0, 0, 1 )
  if( doPreprocessing == TRUE )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.01, 0.99 ),
        brainExtractionModality = "t1",
        templateTransformType = NULL,
        doBiasCorrection = TRUE,
        doDenoising = TRUE,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage
    brainMask <- t1Preprocessing$brainMask
    }

  referenceImage <- makeImage( c( 170, 256, 256 ),
                               voxval = 1,
                               spacing = c( 1, 1, 1 ),
                               origin = c( 0, 0, 0 ),
                               direction = diag( 3 ) )
  centerOfMassReference <- getCenterOfMass( referenceImage )
  centerOfMassImage <- getCenterOfMass( brainMask )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassReference,
        translation = centerOfMassImage - centerOfMassReference )
  t1PreprocessedWarped <- applyAntsrTransformToImage( xfrm, t1Preprocessed, referenceImage )
  brainMaskWarped <- thresholdImage(
        applyAntsrTransformToImage( xfrm, brainMask, referenceImage ), 0.5, 1.1, 1, 0 )

  ################################
  #
  # Gaussian normalize intensity based on brain mask
  #
  ################################

  meanT1 <- mean( t1PreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
  sdT1 <- sd( t1PreprocessedWarped[brainMaskWarped > 0], na.rm = TRUE )
  t1PreprocessedWarped <- ( t1PreprocessedWarped - meanT1 ) / sdT1

  t1PreprocessedWarped <- t1PreprocessedWarped * brainMaskWarped

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

  if( verbose == TRUE )
    {
    cat( "Claustrum:  retrieving axial model weights.\n" )
    }

  unetAxialModels <- list()
  for( i in seq.int( numberOfModels ) )
    {
    weightsFileName <- getPretrainedNetwork( paste0( "claustrum_axial_", i - 1 ) )

    unetAxialModels[[i]] <- createSysuMediaUnetModel2D( c( imageSize, numberOfChannels ), anatomy = "claustrum" )
    unetAxialModels[[i]]$load_weights( weightsFileName )
    }

  if( verbose == TRUE )
    {
    cat( "Claustrum:  retrieving coronal model weights.\n" )
    }

  unetCoronalModels <- list()
  for( i in seq.int( numberOfModels ) )
    {
    weightsFileName <- getPretrainedNetwork( paste0( "claustrum_coronal_", i - 1 ) )

    unetCoronalModels[[i]] <- createSysuMediaUnetModel2D( c( imageSize, numberOfChannels ), anatomy = "claustrum" )
    unetCoronalModels[[i]]$load_weights( weightsFileName )
    }

  ################################
  #
  # Extract slices
  #
  ################################

  dimensionsToPredict <- c( 2, 3 )  # (Coronal, Axial)

  batchCoronalX <- array( data = 0,
    c( sum( dim( t1PreprocessedWarped )[2]), imageSize, numberOfChannels ) )
  batchAxialX <- array( data = 0,
    c( sum( dim( t1PreprocessedWarped )[3]), imageSize, numberOfChannels ) )

  rotate <- function( x ) t( apply( x, 2, rev ) )
  rotateReverse <- function( x ) apply( t( x ), 2, rev )

  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    numberOfSlices <- dim( t1PreprocessedWarped )[dimensionsToPredict[d]]

    if( verbose == TRUE )
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

      t1Slice <- padOrCropImageToSize( extractSlice( t1PreprocessedWarped, i, dimensionsToPredict[d] ), imageSize )
      if( dimensionsToPredict[d] == 2 )
        {
        batchCoronalX[i,,,1] <- rotate( as.array( t1Slice ) )
        } else {
        batchAxialX[i,,,1] <- rotateReverse( as.array( t1Slice ) )
        }
      }
    if( verbose == TRUE )
      {
      cat( "\n" )
      }
    }

  ################################
  #
  # Do prediction and then restack into the image
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Coronal prediction.\n" )
    }

  predictionCoronal <- predict( unetCoronalModels[[1]], batchCoronalX, verbose = verbose )
  if( numberOfModels > 1 )
    {
    for( i in seq.int( from = 2, to = numberOfModels ) )
      {
      predictionCoronal <- predictionCoronal + predict( unetCoronalModels[[i]], batchCoronalX, verbose = verbose )
      }
    }
  predictionCoronal <- predictionCoronal / numberOfModels

  for( i in seq.int( dim( t1PreprocessedWarped )[2] ) )
    {
    predictionCoronal[i,,,1] <- rotateReverse( predictionCoronal[i,,,1] )
    }

  if( verbose == TRUE )
    {
    cat( "Axial prediction.\n" )
    }

  predictionAxial <- predict( unetAxialModels[[1]], batchAxialX, verbose = verbose )
  if( numberOfModels > 1 )
    {
    for( i in seq.int( from = 2, to = numberOfModels ) )
      {
      predictionAxial <- predictionAxial + predict( unetAxialModels[[i]], batchAxialX, verbose = verbose )
      }
    }
  predictionAxial <- predictionAxial / numberOfModels

  for( i in seq.int( dim( t1PreprocessedWarped )[3] ) )
    {
    predictionAxial[i,,,1] <- rotate( predictionAxial[i,,,1] )
    }

  if( verbose == TRUE )
    {
    cat( "Restack image and transform back to native space.\n" )
    }

  permutations <- list()
  permutations[[1]] <- c( 1, 2, 3 )
  permutations[[2]] <- c( 2, 1, 3 )
  permutations[[3]] <- c( 2, 3, 1 )

  predictionImageAverage <- antsImageClone( t1PreprocessedWarped ) * 0

  for( d in seq.int( length( dimensionsToPredict ) ) )
    {
    whichBatchSlices <- 1:dim( t1PreprocessedWarped )[dimensionsToPredict[d]]
    predictionPerDimension <- NULL
    if( dimensionsToPredict[d] == 2 )
      {
      predictionPerDimension <- predictionCoronal[whichBatchSlices,,,]
      } else {
      predictionPerDimension <- predictionAxial[whichBatchSlices,,,]
      }
    predictionArray <- aperm( drop( predictionPerDimension ), permutations[[dimensionsToPredict[d]]] )
    predictionImage <- antsCopyImageInfo( t1PreprocessedWarped,
      padOrCropImageToSize( as.antsImage( predictionArray ), dim( t1PreprocessedWarped ) ) )
    predictionImageAverage <- predictionImageAverage + ( predictionImage - predictionImageAverage ) / d
    }

  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      predictionImageAverage, t1 ) * thresholdImage( brainMask, 0.5, 1, 1, 0 )

  return( probabilityImage = probabilityImage )
}

