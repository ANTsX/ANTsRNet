#' Perform TID-based assessment of an image.
#'
#' Use a ResNet architecture to estimate image quality in 2D or 3D using the TID
#' image database described in
#'
#'    \url{https://www.sciencedirect.com/science/article/pii/S0923596514001490}
#'
#'
#' where the image assessment is either "global", i.e., a single number or an image
#' based on the specified patch size.  In the 3-D case, neighboring slices are used
#' for each estimate.  Note that parameters should be kept as consistent as possible
#' in order to enable comparison.  Patch size should be roughly 1/12th to 1/4th of
#' image size to enable locality. A global estimate can be gained by setting
#' \code{patchSize = "global"}.
#'
#' @param image the input image.  Either 2D or 3D.
#' @param mask optional mask for designating calculation ROI.
#' @param patchSize integer prime number for patch size; 101 is good. otherwise,
#' choose \code{"global"} for a single global estimate of quality.
#' @param strideLength optional value to speed up computation (typically less than
#' patch size).  Integer or vector of image dimension length.
#' @param paddingSize positive or negative integer (or vector of image dimension
#' length) for (de)padding to remove edge effects.
#' @param dimensionsToPredict if image dimension is 3, this parameter specifies
#' which dimension(s) should be used for prediction.  If more than one dimension
#' is specified, the results are averaged.
#' @param outputDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be reused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param whichModel model type e.g. string tidsQualityAssessment, koniqMBCS
#' @param verbose print progress.
#' @return list of QC results predicting both both human rater's mean and standard
#' deviation of the MOS ("mean opinion scores").  See the TID2013 paper.  Both
#' aggregate and spatial scores are returned, the latter in the form of an image.
#' @author Avants BB
#' @examples
#' \dontrun{
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' mask <- getMask( image )
#' tid <- tidNeuralImageAssessment( image, mask = mask, patchSize = 101L,
#'           strideLength = 7L, paddingSize = 0L )
#' plot( image, tid$MOS, alpha = 0.5)
#' cat( "mean MOS = ", tid$MOS.mean, "\n" )
#' cat( "sd MOS = ", tid$MOS.standardDeviationMean, "\n" )
#' }
#' @export
tidNeuralImageAssessment <- function( image, mask, patchSize = 101L,
  strideLength, paddingSize = 0L, dimensionsToPredict = 1,
  outputDirectory = NULL, whichModel="tidsQualityAssessment", verbose = FALSE )
{
  is.prime <- function( n )
    {
    return( n == 2L || all( n %% 2L:max( 2, floor( sqrt( n ) ) ) != 0 ) )
    }
  validModels = c("tidsQualityAssessment", "koniqMBCS")
  if ( ! any( whichModel %in% validModels ) ) 
    {
    cat( validModels )
    stop(" : Please pass valid model : ")
    }
  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }
  modelAndWeightsFileName <- paste0( outputDirectory, whichModel, ".h5" )
  if( ! file.exists( modelAndWeightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Neural QA:  downloading model and weights.\n" )
      }
    modelAndWeightsFileName <- getPretrainedNetwork( whichModel, modelAndWeightsFileName )
    }
  tidModel <- load_model_hdf5( filepath = modelAndWeightsFileName )

  paddingSizeVector <- paddingSize
  if( length( paddingSize ) == 1 )
    {
    paddingSizeVector <- rep( paddingSize, image@dimension )
    }

  paddedImageSize <- dim( image ) + paddingSizeVector
  paddedImage <- padOrCropImageToSize( image, paddedImageSize )

  numberOfChannels <- 3

  if ( missing( strideLength ) ) {
    strideLength = round( min( patchSize ) / 2 )
    if ( image@dimension == 3 ) strideLength = c( strideLength, strideLength, 1)
    }
  ###############
  #
  #  Global
  #
  ###############

  if( patchSize == 'global' )
    {
    if ( whichModel == "tidsQualityAssessment" ) 
      evaluationImage <- paddedImage %>% iMath( "Normalize" ) * 255

    if ( whichModel == "koniqMBCS" ) 
      evaluationImage <- ( paddedImage %>% iMath( "Normalize" ) ) * 2.0 - 1.0 

    if( image@dimension == 2 )
      {
      batchX <- array( dim = c( 1, dim( evaluationImage ), numberOfChannels ) )
      for( k in seq.int( 3 ) )
        {
        batchX[1,,,k] <- as.array( evaluationImage )
        }
      predictedData <- predict( tidModel, batchX, verbose = verbose )
      if ( whichModel == "tidsQualityAssessment" ) {
        return( list( MOS = NA,
                    MOS.standardDeviation = NA,
                    MOS.mean = predictedData[1, 1],
                    MOS.standardDeviationMean = predictedData[1, 2] ) )
        }
      if ( whichModel == "koniqMBCS" ) {
        return( list(
                    MOS.mean = predictedData[1, 1],
                    brightness.mean = predictedData[1, 2],
                    contrast.mean = predictedData[1, 3],
                    sharpness.mean = predictedData[1, 4] 
		    ) )
        }

      } else if( image@dimension == 3 ) {

      mosMean <- 0
      mosStandardDeviation <- 0
      brightness = 0
      contrast = 0
      sharpness = 0

      x <- seq.int( image@dimension )
      for( d in seq.int( length( dimensionsToPredict ) ) )
        {
        batchX <- array( data = 0,
          c( paddedImageSize[dimensionsToPredict[d]], paddedImageSize[!x %in% dimensionsToPredict[d]], numberOfChannels ) )

        batchX[1,,,1] <- as.array( extractSlice( evaluationImage, 1, dimensionsToPredict[d] ) )
        batchX[1,,,2] <- as.array( extractSlice( evaluationImage, 1, dimensionsToPredict[d] ) )
        batchX[1,,,3] <- as.array( extractSlice( evaluationImage, 2, dimensionsToPredict[d] ) )
        for( i in seq.int( from = 2, to = paddedImageSize[dimensionsToPredict[d]] - 1 ) )
          {
          batchX[i,,,1] <- as.array( extractSlice( evaluationImage, i - 1, dimensionsToPredict[d] ) )
          batchX[i,,,2] <- as.array( extractSlice( evaluationImage, i    , dimensionsToPredict[d] ) )
          batchX[i,,,3] <- as.array( extractSlice( evaluationImage, i + 1, dimensionsToPredict[d] ) )
          }
        batchX[paddedImageSize[dimensionsToPredict[d]],,,1] <-
          as.array( extractSlice( evaluationImage, paddedImageSize[dimensionsToPredict[d]] - 1, dimensionsToPredict[d] ) )
        batchX[paddedImageSize[dimensionsToPredict[d]],,,2] <-
          as.array( extractSlice( evaluationImage, paddedImageSize[dimensionsToPredict[d]], dimensionsToPredict[d] ) )
        batchX[paddedImageSize[dimensionsToPredict[d]],,,3] <-
          as.array( extractSlice( evaluationImage, paddedImageSize[dimensionsToPredict[d]], dimensionsToPredict[d] ) )

        predictedData <- predict( tidModel, batchX, verbose = verbose )

        mosMean <- mosMean + predictedData[1, 1]
        mosStandardDeviation <- mosStandardDeviation + predictedData[1, 2]
        if ( whichModel == "koniqMCBS" ) 
	  {
	  brightness = mosStandardDeviation	  
	  contrast = contrast + predictedData[1, 3]
          sharpness = sharpness + predictedData[1, 4]	  
          }
	}
      mosMean <- mosMean / length( dimensionsToPredict )
      mosStandardDeviation <- mosStandardDeviation / length( dimensionsToPredict )
      brightness = brightness / length( dimensionsToPredict )
      contrast = contrast/ length( dimensionsToPredict )
      sharpness = sharpness / length( dimensionsToPredict )
      if ( whichModel == "tidsQualityAssessment" )
        return( list( MOS.mean = mosMean,
                    MOS.standardDeviationMean = mosStandardDeviation ) )
      if ( whichModel == "koniqMBCS" )
        return( list( MOS.mean = mosMean, brightness.mean=brightness, contrast.mean=contrast, sharpness.mean=sharpness) )
      }
    } else {

    ###############
    #
    #  Patchwise
    #
    ###############

    evaluationImage <- paddedImage

    if( ! is.prime( patchSize ) )
      {
      stop( "Should pass a prime number for patch size." )
      }
    strideLengthVector <- strideLength
    if( length( strideLength ) == 1 )
      {
      if( image@dimension == 2 )
        {
        strideLengthVector <- c( strideLength, strideLength )
        }
      }

    patchSizeVector <- c( patchSize, patchSize )

    if( image@dimension == 2 )
      {
      dimensionsToPredict <- 1
      }

    permutations <- list()

    MOS <- image * 0
    MOS.standardDeviation <- image * 0
    contrast = image * 0.0
    sharpness = image * 0.0

    for( d in seq.int( length( dimensionsToPredict ) ) )
      {
      if( image@dimension == 3 )
        {
        permutations[[1]] <- c( 1, 2, 3 )
        permutations[[2]] <- c( 1, 3, 2 )
        permutations[[3]] <- c( 2, 3, 1 )

        if( dimensionsToPredict[d] == 1 )
          {
          patchSizeVector <- c( patchSize, patchSize, numberOfChannels )
          if( length( strideLength ) == 1 )
            {
            strideLengthVector <- c( strideLength, strideLength, 1 )
            }
          } else if( dimensionsToPredict[d] == 2 ) {
          patchSizeVector <- c( patchSize, numberOfChannels, patchSize )
          if( length( strideLength ) == 1 )
            {
            strideLengthVector <- c( strideLength, 1, strideLength )
            }
          } else if( dimensionsToPredict[d] == 3 ) {
          patchSizeVector <- c( numberOfChannels, patchSize, patchSize )
          if( length( strideLength ) == 1 )
            {
            strideLengthVector <- c( 1, strideLength, strideLength )
            }
          } else {
          stop( "dimensionsToPrediction should be 1, 2, and/or 3 for 3-D image." )
          }
        }

      patches <- extractImagePatches( evaluationImage, patchSizeVector,
            strideLength = strideLengthVector, returnAsArray = FALSE )

      patchesMOS <- list()
      patchesMOS.standardDeviation <- list()
      patchesContrast = list()
      patchesSharpness = list()
      batchX <- array( dim = c( length( patches ), c( patchSize, patchSize ), numberOfChannels ) )

      isGoodPatch <- rep( FALSE, length( patches ) )
      for( i in seq.int( length( patches ) ) )
        {
        if( var( as.numeric( patches[[i]] ) ) > 0 )
          {
          isGoodPatch[i] <- TRUE
          patchImage <- patches[[i]]
          patchImage <- patchImage - min( patchImage )
          if( max( patchImage ) > 0 )
            {
            if ( whichModel == "tidsQualityAssessment" ) patchImage <- patchImage / max( patchImage ) * 255
            if ( whichModel == "koniqMBCS" ) patchImage <- patchImage / max( patchImage ) * 2.0 - 1.0
            }
          if( image@dimension == 2 )
            {
            for( j in seq.int( numberOfChannels ) )
              {
              batchX[i,,,j] <- as.array( patchImage )
              }
            }
          if( image@dimension == 3 )
            {
            batchX[i,,,] <- aperm( as.array( patchImage ), permutations[[dimensionsToPredict[d]]] )
            }
          }
        }

      goodBatchX <- array( batchX[isGoodPatch,,,], dim = c( sum( isGoodPatch ), c( patchSize, patchSize ), numberOfChannels ) )
      predictedData <- predict( tidModel, goodBatchX, verbose = verbose )

      count <- 1
      for( i in seq.int( length( patches ) ) )
        {
        if( isGoodPatch[i] )
          {
          patchesMOS[[i]] <- patchImage * 0 + predictedData[count,1]
          patchesMOS.standardDeviation[[i]] <- patchImage * 0 + predictedData[count,2]
          if ( whichModel == "koniqMBCS" ) {
	    patchesContrast[[i]] = patchImage * 0 + predictedData[count,3]
	    patchesSharpness[[i]] = patchImage * 0 + predictedData[count,4]
            }
  	  count <- count + 1
          } else {
          patchesMOS[[i]] <- patchImage * 0
          patchesMOS.standardDeviation[[i]] <- patchImage * 0
	  if ( whichModel == "koniqMBCS" ) {
            patchesContrast[[i]] = patchImage * 0
	    patchesSharpness[[i]] = patchImage * 0
 	    }
          }
        }
      MOS <- MOS + padOrCropImageToSize(
        reconstructImageFromPatches( patchesMOS, evaluationImage,
                                     strideLength = strideLengthVector ), dim( image ) )
      MOS.standardDeviation <- MOS.standardDeviation + padOrCropImageToSize(
        reconstructImageFromPatches( patchesMOS.standardDeviation, evaluationImage,
                                     strideLength = strideLengthVector ), dim( image ) )
      if ( whichModel == "koniqMBCS" ) {
        contrast <- contrast + padOrCropImageToSize(
          reconstructImageFromPatches( patchesContrast, evaluationImage,
                                     strideLength = strideLengthVector ), dim( image ) )
        sharpness <- sharpness + padOrCropImageToSize(
          reconstructImageFromPatches( patchesSharpness, evaluationImage,
                                     strideLength = strideLengthVector ), dim( image ) )
        }

      }

    MOS <- MOS / length( dimensionsToPredict )
    MOS.standardDeviation <- MOS.standardDeviation / length( dimensionsToPredict )
    if ( whichModel == "koniqMBCS" ) {
      sharpness = sharpness / length( dimensionsToPredict )
      contrast = contrast / length( dimensionsToPredict )
    }
    if( missing( mask ) )
      {
      if ( whichModel == "tidsQualityAssessment" )
        return( list( MOS = MOS,
                    MOS.standardDeviation = MOS.standardDeviation,
                    MOS.mean = mean( MOS ),
                    MOS.standardDeviationMean = mean( MOS.standardDeviation ) ) )

      if ( whichModel == "koniqMBCS" )
        return( list( MOS = MOS,
                    brightness = MOS.standardDeviation, contrast=contrast, sharpness=sharpness,
                    MOS.mean = mean( MOS ),
                    brightness.mean = mean( MOS.standardDeviation ), contrast.mean=mean(contrast), sharpness.mean=mean(sharpness) ) )

      } else {
      if ( whichModel == "tidsQualityAssessment" )
        return( list( MOS = MOS * mask,
                    MOS.standardDeviation = MOS.standardDeviation * mask,
                    MOS.mean = mean( MOS[mask >= 0.5] ),
                    MOS.standardDeviationMean = mean( MOS.standardDeviation[mask >= 0.5] ) ) )
      if ( whichModel == "koniqMBCS" )
        return( list( MOS = MOS * mask,
                    brightness = MOS.standardDeviation * mask, contrast=contrast*mask, sharpness=sharpness*mask,
                    MOS.mean = mean( MOS[mask >= 0.5] ),
                    brightness.mean = mean( MOS.standardDeviation[mask >= 0.5] ), contrast.mean=mean(contrast[mask >= 0.5]), sharpness.mean=mean(sharpness[mask >= 0.5]) ) )

      }
    }
}
