#' Perform MOS-based assessment of an image.
#'
#' Use a ResNet architecture to estimate image quality in 2D or 3D using subjective
#' QC image databases described in
#'
#'    \url{https://www.sciencedirect.com/science/article/pii/S0923596514001490}
#'
#' or
#'
#'    \url{https://doi.org/10.1109/TIP.2020.2967829}
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
#' @param patchSize integer (prime) number for patch size; 101 is good. otherwise,
#' choose \code{"global"} for a single global estimate of quality.
#' @param strideLength optional value to speed up computation (typically less than
#' patch size).  Integer or vector of image dimension length.
#' @param paddingSize positive or negative integer (or vector of image dimension
#' length) for (de)padding to remove edge effects.
#' @param dimensionsToPredict if image dimension is 3, this parameter specifies
#' which dimension(s) should be used for prediction.  If more than one dimension
#' is specified, the results are averaged.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be reused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to
#' ~/.keras/ANTsXNet/.
#' @param whichModel model type e.g. string tidsQualityAssessment, koniqMS, koniqMS2 or koniqMS3 where
#' the former predicts mean opinion score (MOS) and MOS standard deviation and
#' the latter koniq models predict mean opinion score (MOS) and sharpness.  One
#' may also directly pass a tensorflow model here. In this case, we assume that
#' the input image is scaled by the \code{imageScaling} parameter.
#' @param imageScaling a two-vector where the first value is the multiplier and
#' the second value the subtractor so each image will be scaled as
#' \code{img = iMath(img,"Normalize")*m  - s }.
#' @param doPatchScaling boolean controlling whether each patch is scaled or
#  (if FALSE) only a global scaling of the image is used.
#' @param verbose print progress.
#' @return list of QC results predicting both both human rater's mean and standard
#' deviation of the MOS ("mean opinion scores") or sharpness depending on the
#' selected network.  Both aggregate and spatial scores are returned, the latter
#' in the form of an image.
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
  strideLength, paddingSize = 0L, dimensionsToPredict = 1L,
  antsxnetCacheDirectory = NULL, whichModel="tidsQualityAssessment",
  imageScaling,
  doPatchScaling = FALSE,
  verbose = FALSE )
{
  if ( missing( imageScaling ) )
    imageScaling = c(255,127.5)
  is.prime <- function( n )
    {
    return( n == 2L || all( n %% 2L:max( 2, floor( sqrt( n ) ) ) != 0 ) )
    }
  if ( class( whichModel )[1] != "character" ) {
     userModel = TRUE
     tidModel = whichModel
     whichModel = "userInput"
  } else {
    validModels <- c( "tidsQualityAssessment", "koniqMS", "koniqMS2", "koniqMS3", "userInput"  )
    if ( ! any( whichModel %in% validModels ) )
      {
      cat( validModels )
      stop( "Please pass valid model" )
      }

    if( is.null( antsxnetCacheDirectory ) )
      {
      antsxnetCacheDirectory <- "ANTsXNet"
      }

    if( verbose == TRUE )
      {
      cat( "Neural QA:  retreiving model and weights.\n" )
      }

    modelAndWeightsFileName <- getPretrainedNetwork( whichModel, antsxnetCacheDirectory = antsxnetCacheDirectory )
    tidModel <- load_model_hdf5( filepath = modelAndWeightsFileName, compile = FALSE )
    }

  isKoniq <- grepl( "koniq", whichModel )
  paddingSizeVector <- paddingSize
  if( length( paddingSize ) == 1 )
    {
    paddingSizeVector <- rep( paddingSize, image@dimension )
    }

  paddedImageSize <- dim( image ) + paddingSizeVector
  paddedImage <- padOrCropImageToSize(
    smoothImage(image,0.5,sigmaInPhysicalCoordinates=FALSE),
    paddedImageSize )

  numberOfChannels <- 3

  if( missing( strideLength ) & patchSize != "global" )
    {
    strideLength = round( min( patchSize ) / 2 )
    if( image@dimension == 3 )
      {
      strideLength <- c( strideLength, strideLength, 1 )
      }
    }

  ###############
  #
  #  Global
  #
  ###############

  if( patchSize == 'global' )
    {
    if( whichModel == "tidsQualityAssessment" )
      {
      evaluationImage <- paddedImage %>% iMath( "Normalize" ) * 255
      }

    if( isKoniq )
      {
      evaluationImage <- ( paddedImage %>% iMath( "Normalize" ) ) * 2.0 - 1.0
      }

    if( whichModel == "userInput" )
      {
      evaluationImage <- paddedImage %>% iMath( "Normalize" ) * imageScaling[1] - imageScaling[2]
      }


    if( image@dimension == 2 )
      {
      batchX <- array( dim = c( 1, dim( evaluationImage ), numberOfChannels ) )
      for( k in seq.int( 3 ) )
        {
        batchX[1,,,k] <- as.array( evaluationImage )
        }
      predictedData <- predict( tidModel, batchX, verbose = verbose )
      if( whichModel == "tidsQualityAssessment" )
        {
        return( list( MOS = NA,
                      MOS.standardDeviation = NA,
                      MOS.mean = predictedData[1, 1],
                      MOS.standardDeviationMean = predictedData[1, 2] ) )
        } else if( isKoniq ) {
        return( list(
                    MOS.mean = predictedData[1, 1],
                    sharpness.mean = predictedData[1, 2]
	           	    ) )
        }
        if ( whichModel == "userInput" )
          {
            return( list(
                        MOS.mean = predictedData[1, 1],
                        SD.mean = predictedData[1, 2]
    	           	    ) )
          }


      } else if( image@dimension == 3 ) {

      mosMean <- 0
      mosStandardDeviation <- 0

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
	      }
      mosMean <- mosMean / length( dimensionsToPredict )
      mosStandardDeviation <- mosStandardDeviation / length( dimensionsToPredict )
      if( whichModel == "tidsQualityAssessment" | whichModel == 'userInput')
        return( list( MOS.mean = mosMean,
                      MOS.standardDeviationMean = mosStandardDeviation ) )
      if ( isKoniq )
        return( list( MOS.mean = mosMean, sharpness.mean=mosStandardDeviation) )
      }
    } else {

    ###############
    #
    #  Patchwise
    #
    ###############

    evaluationImage <- paddedImage
    if( whichModel == "tidsQualityAssessment" )
      {
      evaluationImage <- paddedImage %>% iMath( "Normalize" ) * 255
      }

    if( isKoniq )
      {
      evaluationImage <- ( paddedImage %>% iMath( "Normalize" ) ) * 2.0 - 1.0
      }

    if( whichModel == "userInput" )
      {
      evaluationImage <- paddedImage %>% iMath( "Normalize" ) * imageScaling[1] - imageScaling[2]
      }


    if( ! is.prime( patchSize ) )
      {
#      message( "patch_size should be a prime number:  13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97..." )
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
          stop( "dimensionsToPredict elements should be 1, 2, and/or 3 for 3-D image." )
          }
        }

      patches <- extractImagePatches( evaluationImage, patchSize = patchSizeVector,
            strideLength = strideLengthVector, returnAsArray = FALSE )

      batchX <- array( dim = c( length( patches ), patchSize, patchSize, numberOfChannels ) )

      isGoodPatch <- rep( FALSE, length( patches ) )
      for( i in seq.int( length( patches ) ) )
        {
        if( var( as.numeric( patches[[i]] ) ) > 0 )
          {
          isGoodPatch[i] <- TRUE
          patchImage <- patches[[i]]
          if ( doPatchScaling ) patchImage <- patchImage - min( patchImage )
          if( max( patchImage ) > 0 & doPatchScaling )
            {
            if( whichModel == "tidsQualityAssessment" )
              {
              patchImage <- patchImage / max( patchImage ) * 255
              } else if( isKoniq ) {
              patchImage <- patchImage / max( patchImage ) * 2.0 - 1.0
            } else if( whichModel == "userInput" ) {
              patchImage <- patchImage / max( patchImage ) * imageScaling[1] - imageScaling[2]
              }
            }
          if( image@dimension == 2 )
            {
            for( j in seq.int( numberOfChannels ) )
              {
              batchX[i,,,j] <- as.array( patchImage )
              }
            } else if( image@dimension == 3 ) {
            batchX[i,,,] <- aperm( as.array( patchImage ), permutations[[dimensionsToPredict[d]]] )
            }
          }
        }

      # goodBatchX <- array( batchX[isGoodPatch,,,], dim = c( sum( isGoodPatch ), c( patchSize, patchSize ), numberOfChannels ) )
      goodBatchX <- batchX[isGoodPatch,,,]
      predictedData <- predict( tidModel, goodBatchX, verbose = verbose )

      patchesMOS <- list()
      patchesMOS.standardDeviation <- list()

      zeroPatchImage <- patchImage * 0

      count <- 1
      for( i in seq.int( length( patches ) ) )
        {
        if( isGoodPatch[i] )
          {
          patchesMOS[[i]] <- zeroPatchImage + predictedData[count,1]
          patchesMOS.standardDeviation[[i]] <- zeroPatchImage + predictedData[count,2]
  	      count <- count + 1
          } else {
          patchesMOS[[i]] <- zeroPatchImage
          patchesMOS.standardDeviation[[i]] <- zeroPatchImage
          }
        }
      MOS <- MOS + padOrCropImageToSize(
        reconstructImageFromPatches( patchesMOS, evaluationImage,
                                     strideLength = strideLengthVector ), dim( image ) )
      MOS.standardDeviation <- MOS.standardDeviation + padOrCropImageToSize(
        reconstructImageFromPatches( patchesMOS.standardDeviation, evaluationImage,
                                     strideLength = strideLengthVector ), dim( image ) )
      }

    MOS <- MOS / length( dimensionsToPredict )
    MOS.standardDeviation <- MOS.standardDeviation / length( dimensionsToPredict )
    if( missing( mask ) )
      {
      if ( whichModel == "tidsQualityAssessment" | whichModel == "userInput" )
        return( list( MOS = MOS,
                    MOS.standardDeviation = MOS.standardDeviation,
                    MOS.mean = mean( MOS ),
                    MOS.standardDeviationMean = mean( MOS.standardDeviation ) ) )

      if ( isKoniq )
        return( list(
          MOS = MOS,
          sharpness = MOS.standardDeviation,
          MOS.mean = mean( MOS ),
          sharpness.mean=mean(MOS.standardDeviation) ) )

      } else {
      if ( whichModel == "tidsQualityAssessment" | whichModel == "userInput" )
        return( list( MOS = MOS * mask,
                    MOS.standardDeviation = MOS.standardDeviation * mask,
                    MOS.mean = mean( MOS[mask != 0] ),
                    MOS.standardDeviationMean = mean( MOS.standardDeviation[mask != 0] ) ) )
      if ( isKoniq )
        return( list(
          MOS = MOS * mask,
          sharpness = MOS.standardDeviation * mask,
          MOS.mean = mean( MOS[mask != 0] ),
          sharpness.mean = mean( MOS.standardDeviation[mask != 0] )
        ) )

      }
    }
}
