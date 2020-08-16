#' Perform TID-based quality assessment of an image.
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
#' template and model weights.  Since these can be resused, if
#' \code{is.null(outputDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return list of QC results predicting both both human rater's mean and standard
#' deviation of the MOS ("mean opinion scores").  See the TID2013 paper.  Both
#' aggregate and spatial scores are returned, the latter in the form of an image.
#' @author Avants BB
#' @examples
#' \dontrun{
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' mask <- getMask( image )
#' tid <- tidQualityAssessment( image, mask = mask, patchSize = 101L,
#'           strideLength = 7L, paddingSize = 0L )
#' plot( image, tid$MOS, alpha = 0.5)
#' cat( "mean MOS = ", tid$MOS.mean, "\n" )
#' cat( "sd MOS = ", tid$MOS.standardDeviationMean, "\n" )
#' }
#' @export
tidQualityAssessment <- function( image, mask, patchSize = 101L,
  strideLength = 7L, paddingSize = 0L, dimensionsToPredict = 1,
  outputDirectory = NULL, verbose = FALSE )
{
  is.prime <- function( n )
    {
    return( n == 2L || all( n %% 2L:max( 2, floor( sqrt( n ) ) ) != 0 ) )
    }

  if( is.null( outputDirectory ) )
    {
    outputDirectory <- system.file( "extdata", package = "ANTsRNet" )
    }
  modelAndWeightsFileName <- paste0( outputDirectory, "tidsQualityAssessment.h5" )
  if( ! file.exists( modelAndWeightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "TID QA:  downloading model and weights.\n" )
      }
    modelAndWeightsFileName <- getPretrainedNetwork( "tidsQualityAssessment", modelAndWeightsFileName )
    }
  tidModel <- load_model_hdf5( filepath = modelAndWeightsFileName )

  paddingSizeVector <- paddingSize
  if( length( paddingSize ) == 1 )
    {
    paddingSizeVector <- rep( paddingSize, image@dimension )
    }

  paddedImageSize <- dim( image ) + paddingSizeVector
  paddedImage <- padOrCropImageToSize( image, paddedImageSize )
  evaluationImage <- paddedImage %>% iMath( "Normalize" ) * 255

  numberOfChannels <- 3

  ###############
  #
  #  Global
  #
  ###############

  if( patchSize == 'global' )
    {
    if( image@dimension == 2 )
      {
      batchX <- array( dim = c( 1, dim( evaluationImage ), numberOfChannels ) )
      for( k in seq.int( 3 ) )
        {
        batchX[1,,,k] <- as.array( evaluationImage )
        }
      predictedData <- predict( tidModel, batchX, verbose = verbose )

      return( list( MOS = NA,
                    MOS.standardDeviation = NA,
                    MOS.mean = predictedData[1, 1],
                    MOS.standardDeviationMean = predictedData[1, 2] ) )
      } else if( image@dimension == 3 ) {

      mosMean <- 0
      mosStandardDeviation <- 0

      x <- seq.int( image@dimension )
      for( d in seq.int( length( dimensionsToPredict ) ) )
        {
        batchX <- array( data = 0,
          c( paddedImageSize[dimensionsToPredict[d]], x[!x %in% dimensionsToPredict[d]], numberOfChannels ) )

        batchX[1,,,1] <- extractSlice( evaluationImage, 1, dimensionsToPredict[d] )
        batchX[1,,,2] <- extractSlice( evaluationImage, 1, dimensionsToPredict[d] )
        batchX[1,,,3] <- extractSlice( evaluationImage, 2, dimensionsToPredict[d] )
        for( i in seq.int( from = 2, to = paddedImageSize[dimensionsToPredict[d]] - 1 ) )
          {
          batchX[i,,,1] <- extractSlice( evaluationImage, i - 1, dimensionsToPredict[d] )
          batchX[i,,,2] <- extractSlice( evaluationImage, i    , dimensionsToPredict[d] )
          batchX[i,,,3] <- extractSlice( evaluationImage, i + 1, dimensionsToPredict[d] )
          }
        batchX[paddedImageSize[dimensionsToPredict[d]],,,1] <-
          extractSlice( evaluationImage, paddedImageSize[dimensionsToPredict[d]] - 1, dimensionsToPredict[d] )
        batchX[paddedImageSize[dimensionsToPredict[d]],,,2] <-
          extractSlice( evaluationImage, paddedImageSize[dimensionsToPredict[d]], dimensionsToPredict[d] )
        batchX[paddedImageSize[dimensionsToPredict[d]],,,3] <-
          extractSlice( evaluationImage, paddedImageSize[dimensionsToPredict[d]], dimensionsToPredict[d] )

        predictedData <- predict( tidModel, batchX, verbose = verbose )

        mosMean <- mosMean + predictedData[1, 1]
        mosStandardDeviation <- mosStandardDeviation + predictedData[1, 2]
        }
      mosMean <- mosMean / length( dimensionsToPredict )
      mosStandardDeviation <- mosStandardDeviation / length( dimensionsToPredict )

      return( list( MOS.mean = mosMean,
                    MOS.standardDeviationMean = mosStandardDeviation ) )
      }
    } else {

    ###############
    #
    #  Patchwise
    #
    ###############

    if( ! is.prime( patchSize ) )
      {
      stop( "Should pass a prime number for patch size." )
      }
    strideLengthVector <- strideLength
    if( length( strideLength ) == 1 )
      {
      strideLengthVector <- rep( strideLength, image@dimension )
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
          } else if( dimensionsToPredict[d] == 2 ) {
          patchSizeVector <- c( patchSize, numberOfChannels, patchSize )
          } else if( dimensionsToPredict[d] == 3 ) {
          patchSizeVector <- c( numberOfChannels, patchSize, patchSize )
          }
        }

      patches <- extractImagePatches( evaluationImage, patchSizeVector,
            strideLength = strideLength, returnAsArray = FALSE )

      patchesMOS <- list()
      patchesMOS.standardDeviation <- list()
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
            patchImage <- patchImage / max( patchImage ) * 255
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

      goodBatchX <- array( batchX[isGoodPatch,,,], dim = c( sum( isGoodPatch ), patchSizeVector, numberOfChannels ) )
      predictedData <- predict( tidModel, goodBatchX, verbose = verbose )

      count <- 1
      for( i in seq.int( length( patches ) ) )
        {
        if( isGoodPatch[i] )
          {
          patchesMOS[[i]] <- patchImage * 0 + predictedData[count,1]
          patchesMOS.standardDeviation[[i]] <- patchImage * 0 + predictedData[count,2]
          count <- count + 1
          } else {
          patchesMOS[[i]] <- patchImage * 0
          patchesMOS.standardDeviation[[i]] <- patchImage * 0
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
      return( list( MOS = reconMOS,
                    MOS.standardDeviation = reconMOS.standardDeviation,
                    MOS.mean = mean( MOS ),
                    MOS.standardDeviationMean = mean( MOS.standardDeviation ) ) )
      } else {
      return( list( MOS = reconMOS * mask,
                    MOS.standardDeviation = reconMOS.standardDeviation * mask,
                    MOS.mean = mean( MOS[mask >= 0.5] ),
                    MOS.standardDeviationMean = mean( MOS.standardDeviation[mask >= 0.5] ) ) )
      }
    }
}


