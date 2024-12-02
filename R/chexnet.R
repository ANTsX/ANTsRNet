#' Check x-ray lung orientation.
#'
#' Check the correctness of image orientation, i.e., flipped left-right, up-down,
#' or both.  If True, attempts to correct before returning corrected image.  Otherwise
#' it returns NULL.
#'
#' @param image input 3-D lung image.
#' @param verbose print progress.
#' @return segmentation and probability images
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' }
#' @import keras
#' @export
checkXrayLungOrientation <- function( image,
                                      verbose = FALSE )
{
  if( image@dimension != 2 )
    {
    stop( "ImageDimension must be equal to 2.")
    }

  resampledImageSize <- c( 224, 224 )
  if( any( dim( image ) != resampledImageSize ) )
    {
    if( verbose )
      {
      cat( "Resampling image to ", resampledImageSize, "\n" )
      }
    resampledImage <- resampleImage( image, resampledImageSize, useVoxels = TRUE )
    }
  else
    {
    resampledImage <- antsImageClone( image )
    }

  model <- createResNetModel2D( c( resampledImageSize, 1 ),
                                numberOfOutputs = 3,
                                mode = "classification",
                                layers = c( 1, 2, 3, 4 ),
                                residualBlockSchedule = c( 2, 2, 2, 2 ),
                                lowestResolution = 64,
                                cardinality = 1,
                                squeezeAndExcite = FALSE )

  weightsFileName <- getPretrainedNetwork( "xrayLungOrientation" )
  model$load_weights( weightsFileName )

  imageMin <- ANTsRCore::min( resampledImage )
  imageMax <- ANTsRCore::max( resampledImage )
  normalizedImage <- antsImageClone( resampledImage )
  normalizedImage <- ( normalizedImage - imageMin ) / ( imageMax - imageMin )

  batchX <- array( data = as.array( normalizedImage ), dim = c( 1, resampledImageSize, 1 ) )
  batchY <- model %>% predict( batchX, verbose = verbose )

  # batchY is a 3-element array:
  #   batchY[0] = Pr(image is correctly oriented)
  #   batchY[1] = Pr(image is flipped up/down)
  #   batchY[2] = Pr(image is flipped left/right)

  if( batchY[1, 1] > 0.5 )
    {
    return( NULL )
    } else {
    if( verbose )
      {
      message( "Possible incorrect orientation.  Attempting to correct." )
      }
    normalizedImageArray <- as.array( normalizedImage )
    imageUpDown <- as.antsImage( normalizedImageArray[,dim( normalizedImageArray )[2]:1],
                                 origin = antsGetOrigin( resampledImage ),
                                 spacing = antsGetSpacing( resampledImage ),
                                 direction = antsGetDirection( resampledImage ) )
    imageLeftRight <- as.antsImage( normalizedImageArray[dim( normalizedImageArray )[1]:1,],
                                 origin = antsGetOrigin( resampledImage ),
                                 spacing = antsGetSpacing( resampledImage ),
                                 direction = antsGetDirection( resampledImage ) )
    imageBoth <- as.antsImage( ( normalizedImageArray[,dim( normalizedImageArray )[2]:1] )[dim( normalizedImageArray )[1]:1,],
                                 origin = antsGetOrigin( resampledImage ),
                                 spacing = antsGetSpacing( resampledImage ),
                                 direction = antsGetDirection( resampledImage ) )

    batchX <- array( data = 0, dim = c( 3, resampledImageSize, 1 ) )
    batchX[1,,,1] <- as.array( imageUpDown )
    batchX[2,,,1] <- as.array( imageLeftRight )
    batchX[3,,,1] <- as.array( imageBoth )

    batchY <- model %>% predict( batchX, verbose = verbose )

    imageArray <- as.array( image )
    orientedImage <- NULL
    if( batchY[1, 1] > batchY[2, 1] && batchY[1, 1] > batchY[3, 1] )
      {
      if( verbose )
        {
        message( "Image is flipped up-down." )
        }
      orientedImage <- as.antsImage( imageArray[,dim( imageArray )[2]:1],
                                 origin = antsGetOrigin( image ),
                                 spacing = antsGetSpacing( image ),
                                 direction = antsGetDirection( image ) )
      } else if( batchY[2, 1] > batchY[1, 1] && batchY[2, 1] > batchY[3, 1] ) {
      if( verbose )
        {
        message( "Image is flipped right-left." )
        }
      orientedImage <- as.antsImage( imageArray[dim( imageArray )[1]:1,],
                                 origin = antsGetOrigin( image ),
                                 spacing = antsGetSpacing( image ),
                                 direction = antsGetDirection( image ) )
      } else {
      if( verbose )
        {
        message( "Image is flipped up-down and right-left." )
        }
      orientedImage <- as.antsImage( ( imageArray[,dim( imageArray )[2]:1] )[dim( imageArray )[1]:1,],
                                 origin = antsGetOrigin( image ),
                                 spacing = antsGetSpacing( image ),
                                 direction = antsGetDirection( image ) )
      }

    return( orientedImage )
    }
}


#' CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
#'
#' ANTsXNet reproduction of https://arxiv.org/pdf/1711.05225.pdf.  This includes
#' our own network architecture and training (including data augmentation).
#'
#' Includes a network for checking the correctness of image orientation, i.e.,
#' flipped left-right, up-down, or both.
#'
#' Disease categories:
#'        Atelectasis
#'        Cardiomegaly
#'        Consolidation
#'        Edema
#'        Effusion
#'        Emphysema
#'        Fibrosis
#'        Hernia
#'        Infiltration
#'        Mass
#'        No Finding
#'        Nodule
#'        Pleural Thickening
#'        Pneumonia
#'        Pneumothorax
#'
#' @param image input 3-D lung image.
#' @param lungMask ANTsImage.  If None is specified, one is estimated.
#' @param checkImageOrientation Check the correctness of image orientation, i.e.,
#' flipped left-right, up-down, or both.  If TRUE, attempts to correct before prediction.
#' @param useANTsXNetVariant Use an extension of the original chexnet approach
#' by adding a left/right lung masking and including those two masked regions
#' in the red and green channels, respectively.
#' @param includeTuberculosisDiagnosis Include the output of an additional network
#' trained on TB data but using the ANTsXNet variant chexnet data as the initial
#' weights.
#' @param verbose print progress.
#' @return classification scores for each of the 14 or 15 categories.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' }
#' @import keras
#' @export
chexnet <- function( image,
                     lungMask = NULL,
                     checkImageOrientation = FALSE,
                     useANTsXNetVariant = TRUE,
                     includeTuberculosisDiagnosis = FALSE,
                     verbose = FALSE )
{
  if( image@dimension != 2 )
    {
    stop( "ImageDimension must be equal to 2.")
    }

  if( checkImageOrientation )
    {
    image <- checkXrayLungOrientation( image )
    }

  diseaseCategories <- c( 'Atelectasis',
                          'Cardiomegaly',
                          'Effusion',
                          'Infiltration',
                          'Mass',
                          'Nodule',
                          'Pneumonia',
                          'Pneumothorax',
                          'Consolidation',
                          'Edema',
                          'Emphysema',
                          'Fibrosis',
                          'Pleural_Thickening',
                          'Hernia' )

  ################################
  #
  # Resample to image size
  #
  ################################

  imageSize <- c( 224, 224 )
  if( any( dim( image ) != imageSize ) )
    {
    if( verbose )
      {
      cat( "Resampling image to ", imageSize, "\n" )
      }
    resampledImage <- resampleImage( image, imageSize, useVoxels = TRUE )
    }
  else
    {
    resampledImage <- antsImageClone( image )
    }

  if( is.null( lungMask ) )
    {
    if( verbose )
      {
      cat( "No lung mask provided.  Estimating using antsxnet." )
      }
    lungExtract <- lungExtraction( image, modality = "xray",
                                   verbose = verbose )
    lungMask <- lungExtract$segmentationImage
    }

  if( any( dim( lungMask ) != imageSize ) )
    {
    resampledLungMask <- resampleImage( lungMask, imageSize, useVoxels = TRUE, interpType = 1 )
    } else {
    resampledLungMask <- antsImageClone( lungMask )
    }

  # use imagenet mean,std for normalization
  imagenetMean <- c( 0.485, 0.456, 0.406 )
  imagenetStd <- c( 0.229, 0.224, 0.225 )

  numberOfChannels <- 3

  tbPrediction <- NULL
  if( includeTuberculosisDiagnosis )
    {
    modelFileName <- getPretrainedNetwork( "tb_antsxnet_model" )
    model <- tensorflow::tf$keras$models$load_model( modelFileName, compile = FALSE )

    batchX <- array( data = 0, dim = c( 1, imageSize, numberOfChannels ) )
    imageArray <- as.array( resampledImage )
    imageArray <- ( imageArray - min( imageArray ) ) / ( max( imageArray ) - min( imageArray ) )

    batchX[1,,,1] <- ( imageArray - imagenetMean[1] ) / ( imagenetStd[1] )
    batchX[1,,,2] <- ( imageArray - imagenetMean[2] ) / ( imagenetStd[2] )
    batchX[1,,,2] <- batchX[1,,,2] * as.array( thresholdImage( resampledLungMask, 1, 1, 1, 0 ) )
    batchX[1,,,3] <- ( imageArray - imagenetMean[3] ) / ( imagenetStd[3] )
    batchX[1,,,3] <- batchX[1,,,3] * as.array( thresholdImage( resampledLungMask, 2, 2, 1, 0 ) )

    batchY <- model %>% predict( batchX, verbose = verbose )
    tbPrediction <- batchY[1]
    }

  if( ! useANTsXNetVariant )
    {
    modelFileName <- getPretrainedNetwork( "chexnetClassificationModel" )
    model <- tensorflow::tf$keras$models$load_model( modelFileName, compile = FALSE )

    batchX <- array( data = 0, dim = c( 1, imageSize, numberOfChannels ) )
    imageArray <- as.array( resampledImage )
    imageArray <- ( imageArray - min( imageArray ) ) / ( max( imageArray ) - min( imageArray ) )
    for( c in seq.int( numberOfChannels ) )
      {
      batchX[1,,,c] <- ( imageArray - imagenetMean[c] ) / ( imagenetStd[c] )
      }

    batchY <- model %>% predict( batchX, verbose = verbose )
    diseaseCategoryDf = data.frame( batchY )
    colnames( diseaseCategoryDf ) <- diseaseCategories
    if( includeTuberculosisDiagnosis )
      {
      diseaseCategoryDf$Tuberculosis <- c( tbPrediction )
      }
    return( diseaseCategoryDf )

    } else {
    modelFileName <- getPretrainedNetwork( "chexnetClassificationANTsXNetModel" )
    model <- tensorflow::tf$keras$models$load_model( modelFileName, compile = FALSE )

    batchX <- array( data = 0, dim = c( 1, imageSize, numberOfChannels ) )
    imageArray <- as.array( resampledImage )
    imageArray <- ( imageArray - min( imageArray ) ) / ( max( imageArray ) - min( imageArray ) )

    batchX[1,,,1] <- ( imageArray - imagenetMean[1] ) / ( imagenetStd[1] )
    batchX[1,,,2] <- ( imageArray - imagenetMean[2] ) / ( imagenetStd[2] )
    batchX[1,,,2] <- batchX[1,,,2] * as.array( thresholdImage( resampledLungMask, 1, 1, 1, 0 ) )
    batchX[1,,,3] <- ( imageArray - imagenetMean[3] ) / ( imagenetStd[3] )
    batchX[1,,,3] <- batchX[1,,,3] * as.array( thresholdImage( resampledLungMask, 2, 2, 1, 0 ) )

    batchY <- model %>% predict( batchX, verbose = verbose )
    diseaseCategoryDf = data.frame( batchY )
    colnames( diseaseCategoryDf ) <- diseaseCategories
    if( includeTuberculosisDiagnosis )
      {
      diseaseCategoryDf$Tuberculosis <- c( tbPrediction )
      }
    return( diseaseCategoryDf )
    }
}
