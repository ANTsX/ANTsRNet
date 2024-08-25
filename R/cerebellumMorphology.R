#' Cerebellum tissue segmentation, Schmahmann parcellation, and thickness.
#'
#' Perform cerebellum segmentation using a modification of the set of
#' MaGET cerebellum atlases --- \url{https://www.cobralab.ca/cerebellum-lobules}.
#'
#' \url{https://www.nature.com/articles/s41598-024-59440-6}
#'
#' The tissue labeling is as follows:
#' \itemize{
#'   \item{Label 1 :}{CSF}
#'   \item{Label 2 :}{Gray matter}
#'   \item{Label 3 :}{White matter}
#' }
#'
#' The parcellation labeling is as follows:
#' \itemize{
#'   \item{Label 1  :}{L_I_II}
#'   \item{Label 2  :}{L_III}
#'   \item{Label 3  :}{L_IV}
#'   \item{Label 4  :}{L_V}
#'   \item{Label 5  :}{L_VI}
#'   \item{Label 6  :}{L_Crus_I}
#'   \item{Label 7  :}{L_Crus_II}
#'   \item{Label 8  :}{L_VIIB}
#'   \item{Label 9  :}{L_VIIIA}
#'   \item{Label 10 :}{L_VIIIB}
#'   \item{Label 11 :}{L_IX}
#'   \item{Label 12 :}{L_X}
#'   \item{Label 101:}{R_I_II}
#'   \item{Label 102:}{R_III}
#'   \item{Label 103:}{R_IV}
#'   \item{Label 104:}{R_V}
#'   \item{Label 105:}{R_VI}
#'   \item{Label 106:}{R_Crus_I}
#'   \item{Label 107:}{R_Crus_II}
#'   \item{Label 108:}{R_VIIB}
#'   \item{Label 109:}{R_VIIIA}
#'   \item{Label 110:}{R_VIIIB}
#'   \item{Label 111:}{R_IX}
#'   \item{Label 112:}{R_X}
#' }
#'
#' @param t1 raw or preprocessed 3-D T1-weighted whole head image.
#' @param cerebellumMask Option for initialization.  If not specified, the
#' cerebellum ROI is determined using ANTsXNet brain_extraction followed by
#' registration to a template.
#' @param computeThicknessImage Compute KellyKapowski thickness image of the gray
#' matter.
#' @param doPreprocessing Perform N4 bias correction and spatiall normalize to template space.
#' @param verbose print progress.
#' @return List consisting of the multiple segmentation images and probability
#' images for each label and foreground.  Optional thickness image.
#' @author Tustison NJ, Tustison MG
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' cereb <- cerebellumMorphology( image )
#' }
#' @export
cerebellumMorphology <- function( t1, cerebellumMask = NULL,
  computeThicknessImage = FALSE, doPreprocessing = TRUE,
  verbose = FALSE )
{
  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  transformType <- "antsRegistrationSyNQuick[s]"
  whichtoinvert <- c( TRUE, FALSE, TRUE )
  # transformType <- "antsRegistrationSyNQuick[a]"
  # whichtoinvert <- c( TRUE, TRUE )

  ################################
  #
  # Get the templates, masks, priors
  #
  ################################

  t1Template <- antsImageRead( getANTsXNetData( "magetTemplate" ) )
  t1TemplateBrainMask <- antsImageRead( getANTsXNetData( "magetTemplateBrainMask" ) )
  t1TemplateBrain <- t1Template * t1TemplateBrainMask
  t1CerebellumTemplate <- antsImageRead( getANTsXNetData( "magetCerebellumTemplate" ) )
  t1CerebellumTemplate <- iMath( t1CerebellumTemplate, "Normalize" )
  cerebellumxTemplateXfrm <- getANTsXNetData( "magetCerebellumxTemplate0GenericAffine" )

  # spatial priors are in the space of the cerebellar template.  First three are
  # csf, gm, and wm followed by the regions.
  spatialPriorsFileNamePath <- getANTsXNetData( "magetCerebellumTemplatePriors" )
  spatialPriors <- antsImageRead( spatialPriorsFileNamePath )
  # priorsImageList <- splitNDImageToList( spatialPriors )
  spatialPriorsArray <- as.array( spatialPriors )
  priorsImageList <- list()
  for( i in seq.int( dim( spatialPriorsArray )[[4]] ) )
    {
    priorsImageList[[i]] <- as.antsImage( spatialPriorsArray[,,,i], reference = t1CerebellumTemplate )
    }

  ################################
  #
  # Preprocess images
  #
  ################################

  t1Preprocessed <- antsImageClone( t1 )
  t1Mask <- NULL

  templateTransforms <- NULL
  if( doPreprocessing )
    {
    if( verbose )
      {
      cat( "Preprocessing T1.\n" )
      }
    # Do bias correction
    t1Preprocessed <- n4BiasFieldCorrection( t1Preprocessed, shrinkFactor = 4, verbose = verbose )
    }

  if( is.null( cerebellumMask ) )
    {
    # Brain extraction
    probabilityMask <- brainExtraction( t1Preprocessed, modality = "t1" )
    t1Mask <- thresholdImage( probabilityMask, 0.5, 1, 1, 0 )
    t1BrainPreprocessed <- t1Preprocessed * t1Mask

    # Warp to template and concatenate with cerebellum x template transform
    if( verbose )
      {
      cat( "Register T1 to whole brain template.\n" )
      }

    registration <- antsRegistration( fixed = t1TemplateBrain, moving = t1BrainPreprocessed,
        typeofTransform = transformType, verbose = verbose )
    registration$invtransforms <- append( registration$invtransforms, cerebellumxTemplateXfrm )
    registration$fwdtransforms <- append( registration$fwdtransforms, cerebellumxTemplateXfrm, 0 )
    templateTransforms <- list( fwdtransforms = registration$fwdtransforms,
                                invtransforms = registration$invtransforms )
    } else {
    t1CerebellumTemplateMask <- thresholdImage( t1CerebellumTemplate, -0.01, 100, 0, 1 )
    t1CerebellumTemplateMask <- antsApplyTransforms( t1Template, t1CerebellumTemplateMask,
                                                     transformlist = cerebellumxTemplateXfrm,
                                                     interpolator = 'nearestNeighbor',
                                                     whichtoinvert = c( TRUE ) )
    if( verbose )
      {
      cat( "Register T1 cerebellum to the cerebellum of the whole brain template.\n" )
      }

    registration <- antsRegistration(fixed = t1TemplateBrain * t1CerebellumTemplateMask,
                                     moving = t1Preprocessed * cerebellumMask,
                                     typeofTransform  = transformType, verbose = verbose )
    registration$invtransforms <- append( registration$invtransforms, cerebellumxTemplateXfrm )
    registration$fwdtransforms <- append( registration$fwdtransforms, cerebellumxTemplateXfrm, 0 )
    templateTransforms <- list( fwdtransforms = registration$fwdtransforms,
                                invtransforms = registration$invtransforms )
    }

  t1PreprocessedInCerebellumSpace <- antsApplyTransforms(t1CerebellumTemplate, t1Preprocessed,
                                                         transformlist = registration$fwdtransforms )
  t1PreprocessedMaskInCerebellumSpace <- NULL
  if( ! is.null( cerebellumMask ) )
    {
    t1PreprocessedMaskInCerebellumSpace <- antsApplyTransforms(t1CerebellumTemplate, cerebellumMask,
                                                               transformlist = registration$fwdtransforms )
    }

  ################################
  #
  # Create models, do prediction, and normalize to original t1 space
  #
  ################################

  tissueLabels <- c( 0, 1, 2, 3 )
  regionLabels <- c( 0, seq.int( 1, 12 ), seq.int( 101, 112 ) )

  imageSize <- c( 240, 144, 144 )

  cerebellumProbabilityImage <- NULL
  tissueProbabilityImages <- list()
  regionProbabilityImages <- list()
  whichPriors <- NULL

  startM <- 1
  if( ! is.null( cerebellumMask ) )
    {
    startM <- 2
    cerebellumProbabilityImage <- antsImageClone( cerebellumMask )
    }
  for( m in seq.int( startM, 3 ) )
    {
    if( m == 1 )
      {
      labels <- c( 0, 1 )
      channelSize <- 2
      whichPriors <- NULL
      networkName <- "cerebellumWhole"
      additionalOptions <- c( "attentionGating" )
      } else if( m == 2 ) {
      labels <- tissueLabels
      channelSize <- length( labels )
      whichPriors <- c( 1, 2, 3 )
      networkName <- "cerebellumTissue"
      additionalOptions <- NA
      } else {
      labels <- regionLabels
      channelSize <- length( labels )
      whichPriors <- c( seq.int( 4, 15 ), seq.int( 17, 28 ) )
      networkName <- "cerebellumLabels"
      additionalOptions <- c( "attentionGating" )
      }

    numberOfClassificationLabels <- length( labels )
    unetModel <- createUnetModel3D( c( imageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = "classification",
      numberOfFilters = c( 32, 64, 96, 128, 256 ),
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      dropoutRate = 0.0, weightDecay = 0, additionalOptions = additionalOptions )

    if( verbose )
      {
      cat( "Processing ", networkName, "\n" )
      }

    ################################
    #
    # Load weights
    #
    ################################

    if( verbose )
      {
      cat( "Retrieving model weights.\n" )
      }

    weightsFileName <- getPretrainedNetwork( networkName )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose )
      {
      cat( "Prediction.\n" )
      }

    if( m > 1 )
      {
      t1PreprocessedInCerebellumSpace <- t1PreprocessedInCerebellumSpace * t1PreprocessedMaskInCerebellumSpace
      }

    t1PreprocessedInCerebellumSpace <- iMath( t1PreprocessedInCerebellumSpace, "Normalize" )

    batchX <- array( data = 0, dim = c( 2, imageSize, channelSize ) )
    batchX[1,,,,1] <- as.array( padOrCropImageToSize( t1PreprocessedInCerebellumSpace, imageSize ) )
    batchX[2,,,,1] <- batchX[1,imageSize[1]:1,,,1]

    if( m == 1 )
      {
      for( j in seq.int( dim( batchX )[1] ) )
        {
        batchX[j,,,,2] <- as.array( padOrCropImageToSize( t1CerebellumTemplate, imageSize ) )
        }
      }
    if( m > 1 )
      {
      for( i in seq.int( length( whichPriors ) ) )
        {
        for( j in seq.int( dim( batchX )[1] ) )
          {
          batchX[j,,,,i+1] <- as.array( padOrCropImageToSize( priorsImageList[[whichPriors[i]]], imageSize ) )
          }
        }
      }

    predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    decropToCerebellumTemplateSpace <- function( targetImage, referenceImage )
      {
      targetImagePadded <- padOrCropImageToSize( targetImage, dim( referenceImage ) )
      targetImagePadded <- iMath( targetImagePadded, "PadImage", 1 )
      lowerIndices <- c( 3, 3, 2 )
      upperIndices <- dim( referenceImage )
      targetImageDecropped <- cropIndices( targetImagePadded, lowerIndices, upperIndices )
      targetImageDecropped <- antsCopyImageInfo( referenceImage, targetImageDecropped )
      identityXfrm <- createAntsrTransform( type = "Euler3DTransform" )
      targetImageDecropped <- applyAntsrTransformToImage( identityXfrm, targetImageDecropped, referenceImage, interpolation = "linear" )
      return( targetImageDecropped )
      }

    if( m == 1 )
      {
      # whole cerebellum
      probabilityImage <- as.antsImage( 0.5 * ( predictedData[1,,,,2] +
                                        predictedData[2,dim( predictedData )[2]:1,,,2] ) )
      probabilityImage <- decropToCerebellumTemplateSpace( probabilityImage, t1CerebellumTemplate )
      t1PreprocessedMaskInCerebellumSpace <- thresholdImage( probabilityImage, 0.5, 1, 1, 0 )

      probabilityImage <- antsApplyTransforms( fixed = t1,
          moving = probabilityImage,
          transformlist = templateTransforms$invtransforms,
          whichtoinvert = whichtoinvert, interpolator = "linear", verbose = verbose )
      cerebellumProbabilityImage <- probabilityImage
      } else if( m == 2 ) {

      # tissue labels
      for( i in seq.int( length( tissueLabels ) ) )
        {
        probabilityImage <- as.antsImage( 0.5 * ( predictedData[1,,,,i] +
                                          predictedData[2,dim( predictedData )[2]:1,,,i] ) )
        probabilityImage <- decropToCerebellumTemplateSpace( probabilityImage, t1CerebellumTemplate )
        probabilityImage <- antsApplyTransforms( fixed = t1,
            moving = probabilityImage,
            transformlist = templateTransforms$invtransforms,
            whichtoinvert = whichtoinvert, interpolator = "linear", verbose = verbose )
        tissueProbabilityImages[[i]] <- probabilityImage
        }
      } else {

      for( i in seq.int( 2, 13 ) )
        {
        tmpArray <- predictedData[2,,,,i]
        predictedData[2,,,,i] <- predictedData[2,,,,i+12]
        predictedData[2,,,,i+12] <- tmpArray
        }

      # region labels
      for( i in seq.int( length( regionLabels ) ) )
        {
        probabilityImage <- as.antsImage( 0.5 * ( predictedData[1,,,,i] +
                                          predictedData[2,dim( predictedData )[2]:1,,,i] ) )
        probabilityImage <- decropToCerebellumTemplateSpace( probabilityImage, t1CerebellumTemplate )
        probabilityImage <- antsApplyTransforms( fixed = t1,
            moving = probabilityImage,
            transformlist = templateTransforms$invtransforms,
            whichtoinvert = whichtoinvert, interpolator = "linear", verbose = verbose )
        regionProbabilityImages[[i]] <- probabilityImage
        }
      }
    }

  ################################
  #
  # Convert probability images to segmentations
  #
  ################################

  # region labels

  probabilityImages <- regionProbabilityImages
  labels <- regionLabels

  imageMatrix <- imageListToMatrix( probabilityImages[2:length( probabilityImages )], t1 * 0 + 1 )
  backgroundForegroundMatrix <- rbind( imageListToMatrix( list( probabilityImages[[1]] ), t1 * 0 + 1 ),
                                      colSums( imageMatrix ) )
  foregroundMatrix <- matrix( apply( backgroundForegroundMatrix, 2, which.max ), nrow = 1 ) - 1
  segmentationMatrix <- ( matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) + 1 ) * foregroundMatrix
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]]

  relabeledImage <- antsImageClone( segmentationImage )
  for( i in seq.int( length( labels ) ) )
    {
    relabeledImage[( segmentationImage == i )] <- labels[i]
    }
  regionSegmentation <- antsImageClone( relabeledImage )

  # tissue labels

  probabilityImages <- tissueProbabilityImages
  labels <- tissueLabels

  imageMatrix <- imageListToMatrix( probabilityImages[2:length( probabilityImages )], t1 * 0 + 1 )
  backgroundForegroundMatrix <- rbind( imageListToMatrix( list( probabilityImages[[1]] ), t1 * 0 + 1 ),
                                      colSums( imageMatrix ) )
  foregroundMatrix <- matrix( apply( backgroundForegroundMatrix, 2, which.max ), nrow = 1 ) - 1
  segmentationMatrix <- ( matrix( apply( imageMatrix, 2, which.max ), nrow = 1 ) + 1 ) * foregroundMatrix
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]]

  relabeledImage <- antsImageClone( segmentationImage )
  for( i in seq.int( length( labels ) ) )
    {
    relabeledImage[( segmentationImage == i )] <- labels[i]
    }
  tissueSegmentation <- antsImageClone( relabeledImage )

  if( computeThicknessImage )
    {
    ################################
    #
    # Compute thickness image using KK
    #
    ################################

    kk = kellyKapowski( s = tissueSegmentation,
                        g = tissueProbabilityImages[[3]],
                        w = tissueProbabilityImages[[4]],
                        its = 45,
                        r = 0.025,
                        m = 1.5,
                        x = 0,
                        verbose = verbose )

    results <- list( cerebellumProbabilityImage = cerebellumProbabilityImage,
                     parcellationSegmentationImage = regionSegmentation,
                     parcellationProbabilityImages = regionProbabilityImages,
                     tissueSegmentationImage = tissueSegmentation,
                     tissueProbabilityImages = tissueProbabilityImages,
                     thicknessImage = kk
                  )
    return( results )
    } else {
    results <- list( cerebellumProbabilityImage = cerebellumProbabilityImage,
                     parcellationSegmentationImage = regionSegmentation,
                     parcellationProbabilityImages = regionProbabilityImages,
                     tissueSegmentationImage = tissueSegmentation,
                     tissueProbabilityImages = tissueProbabilityImages
                  )
    return( results )
    }
}

