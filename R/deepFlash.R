#' Hippocampal/Enthorhinal segmentation using "Deep Flash"
#'
#' Perform hippocampal/entorhinal segmentation in T1 images using
#' labels from Mike Yassa's lab
#'
#' \url{https://faculty.sites.uci.edu/myassa/}
#'
#' The labeling is as follows:
#' \itemize{
#'   \item{Label 0 :}{background}
#'   \item{Label 5 :}{left aLEC}
#'   \item{Label 6 :}{right aLEC}
#'   \item{Label 7 :}{left pMEC}
#'   \item{Label 8 :}{right pMEC}
#'   \item{Label 9 :}{left perirhinal}
#'   \item{Label 10:}{right perirhinal}
#'   \item{Label 11:}{left parahippocampal}
#'   \item{Label 12:}{right parahippocampal}
#'   \item{Label 13:}{left DG/CA3}
#'   \item{Label 14:}{right DG/CA3}
#'   \item{Label 15:}{left CA1}
#'   \item{Label 16:}{right CA1}
#'   \item{Label 17:}{left subiculum}
#'   \item{Label 18:}{right subiculum}
#' }
#'
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * affine registration to deep flash template.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param t1 raw or preprocessed 3-D T1-weighted brain image.
#' @param t2 optional raw or preprocessed 3-D T2-weighted brain image.
#' @param useContralaterality If TRUE, use both hemispherical models to also
#' predict the corresponding contralateral segmentation and use both sets of
#' priors to produce the results.  Mainly used for debugging.
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return list consisting of the segmentation image and probability images for
#' each label.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' results <- deepFlash( image )
#' }
#' @export
deepFlash <- function( t1, t2 = NULL, useContralaterality = TRUE, doPreprocessing = TRUE,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
{
  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  ################################
  #
  # Preprocess image
  #
  ################################

  t1Preprocessed <- t1
  t1Preprocessing <- NULL
  t1PreprocessedFlipped <- NULL
  if( doPreprocessing == TRUE )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.01, 0.995 ),
        brainExtractionModality = "t1",
        template = "deepFlashTemplateT1",
        templateTransformType = "antsRegistrationSyNQuickRepro[a]",
        doBiasCorrection = TRUE,
        doDenoising = TRUE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage
    t1Preprocessed <- ( t1Preprocessed - mean( t1Preprocessed ) ) / sd( t1Preprocessed )
    if( useContralaterality )
      {
      t1PreprocessedDimension <- dim( t1Preprocessed )
      t1PreprocessedArray <- as.array( t1Preprocessed )
      t1PreprocessedArrayFlipped <- t1PreprocessedArray[t1PreprocessedDimension[1]:1,,]
      t1PreprocessedFlipped <- as.antsImage( t1PreprocessedArrayFlipped, reference = t1Preprocessed )
      }
    }

  t2Preprocessed <- t2
  t2PreprocessedFlipped <- NULL
  if( ! is.null( t2 ) )
    {
    if( doPreprocessing == TRUE )
      {
      t2Preprocessing <- preprocessBrainImage( t2,
          truncateIntensity = c( 0.01, 0.995 ),
          brainExtractionModality = NULL,
          templateTransformType = NULL,
          doBiasCorrection = TRUE,
          doDenoising = TRUE,
          antsxnetCacheDirectory = antsxnetCacheDirectory,
          verbose = verbose )
      t2Preprocessed <- antsApplyTransforms( fixed = t1Preprocessed,
          moving = t2Preprocessing$preprocessedImage,
          transformlist = t1Preprocessing$templateTransforms$fwdtransforms,
          verbose = verbose )
      t2Preprocessed <- ( t2Preprocessed - mean( t2Preprocessed ) ) / sd( t2Preprocessed )
      if( useContralaterality )
        {
        t2PreprocessedDimension <- dim( t2Preprocessed )
        t2PreprocessedArray <- as.array( t2Preprocessed )
        t2PreprocessedArrayFlipped <- t2PreprocessedArray[t2PreprocessedDimension[1]:1,,]
        t2PreprocessedFlipped <- as.antsImage(t2PreprocessedArrayFlipped, reference = t2Preprocessed )
        }
      }
    }

  probabilityImages <- list()
  labels <- c( 0, 5:18 )
  imageSize <- c( 64, 64, 96 )

  ################################
  #
  # Process left/right in split network
  #
  ################################

  ################################
  #
  # Download spatial priors
  #
  ################################

  spatialPriorsFileNamePath <- getANTsXNetData( "deepFlashPriors",
    antsxnetCacheDirectory = antsxnetCacheDirectory )
  spatialPriors <- antsImageRead( spatialPriorsFileNamePath )
  priorsImageList <- splitNDImageToList( spatialPriors )
  for( i in seq.int( length( priorsImageList ) ) )
    {
    priorsImageList[[i]] <- antsCopyImageInfo( t1Preprocessed, priorsImageList[[i]] )
    }

  labelsLeft <- labels[seq.int( 2, length( labels ), by = 2)]
  priorsImageLeftList <- priorsImageList[seq.int(2, length( priorsImageList ), by = 2 )]
  probabilityImagesLeft <- list()
  foregroundProbabilityImageLeft <- NULL
  lowerBoundLeft <- c( 77, 75, 57 )
  upperBoundLeft <- c( 140, 138, 152 )
  tmpCropped <- cropIndices( t1Preprocessed, lowerBoundLeft, upperBoundLeft )
  originLeft <- antsGetOrigin( tmpCropped )

  spacing <- antsGetSpacing( tmpCropped )
  direction <- antsGetDirection( tmpCropped )

  labelsRight <- labels[seq.int( 3, length( labels ), by = 2)]
  priorsImageRightList <- priorsImageList[seq.int(3, length( priorsImageList ), by = 2 )]
  probabilityImagesRight <- list()
  foregroundProbabilityImageRight <- NULL
  lowerBoundRight <- c( 21, 75, 57 )
  upperBoundRight <- c( 84, 138, 152 )
  tmpCropped <- cropIndices( t1Preprocessed, lowerBoundRight, upperBoundRight )
  originRight <- antsGetOrigin( tmpCropped )

  ################################
  #
  # Left:  build model and load weights
  #
  ################################

  channelSize <- 1 + length( labelsLeft )
  numberOfClassificationLabels <- 1 + length( labelsLeft )

  networkName <- 'deepFlashLeftT1'
  if( ! is.null( t2 ) )
    {
    networkName <- 'deepFlashLeftBoth'
    channelSize <- channelSize + 1
    }

  unetModel <- createUnetModel3D( c( imageSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
    numberOfFilters = c( 32, 64, 96, 128, 256 ),
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    dropoutRate = 0.0, weightDecay = 0 )

  penultimateLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]$output
  output <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
    activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )
  unetModel <- keras::keras_model( inputs = unetModel$input, outputs = list( unetModel$output, output ) )

  if( verbose == TRUE )
    {
    cat( "DeepFlash: retrieving model weights (left).\n" )
    }
  weightsFileName <- getPretrainedNetwork( networkName, antsxnetCacheDirectory = antsxnetCacheDirectory )
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )

  ################################
  #
  # Left:  do prediction and normalize to native space
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Prediction (left).\n" )
    }

  batchX <- NULL
  if( useContralaterality )
    {
    batchX <- array( data = 0, dim = c( 2, imageSize, channelSize ) )
    } else {
    batchX <- array( data = 0, dim = c( 1, imageSize, channelSize ) )
    }

  t1Cropped <- cropIndices( t1Preprocessed, lowerBoundLeft, upperBoundLeft )
  batchX[1,,,,1] <- as.array( t1Cropped )
  if( useContralaterality )
    {
    t1Cropped <- cropIndices( t1PreprocessedFlipped, lowerBoundLeft, upperBoundLeft )
    batchX[2,,,,1] <- as.array( t1Cropped )
    }
  if( ! is.null( t2 ) )
    {
    t2Cropped <- cropIndices( t2Preprocessed, lowerBoundLeft, upperBoundLeft )
    batchX[1,,,,2] <- as.array( t2Cropped )
    if( useContralaterality )
      {
      t1Cropped <- cropIndices( t2PreprocessedFlipped, lowerBoundLeft, upperBoundLeft )
      batchX[2,,,,2] <- as.array( t2Cropped )
      }
    }

  for( i in seq.int( length( priorsImageLeftList ) ) )
    {
    croppedPrior <- cropIndices( priorsImageLeftList[[i]], lowerBoundLeft, upperBoundLeft )
    for( j in seq.int( dim( batchX )[1] ) )
      {
      batchX[j,,,,i + ( channelSize - length( labelsLeft ) )] <- as.array( croppedPrior )
      }
    }

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  probabilityImagesList <- decodeUnet( predictedData[[1]], t1Cropped )

  for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
    {
    for( j in seq.int( dim( predictedData[[1]] )[1] ) )
      {
      probabilityImage <- probabilityImagesList[[j]][[i]]
      if( i > 1 )
        {
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )
        } else {
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 + 1 )
        }

      if( j == 2 ) # flipped
        {
        probabilityArray <- as.array( probabilityImage )
        probabilityArrayDimension <- dim( probabilityImage )
        probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
        probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
        }

      if( doPreprocessing == TRUE )
        {
        probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        }

      if( j == 1 )  # not flipped
        {
        probabilityImagesLeft <- append( probabilityImagesLeft, probabilityImage )
        } else {    # flipped
        probabilityImagesRight <- append( probabilityImagesRight, probabilityImage )
        }
      }
    }

  ################################
  #
  # Left:  do prediction of foreground and normalize to native space
  #
  ################################

  probabilityImagesList <- decodeUnet( predictedData[[2]], t1Cropped )

  for( j in seq.int( dim( predictedData[[2]] )[1] ) )
    {
    probabilityImage <- probabilityImagesList[[j]][[1]]
    if( i > 1 )
      {
      probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )
      } else {
      probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 + 1 )
      }

    if( j == 2 ) # flipped
      {
      probabilityArray <- as.array( probabilityImage )
      probabilityArrayDimension <- dim( probabilityImage )
      probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
      probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
      }

    if( doPreprocessing == TRUE )
      {
      probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      }

    if( j == 1 )  # not flipped
      {
      foregroundProbabilityImageLeft <- antsImageClone( probabilityImage )
      } else {    # flipped
      foregroundProbabilityImageRight <- antsImageClone( probabilityImage )
      }
    }

  ################################
  #
  # Right:  build model and load weights
  #
  ################################

  channelSize <- 1 + length( labelsRight )
  numberOfClassificationLabels <- 1 + length( labelsRight )

  networkName <- 'deepFlashRightT1'
  if( ! is.null( t2 ) )
    {
    networkName <- 'deepFlashRightBoth'
    channelSize <- channelSize + 1
    }

  unetModel <- createUnetModel3D( c( imageSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
    numberOfFilters = c( 32, 64, 96, 128, 256 ),
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    dropoutRate = 0.0, weightDecay = 0 )

  penultimateLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]$output
  output <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
    activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )
  unetModel <- keras::keras_model( inputs = unetModel$input, outputs = list( unetModel$output, output ) )

  if( verbose == TRUE )
    {
    cat( "DeepFlash: retrieving model weights (Right).\n" )
    }
  weightsFileName <- getPretrainedNetwork( networkName, antsxnetCacheDirectory = antsxnetCacheDirectory )
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )

  ################################
  #
  # Right:  do prediction and normalize to native space
  #
  ################################

  if( verbose == TRUE )
    {
    cat( "Prediction (Right).\n" )
    }

  batchX <- NULL
  if( useContralaterality )
    {
    batchX <- array( data = 0, dim = c( 2, imageSize, channelSize ) )
    } else {
    batchX <- array( data = 0, dim = c( 1, imageSize, channelSize ) )
    }

  t1Cropped <- cropIndices( t1Preprocessed, lowerBoundRight, upperBoundRight )
  batchX[1,,,,1] <- as.array( t1Cropped )
  if( useContralaterality )
    {
    t1Cropped <- cropIndices( t1PreprocessedFlipped, lowerBoundRight, upperBoundRight )
    batchX[2,,,,1] <- as.array( t1Cropped )
    }
  if( ! is.null( t2 ) )
    {
    t2Cropped <- cropIndices( t2Preprocessed, lowerBoundRight, upperBoundRight )
    batchX[1,,,,2] <- as.array( t2Cropped )
    if( useContralaterality )
      {
      t1Cropped <- cropIndices( t2PreprocessedFlipped, lowerBoundRight, upperBoundRight )
      batchX[2,,,,2] <- as.array( t2Cropped )
      }
    }

  for( i in seq.int( length( priorsImageRightList ) ) )
    {
    croppedPrior <- cropIndices( priorsImageRightList[[i]], lowerBoundRight, upperBoundRight )
    for( j in seq.int( dim( batchX )[1] ) )
      {
      batchX[j,,,,i + ( channelSize - length( labelsRight ) )] <- as.array( croppedPrior )
      }
    }

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  probabilityImagesList <- decodeUnet( predictedData[[1]], t1Cropped )

  for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
    {
    for( j in seq.int( dim( predictedData[[1]] )[1] ) )
      {
      probabilityImage <- probabilityImagesList[[j]][[i]]
      if( i > 1 )
        {
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )
        } else {
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 + 1 )
        }

      if( j == 2 ) # flipped
        {
        probabilityArray <- as.array( probabilityImage )
        probabilityArrayDimension <- dim( probabilityImage )
        probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
        probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
        }

      if( doPreprocessing == TRUE )
        {
        probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        }

      if( j == 1 )  # not flipped
        {
        if( useContralaterality )
          {
          probabilityImagesRight[[i]] <- ( probabilityImagesRight[[i]] +  probabilityImage ) / 2
          } else {
          probabilityImagesRight <- append( probabilityImagesRight, probabilityImage )
          }
        } else {    # flipped
        probabilityImagesLeft[[i]] <- ( probabilityImagesLeft[[i]] +  probabilityImage ) / 2
        }
      }
    }

  ################################
  #
  # Right:  do prediction of foreground and normalize to native space
  #
  ################################

  probabilityImagesList <- decodeUnet( predictedData[[2]], t1Cropped )

  for( j in seq.int( dim( predictedData[[2]] )[1] ) )
    {
    probabilityImage <- probabilityImagesList[[j]][[1]]
    if( i > 1 )
      {
      probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )
      } else {
      probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 + 1 )
      }

    if( j == 2 ) # flipped
      {
      probabilityArray <- as.array( probabilityImage )
      probabilityArrayDimension <- dim( probabilityImage )
      probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
      probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
      }

    if( doPreprocessing == TRUE )
      {
      probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      }

    if( j == 1 )  # not flipped
      {
      if( useContralaterality )
        {
        foregroundProbabilityImageRight <- ( foregroundProbabilityImageRight + probabilityImage ) / 2
        } else {
        foregroundProbabilityImageRight <- probabilityImage
        }
      } else {    # flipped
      foregroundProbabilityImageLeft <- ( foregroundProbabilityImageLeft + probabilityImage ) / 2
      }
    }

  ################################
  #
  # Combine priors
  #
  ################################

  probabilityBackgroundImage <- antsImageClone( t1 ) * 0
  for( i in seq.int( from = 2, to = length( probabilityImagesLeft ) ) )
    {
    probabilityBackgroundImage <- probabilityBackgroundImage + probabilityImagesLeft[[i]]
    }
  for( i in seq.int( from = 2, to = length( probabilityImagesRight ) ) )
    {
    probabilityBackgroundImage <- probabilityBackgroundImage + probabilityImagesRight[[i]]
    }

  count <- 1
  probabilityImages[[count]] <- probabilityBackgroundImage * -1 + 1
  count <- count + 1
  for( i in seq.int( from = 2, to = length( probabilityImagesLeft ) ) )
    {
    probabilityImages[[count]] <- probabilityImagesLeft[[i]]
    count <- count + 1
    probabilityImages[[count]] <- probabilityImagesRight[[i]]
    count <- count + 1
    }

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

  foregroundProbabilityImage = foregroundProbabilityImageLeft + foregroundProbabilityImageRight

  results <- list( segmentationImage = relabeledImage,
                   probabilityImages = probabilityImages,
                   foregroundProbabilityImage = foregroundProbabilityImage )

  return( results )
}

#' Hippocampal/Enthorhinal segmentation using "Deep Flash"
#'
#' Perform hippocampal/entorhinal segmentation in T1 images using
#' labels from Mike Yassa's lab
#'
#' \url{https://faculty.sites.uci.edu/myassa/}
#'
#' The labeling is as follows:
#' \itemize{
#'   \item{Label 0 :}{background}
#'   \item{Label 5 :}{left aLEC}
#'   \item{Label 6 :}{right aLEC}
#'   \item{Label 7 :}{left pMEC}
#'   \item{Label 8 :}{right pMEC}
#'   \item{Label 9 :}{left perirhinal}
#'   \item{Label 10:}{right perirhinal}
#'   \item{Label 11:}{left parahippocampal}
#'   \item{Label 12:}{right parahippocampal}
#'   \item{Label 13:}{left DG/CA3}
#'   \item{Label 14:}{right DG/CA3}
#'   \item{Label 15:}{left CA1}
#'   \item{Label 16:}{right CA1}
#'   \item{Label 17:}{left subiculum}
#'   \item{Label 18:}{right subiculum}
#' }
#'
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * denoising,
#'    * brain extraction, and
#'    * affine registration to MNI.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param t1 raw or preprocessed 3-D T1-weighted brain image.
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param doPerHemisphere If TRUE, do prediction based on separate networks per
#' hemisphere.  Otherwise, use the single network trained for both hemispheres.
#' @param whichHemisphereModels "old" or "new".
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' inst/extdata/ subfolder of the ANTsRNet package.
#' @param verbose print progress.
#' @return list consisting of the segmentation image and probability images for
#' each label.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' results <- deepFlash( image )
#' }
#' @export
deepFlashDeprecated <- function( t1, doPreprocessing = TRUE, doPerHemisphere = TRUE,
  whichHemisphereModels = "new", antsxnetCacheDirectory = NULL, verbose = FALSE )
{
  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  ################################
  #
  # Preprocess image
  #
  ################################

  t1Preprocessed <- t1
  if( doPreprocessing == TRUE )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.01, 0.99 ),
        brainExtractionModality = "t1",
        template = "croppedMni152",
        templateTransformType = "antsRegistrationSyNQuickRepro[a]",
        doBiasCorrection = TRUE,
        doDenoising = TRUE,
        antsxnetCacheDirectory = antsxnetCacheDirectory,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    }

  probabilityImages <- list()
  labels <- c( 0, 5:18 )

  ################################
  #
  # Process left/right in same network
  #
  ################################

  if( doPerHemisphere == FALSE )
    {

    ################################
    #
    # Build model and load weights
    #
    ################################

    templateSize <- c( 160L, 192L, 160L )

    unetModel <- createUnetModel3D( c( templateSize, 1 ),
      numberOfOutputs = length( labels ), mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 1e-5, additionalOptions = c( "attentionGating" )  )

    if( verbose == TRUE )
      {
      cat( "DeepFlash: retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "deepFlash", antsxnetCacheDirectory = antsxnetCacheDirectory )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose == TRUE )
      {
      cat( "Prediction.\n" )
      }

    croppedImage <- padOrCropImageToSize( t1Preprocessed, templateSize )
    imageArray <- as.array( croppedImage )

    batchX <- array( data = imageArray, dim = c( 1, templateSize, 1 ) )
    batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesList <- decodeUnet( predictedData, croppedImage )

    for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
      {
      if( i > 1 )
        {
        decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 )
        } else {
        decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 + 1 )
        }
      if( doPreprocessing == TRUE )
        {
        probabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        } else {
        probabilityImages[[i]] <- decroppedImage
        }
      }

    ################################
    #
    # Process left/right in split networks
    #
    ################################

    } else {

    ################################
    #
    # Left:  download spatial priors
    #
    ################################

    spatialPriorsLeftFileNamePath <- getANTsXNetData( "priorDeepFlashLeftLabels",
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    spatialPriorsLeft <- antsImageRead( spatialPriorsLeftFileNamePath )
    priorsImageLeftList <- splitNDImageToList( spatialPriorsLeft )

    ################################
    #
    # Left:  build model and load weights
    #
    ################################

    templateSize <- c( 64L, 96L, 96L )
    labelsLeft <- c( 0, 5, 7, 9, 11, 13, 15, 17 )
    channelSize <- 1 + length( labelsLeft )

    numberOfFilters <- 16
    networkName <- ''
    if( whichHemisphereModels == "old" )
      {
      networkName <- "deepFlashLeft16"
      } else if( whichHemisphereModels == "new" ) {
      networkName <- "deepFlashLeft16new"
      } else {
      stop( "whichHemisphereModels must be \"old\" or \"new\"." )
      }

    unetModel <- createUnetModel3D( c( templateSize, channelSize ),
      numberOfOutputs = length( labelsLeft ), mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFilters, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

    if( verbose == TRUE )
      {
      cat( "DeepFlash: retrieving model weights (left).\n" )
      }
    weightsFileName <- getPretrainedNetwork( networkName, antsxnetCacheDirectory = antsxnetCacheDirectory )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Left:  do prediction and normalize to native space
    #
    ################################

    if( verbose == TRUE )
      {
      cat( "Prediction (left).\n" )
      }

    croppedImage <- cropIndices( t1Preprocessed, c( 31, 52, 1 ), c( 94, 147, 96 ) )
    imageArray <- as.array( croppedImage )
    imageArray <- ( imageArray - mean( imageArray ) ) / sd( imageArray )

    batchX <- array( data = 0, dim = c( 1, templateSize, channelSize ) )
    batchX[1,,,,1] <- imageArray

    for( i in seq.int( length( priorsImageLeftList ) ) )
      {
      croppedPrior <- cropIndices( priorsImageLeftList[[i]], c( 31, 52, 1 ), c( 94, 147, 96 ) )
      batchX[1,,,,i+1] <- as.array( croppedPrior )
      }

    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesList <- decodeUnet( predictedData, croppedImage )

    probabilityImagesLeft <- list()
    for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
      {
      if( i > 1 )
        {
        decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 )
        } else {
        decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 + 1 )
        }
      if( doPreprocessing == TRUE )
        {
        probabilityImagesLeft[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        } else {
        probabilityImagesLeft[[i]] <- decroppedImage
        }
      }

    ################################
    #
    # Right:  download spatial priors
    #
    ################################

    spatialPriorsRightFileNamePath <- getANTsXNetData( "priorDeepFlashRightLabels",
      antsxnetCacheDirectory = antsxnetCacheDirectory )
    spatialPriorsRight <- antsImageRead( spatialPriorsRightFileNamePath )
    priorsImageRightList <- splitNDImageToList( spatialPriorsRight )

    ################################
    #
    # Right:  build model and load weights
    #
    ################################

    templateSize <- c( 64L, 96L, 96L )
    labelsRight <- c( 0, 6, 8, 10, 12, 14, 16, 18 )
    channelSize <- 1 + length( labelsRight )

    numberOfFilters <- 16
    networkName <- ''
    if( whichHemisphereModels == "old" )
      {
      networkName <- "deepFlashRight16"
      } else if( whichHemisphereModels == "new" ) {
      networkName <- "deepFlashRight16new"
      } else {
      stop( "whichHemisphereModels must be \"old\" or \"new\"." )
      }

    unetModel <- createUnetModel3D( c( templateSize, channelSize ),
      numberOfOutputs = length( labelsRight ), mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFilters, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

    if( verbose == TRUE )
      {
      cat( "DeepFlash: retrieving model weights (Right).\n" )
      }
    weightsFileName <- getPretrainedNetwork( networkName, antsxnetCacheDirectory = antsxnetCacheDirectory )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Right:  do prediction and normalize to native space
    #
    ################################

    if( verbose == TRUE )
      {
      cat( "Prediction (Right).\n" )
      }

    croppedImage <- cropIndices( t1Preprocessed, c( 89, 52, 1 ), c( 152, 147, 96 ) )
    imageArray <- as.array( croppedImage )
    imageArray <- ( imageArray - mean( imageArray ) ) / sd( imageArray )

    batchX <- array( data = 0, dim = c( 1, templateSize, channelSize ) )
    batchX[1,,,,1] <- imageArray

    for( i in seq.int( length( priorsImageRightList ) ) )
      {
      croppedPrior <- cropIndices( priorsImageRightList[[i]], c( 89, 52, 1 ), c( 152, 147, 96 ) )
      batchX[1,,,,i+1] <- as.array( croppedPrior )
      }

    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesList <- decodeUnet( predictedData, croppedImage )

    probabilityImagesRight <- list()
    for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
      {
      if( i > 1 )
        {
        decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 )
        } else {
        decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 + 1 )
        }
      if( doPreprocessing == TRUE )
        {
        probabilityImagesRight[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        } else {
        probabilityImagesRight[[i]] <- decroppedImage
        }
      }

    ################################
    #
    # Combine priors
    #
    ################################

    probabilityBackgroundImage <- antsImageClone(t1) * 0
    for( i in seq.int( from = 2, to = length( probabilityImagesLeft ) ) )
      {
      probabilityBackgroundImage <- probabilityBackgroundImage + probabilityImagesLeft[[i]]
      }
    for( i in seq.int( from = 2, to = length( probabilityImagesRight ) ) )
      {
      probabilityBackgroundImage <- probabilityBackgroundImage + probabilityImagesRight[[i]]
      }

    count <- 1
    probabilityImages[[count]] <- probabilityBackgroundImage * -1 + 1
    count <- count + 1
    for( i in seq.int( from = 2, to = length( probabilityImagesLeft ) ) )
      {
      probabilityImages[[count]] <- probabilityImagesLeft[[i]]
      count <- count + 1
      probabilityImages[[count]] <- probabilityImagesRight[[i]]
      count <- count + 1
      }
    }

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

  results <- list( segmentationImage = relabeledImage, probabilityImages = probabilityImages )

  # debugging

  # if( debug == TRUE )
  #   {
  #   inputImage <- unetModel$input
  #   featureLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]
  #   featureFunction <- keras::backend()$`function`( list( inputImage ), list( featureLayer$output ) )
  #   featureBatch <- featureFunction( list( batchX[1,,,,,drop = FALSE] ) )
  #   featureImagesList <- decodeUnet( featureBatch[[1]], croppedImage )
  #   featureImages <- list()
  #   for( i in seq.int( length( featureImagesList[[1]] ) ) )
  #     {
  #     decroppedImage <- decropImage( featureImagesList[[1]][[i]], t1Preprocessed * 0 )
  #     if( doPreprocessing == TRUE )
  #       {
  #       featureImages[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
  #           transformlist = t1Preprocessing$templateTransforms$invtransforms,
  #           whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
  #       } else {
  #       featureImages[[i]] <- decroppedImage
  #       }
  #     }
  #   results[['featureImagesLastLayer']] <- featureImages
  #   }

  return( results )
}

