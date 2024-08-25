#' Hippocampal/Enthorhinal segmentation using "Deep Flash"
#'
#' Perform hippocampal/entorhinal segmentation in T1 and T1/T2 images using
#' labels from Mike Yassa's lab---\url{https://faculty.sites.uci.edu/myassa/}
#'
#' \url{https://www.nature.com/articles/s41598-024-59440-6}
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
#'   \item{Label 13:}{left DG/CA2/CA3/CA4}
#'   \item{Label 14:}{right DG/CA2/CA3/CA4}
#'   \item{Label 15:}{left CA1}
#'   \item{Label 16:}{right CA1}
#'   \item{Label 17:}{left subiculum}
#'   \item{Label 18:}{right subiculum}
#' }
#'
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * affine registration to deep flash template.
#' which is performed on the input images if \code{doPreprocessing = TRUE}.
#'
#' @param t1 raw or preprocessed 3-D T1-weighted brain image.
#' @param t2 optional raw or preprocessed 3-D T2-weighted brain image.
#' @param whichParcellation string --- "yassa".  See above label descriptions.
#' @param doPreprocessing perform preprocessing.  See description above.
#' @param useRankIntensity If false, use histogram matching with cropped template
#' ROI.  Otherwise, use a rank intensity transform on the cropped ROI.  Only for
#' 'yassa' parcellation.
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
deepFlash <- function( t1, t2 = NULL, whichParcellation = "yassa",
  doPreprocessing = TRUE, useRankIntensity = TRUE, verbose = FALSE )
{
  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  if( whichParcellation == "yassa" )
    {
    ################################
    #
    # Options temporarily taken from the user
    #
    ################################

    # @param useContralaterality If TRUE, use both hemispherical models to also
    # predict the corresponding contralateral segmentation and use both sets of
    # priors to produce the results.  Mainly used for debugging.
    #
    # @param useHierarchicalParcellation If TRUE, use both hemispherical models to also
    # predict the corresponding contralateral segmentation and use both sets of
    # priors to produce the results.  Mainly used for debugging.

    useHierarchicalParcellation <- TRUE
    useContralaterality <- TRUE

    ################################
    #
    # Preprocess image
    #
    ################################

    t1Preprocessed <- t1
    t1Mask <- NULL
    t1PreprocessedFlipped <- NULL
    t1Template <- antsImageRead( getANTsXNetData( "deepFlashTemplateT1SkullStripped" ) )
    templateTransforms <- NULL
    if( doPreprocessing )
      {
      if( verbose )
        {
        cat( "Preprocessing T1.\n" )
        }

      # Brain extraction
      probabilityMask <- brainExtraction( t1Preprocessed, modality = "t1", verbose = verbose )
      t1Mask <- thresholdImage( probabilityMask, 0.5, 1, 1, 0)
      t1Preprocessed <- t1Preprocessed * t1Mask

      # Do bias correction
      t1Preprocessed <- n4BiasFieldCorrection( t1Preprocessed, t1Mask, shrinkFactor = 4, verbose = verbose )

      # Warp to template
      registration <- antsRegistration( fixed = t1Template, moving = t1Preprocessed,
        typeofTransform = "antsRegistrationSyNQuickRepro[a]", verbose = verbose )
      templateTransforms <- list( fwdtransforms = registration$fwdtransforms,
                                  invtransforms = registration$invtransforms )
      t1Preprocessed <- registration$warpedmovout

      }
    if( useContralaterality )
      {
      t1PreprocessedDimension <- dim( t1Preprocessed )
      t1PreprocessedArray <- as.array( t1Preprocessed )
      t1PreprocessedArrayFlipped <- t1PreprocessedArray[t1PreprocessedDimension[1]:1,,]
      t1PreprocessedFlipped <- as.antsImage( t1PreprocessedArrayFlipped, reference = t1Preprocessed )
      }

    t2Preprocessed <- t2
    t2PreprocessedFlipped <- NULL
    t2Template <- NULL
    if( ! is.null( t2 ) )
      {
      t2Template <- antsImageRead( getANTsXNetData( "deepFlashTemplateT2SkullStripped" ) )
      t2Template <- antsCopyImageInfo(t1Template, t2Template)
      if( doPreprocessing )
        {
        if( verbose )
          {
          cat( "Preprocessing T2.\n" )
          }

        # Brain extraction
        t2Preprocessed <- t2Preprocessed * t1Mask

        # Do bias correction
        t2Preprocessed <- n4BiasFieldCorrection( t2Preprocessed, t1Mask, shrinkFactor = 4, verbose = verbose )

        # Warp to template
        t2Preprocessed <- antsApplyTransforms( fixed = t1Template, moving = t2Preprocessed,
          transformlist = templateTransforms$fwdtransforms, verbose = verbose )
        }
      if( useContralaterality )
        {
        t2PreprocessedDimension <- dim( t2Preprocessed )
        t2PreprocessedArray <- as.array( t2Preprocessed )
        t2PreprocessedArrayFlipped <- t2PreprocessedArray[t2PreprocessedDimension[1]:1,,]
        t2PreprocessedFlipped <- as.antsImage(t2PreprocessedArrayFlipped, reference = t2Preprocessed )
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

    spatialPriorsFileNamePath <- getANTsXNetData( "deepFlashPriors" )
    spatialPriors <- antsImageRead( spatialPriorsFileNamePath )
    priorsImageList <- splitNDImageToList( spatialPriors )
    for( i in seq.int( length( priorsImageList ) ) )
      {
      priorsImageList[[i]] <- antsCopyImageInfo( t1Preprocessed, priorsImageList[[i]] )
      }

    labelsLeft <- labels[seq.int( 2, length( labels ), by = 2)]
    priorsImageLeftList <- priorsImageList[seq.int(2, length( priorsImageList ), by = 2 )]
    probabilityImagesLeft <- list()
    foregroundProbabilityImagesLeft <- list()
    lowerBoundLeft <- c( 77, 75, 57 )
    upperBoundLeft <- c( 140, 138, 152 )
    tmpCropped <- cropIndices( t1Preprocessed, lowerBoundLeft, upperBoundLeft )
    originLeft <- antsGetOrigin( tmpCropped )

    spacing <- antsGetSpacing( tmpCropped )
    direction <- antsGetDirection( tmpCropped )

    t1TemplateRoiLeft <- cropIndices( t1Template, lowerBoundLeft, upperBoundLeft )
    t1TemplateRoiLeft <- iMath( t1TemplateRoiLeft, "Normalize" ) * 2.0 - 1.0
    t2TemplateRoiLeft <- NULL
    if( ! is.null( t2Template ) )
      {
      t2TemplateRoiLeft <- cropIndices( t2Template, lowerBoundLeft, upperBoundLeft )
      t2TemplateRoiLeft <- iMath( t2TemplateRoiLeft, "Normalize" ) * 2.0 - 1.0
      }

    labelsRight <- labels[seq.int( 3, length( labels ), by = 2)]
    priorsImageRightList <- priorsImageList[seq.int(3, length( priorsImageList ), by = 2 )]
    probabilityImagesRight <- list()
    foregroundProbabilityImagesRight <- list()
    lowerBoundRight <- c( 21, 75, 57 )
    upperBoundRight <- c( 84, 138, 152 )
    tmpCropped <- cropIndices( t1Preprocessed, lowerBoundRight, upperBoundRight )
    originRight <- antsGetOrigin( tmpCropped )

    t1TemplateRoiRight <- cropIndices( t1Template, lowerBoundRight, upperBoundRight )
    t1TemplateRoiRight <- iMath( t1TemplateRoiRight, "Normalize" ) * 2.0 - 1.0
    t2TemplateRoiRight <- NULL
    if( ! is.null( t2Template ) )
      {
      t2TemplateRoiRight <- cropIndices( t2Template, lowerBoundRight, upperBoundRight )
      t2TemplateRoiRight <- iMath( t2TemplateRoiRight, "Normalize" ) * 2.0 - 1.0
      }

    ################################
    #
    # Create model
    #
    ################################

    channelSize <- 1 + length( labelsLeft )
    if( ! is.null( t2 ) )
      {
      channelSize <- channelSize + 1
      }
    numberOfClassificationLabels <- 1 + length( labelsLeft )

    unetModel <- createUnetModel3D( c( imageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
      numberOfFilters = c( 32, 64, 96, 128, 256 ),
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      dropoutRate = 0.0, weightDecay = 0 )

    penultimateLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]$output

    # medial temporal lobe
    output1 <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
      activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )

    if( useHierarchicalParcellation )
      {
      # EC, perirhinal, and parahippo.
      output2 <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
        activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )

      # Hippocampus
      output3 <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
        activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )

      unetModel <- keras::keras_model( inputs = unetModel$input, outputs = list( unetModel$output, output1, output2, output3 ) )
      } else {
      unetModel <- keras::keras_model( inputs = unetModel$input, outputs = list( unetModel$output, output1 ) )
      }

    ################################
    #
    # Left:  build model and load weights
    #
    ################################

    networkName <- 'deepFlashLeftT1'
    if( ! is.null( t2 ) )
      {
      networkName <- 'deepFlashLeftBoth'
      }

    if( useHierarchicalParcellation )
      {
      networkName <- paste0( networkName, "Hierarchical" )
      }

    if( useRankIntensity )
      {
      networkName <- paste0( networkName, "_ri" )
      }

    if( verbose )
      {
      cat( "DeepFlash: retrieving model weights (left).\n" )
      }
    weightsFileName <- getPretrainedNetwork( networkName )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Left:  do prediction and normalize to native space
    #
    ################################

    if( verbose )
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
    if( useRankIntensity )
      {
      t1Cropped <- rankIntensity( t1Cropped )
      } else {
      t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiLeft, 255, 64, FALSE )
      }
    batchX[1,,,,1] <- as.array( t1Cropped )
    if( useContralaterality )
      {
      t1Cropped <- cropIndices( t1PreprocessedFlipped, lowerBoundLeft, upperBoundLeft )
      if( useRankIntensity )
        {
        t1Cropped <- rankIntensity( t1Cropped )
        } else {
        t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiLeft, 255, 64, FALSE )
        }
      batchX[2,,,,1] <- as.array( t1Cropped )
      }
    if( ! is.null( t2 ) )
      {
      t2Cropped <- cropIndices( t2Preprocessed, lowerBoundLeft, upperBoundLeft )
      if( useRankIntensity )
        {
        t2Cropped <- rankIntensity( t2Cropped )
        } else {
        t2Cropped <- histogramMatchImage( t2Cropped, t2TemplateRoiLeft, 255, 64, FALSE )
        }
      batchX[1,,,,2] <- as.array( t2Cropped )
      if( useContralaterality )
        {
        t2Cropped <- cropIndices( t2PreprocessedFlipped, lowerBoundLeft, upperBoundLeft )
        if( useRankIntensity )
          {
          t2Cropped <- rankIntensity( t2Cropped )
          } else {
          t2Cropped <- histogramMatchImage( t2Cropped, t2TemplateRoiLeft, 255, 64, FALSE )
          }
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

    for( i in seq.int( 1 + length( labelsLeft ) ) )
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

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
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
    # Left:  do prediction of mtl, hippocampal, and ec regions and normalize to native space
    #
    ################################

    for( i in seq.int( 2, length( predictedData ) ) )
      {
      probabilityImagesList <- decodeUnet( predictedData[[i]], t1Cropped )
      for( j in seq.int( dim( predictedData[[i]] )[1] ) )
        {
        probabilityImage <- probabilityImagesList[[j]][[1]]
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )

        if( j == 2 ) # flipped
          {
          probabilityArray <- as.array( probabilityImage )
          probabilityArrayDimension <- dim( probabilityImage )
          probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
          probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
          }

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
              whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
          }

        if( j == 1 )  # not flipped
          {
          foregroundProbabilityImagesLeft <- append( foregroundProbabilityImagesLeft, probabilityImage )
          } else {    # flipped
          foregroundProbabilityImagesRight <- append( foregroundProbabilityImagesRight, probabilityImage )
          }
        }
      }

    ################################
    #
    # Right:  build model and load weights
    #
    ################################

    networkName <- 'deepFlashRightT1'
    if( ! is.null( t2 ) )
      {
      networkName <- 'deepFlashRightBoth'
      }

    if( useHierarchicalParcellation )
      {
      networkName <- paste0( networkName, "Hierarchical" )
      }

    if( useRankIntensity )
      {
      networkName <- paste0( networkName, "_ri" )
      }

    if( verbose )
      {
      cat( "DeepFlash: retrieving model weights (Right).\n" )
      }
    weightsFileName <- getPretrainedNetwork( networkName )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Right:  do prediction and normalize to native space
    #
    ################################

    if( verbose )
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
    if( useRankIntensity )
      {
      t1Cropped <- rankIntensity( t1Cropped )
      } else {
      t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiRight, 255, 64, FALSE )
      }
    batchX[1,,,,1] <- as.array( t1Cropped )
    if( useContralaterality )
      {
      t1Cropped <- cropIndices( t1PreprocessedFlipped, lowerBoundRight, upperBoundRight )
      if( useRankIntensity )
        {
        t1Cropped <- rankIntensity( t1Cropped )
        } else {
        t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiRight, 255, 64, FALSE )
        }
      batchX[2,,,,1] <- as.array( t1Cropped )
      }
    if( ! is.null( t2 ) )
      {
      t2Cropped <- cropIndices( t2Preprocessed, lowerBoundRight, upperBoundRight )
      if( useRankIntensity )
        {
        t2Cropped <- rankIntensity( t2Cropped )
        } else {
        t2Cropped <- histogramMatchImage( t2Cropped, t2TemplateRoiRight, 255, 64, FALSE )
        }
      batchX[1,,,,2] <- as.array( t2Cropped )
      if( useContralaterality )
        {
        t2Cropped <- cropIndices( t2PreprocessedFlipped, lowerBoundRight, upperBoundRight )
        if( useRankIntensity )
          {
          t2Cropped <- rankIntensity( t2Cropped )
          } else {
          t2Cropped <- histogramMatchImage( t2Cropped, t2TemplateRoiRight, 255, 64, FALSE )
          }
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

    for( i in seq.int( 1 + length( labelsRight ) ) )
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

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
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
    # Right:  do prediction of mtl, hippocampal, and ec regions and normalize to native space
    #
    ################################

    for( i in seq.int( 2, length( predictedData ) ) )
      {
      probabilityImagesList <- decodeUnet( predictedData[[i]], t1Cropped )
      for( j in seq.int( dim( predictedData[[i]] )[1] ) )
        {
        probabilityImage <- probabilityImagesList[[j]][[1]]
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )

        if( j == 2 ) # flipped
          {
          probabilityArray <- as.array( probabilityImage )
          probabilityArrayDimension <- dim( probabilityImage )
          probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
          probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
          }

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
              whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
          }

        if( j == 1 )  # not flipped
          {
          if( useContralaterality )
            {
            foregroundProbabilityImagesRight[[i-1]] <- ( foregroundProbabilityImagesRight[[i-1]] + probabilityImage ) / 2
            } else {
            foregroundProbabilityImagesRight <- append( foregroundProbabilityImagesRight, probabilityImage )
            }
          } else {    # flipped
          foregroundProbabilityImagesLeft[[i-1]] <- ( foregroundProbabilityImagesLeft[[i-1]] + probabilityImage ) / 2
          }
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

    ################################
    #
    # Convert probability images to segmentation
    #
    ################################

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

    foregroundProbabilityImages <- list()
    for( i in seq.int( length( foregroundProbabilityImagesLeft ) ) )
      {
      foregroundProbabilityImages[[i]] <- foregroundProbabilityImagesLeft[[i]] + foregroundProbabilityImagesRight[[i]]
      }

    results <- NULL
    if( useHierarchicalParcellation )
      {
      results <- list( segmentationImage = relabeledImage,
                      probabilityImages = probabilityImages,
                      medialTemporalLobelProbabilityImage = foregroundProbabilityImages[[1]],
                      otherRegionProbabilityImage = foregroundProbabilityImages[[2]],
                      hippocampalProbabilityImage = foregroundProbabilityImages[[3]]
                    )
      } else {
      results <- list( segmentationImage = relabeledImage,
                      probabilityImages = probabilityImages,
                      medialTemporalLobelProbabilityImage = foregroundProbabilityImages[[1]]
                    )
      }

    return( results )

    } else if( whichParcellation == "wip" ) {

    useContralaterality <- TRUE

    ################################
    #
    # Preprocess image
    #
    ################################

    t1Preprocessed <- t1
    t1Mask <- NULL
    t1PreprocessedFlipped <- NULL
    t1Template <- antsImageRead( getANTsXNetData( "deepFlashTemplate2T1SkullStripped" ) )
    templateTransforms <- NULL
    if( doPreprocessing )
      {
      if( verbose )
        {
        cat( "Preprocessing T1.\n" )
        }

      # Brain extraction
      probabilityMask <- brainExtraction( t1Preprocessed, modality = "t1", verbose = verbose )
      t1Mask <- thresholdImage( probabilityMask, 0.5, 1, 1, 0)
      t1Preprocessed <- t1Preprocessed * t1Mask

      # Do bias correction
      t1Preprocessed <- n4BiasFieldCorrection( t1Preprocessed, t1Mask, shrinkFactor = 4, verbose = verbose )

      # Warp to template
      registration <- antsRegistration( fixed = t1Template, moving = t1Preprocessed,
        typeofTransform = "antsRegistrationSyNQuickRepro[a]", verbose = verbose )
      templateTransforms <- list( fwdtransforms = registration$fwdtransforms,
                                  invtransforms = registration$invtransforms )
      t1Preprocessed <- registration$warpedmovout

      }
    if( useContralaterality )
      {
      t1PreprocessedDimension <- dim( t1Preprocessed )
      t1PreprocessedArray <- as.array( t1Preprocessed )
      t1PreprocessedArrayFlipped <- t1PreprocessedArray[t1PreprocessedDimension[1]:1,,]
      t1PreprocessedFlipped <- as.antsImage( t1PreprocessedArrayFlipped, reference = t1Preprocessed )
      }

    probabilityImages <- list()
    labelsLeft <- c( 104, 105, 106, 108, 109, 110, 114, 115, 126, 6001, 6003, 6008, 6009, 6010 )
    labelsRight <- c( 204, 205, 206, 208, 209, 210, 214, 215, 226, 7001, 7003, 7008, 7009, 7010 )

    labels <- rep( 0, 1 + length( labelsLeft ) + length( labelsRight ) )
    labels[seq( 2, length( labels ), by = 2 )] <- labelsLeft
    labels[seq( 3, length( labels ), by = 2 )] <- labelsRight
    imageSize <- c( 64, 64, 128 )

    ################################
    #
    # Process left/right in split networks
    #
    ################################

    ################################
    #
    # Download spatial priors
    #
    ################################

    priorsLabelsFileNamePath <- getANTsXNetData( "deepFlashTemplate2Labels" )
    priorLabels <- antsImageRead( priorsLabelsFileNamePath )

    priorsImageLeftList <- list()
    for( i in seq.int( length( labelsLeft ) ) )
      {
      priorImage <- thresholdImage( priorLabels, labelsLeft[i], labelsLeft[i], 1, 0 )
      priorsImage <- antsCopyImageInfo( t1Preprocessed, priorImage )
      priorsImageLeftList[[i]] <- smoothImage( priorImage, 1.0 )
      }

    priorsImageRightList <- list()
    for( i in seq.int( length( labelsRight ) ) )
      {
      priorImage <- thresholdImage( priorLabels, labelsRight[i], labelsRight[i], 1, 0 )
      priorsImage <- antsCopyImageInfo( t1Preprocessed, priorImage )
      priorsImageRightList[[i]] <- smoothImage( priorImage, 1.0 )
      }

    probabilityImagesLeft <- list()
    foregroundProbabilityImagesLeft <- list()
    lowerBoundLeft <- c( 115, 109, 83 )
    upperBoundLeft <- c( 178, 172, 210 )
    tmpCropped <- cropIndices( t1Preprocessed, lowerBoundLeft, upperBoundLeft )
    originLeft <- antsGetOrigin( tmpCropped )

    spacing <- antsGetSpacing( tmpCropped )
    direction <- antsGetDirection( tmpCropped )

    t1TemplateRoiLeft <- cropIndices( t1Template, lowerBoundLeft, upperBoundLeft )
    t1TemplateRoiLeft <- iMath( t1TemplateRoiLeft, "Normalize" ) * 2.0 - 1.0

    probabilityImagesRight <- list()
    foregroundProbabilityImagesRight <- list()
    lowerBoundRight <- c( 51, 109, 83 )
    upperBoundRight <- c( 114, 172, 210 )
    tmpCropped <- cropIndices( t1Preprocessed, lowerBoundRight, upperBoundRight )
    originRight <- antsGetOrigin( tmpCropped )

    t1TemplateRoiRight <- cropIndices( t1Template, lowerBoundRight, upperBoundRight )
    t1TemplateRoiRight <- iMath( t1TemplateRoiRight, "Normalize" ) * 2.0 - 1.0

    ################################
    #
    # Create model
    #
    ################################

    channelSize <- 1 + length( labelsLeft )
    if( ! is.null( t2 ) )
      {
      channelSize <- channelSize + 1
      }
    numberOfClassificationLabels <- 1 + length( labelsLeft )

    unetModel <- createUnetModel3D( c( imageSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
      numberOfFilters = c( 32, 64, 96, 128, 256 ),
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      dropoutRate = 0.0, weightDecay = 0 )

    penultimateLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]$output

    # whole complex
    output1 <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
      activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )

    # hippocampus
    output2 <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
      activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )

    # amygdala
    output3 <- penultimateLayer %>% keras::layer_conv_3d( filters = 1, kernel_size = 1L,
      activation = 'sigmoid', kernel_regularizer = keras::regularizer_l2( 0.0 ) )

    unetModel <- keras::keras_model( inputs = unetModel$input,
                                     outputs = list( unetModel$output, output1, output2, output3 ) )

    ################################
    #
    # Left:  build model and load weights
    #
    ################################

    networkName <- 'deepFlash2LeftT1Hierarchical'

    if( verbose )
      {
      cat( "DeepFlash: retrieving model weights (left).\n" )
      }
    weightsFileName <- getPretrainedNetwork( networkName )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Left:  do prediction and normalize to native space
    #
    ################################

    if( verbose )
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
    t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiLeft, 255, 64, FALSE )

    batchX[1,,,,1] <- as.array( t1Cropped )
    if( useContralaterality )
      {
      t1Cropped <- cropIndices( t1PreprocessedFlipped, lowerBoundLeft, upperBoundLeft )
      t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiLeft, 255, 64, FALSE )
      batchX[2,,,,1] <- as.array( t1Cropped )
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

    for( i in seq.int( 1 + length( labelsLeft ) ) )
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

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
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
    # Left:  do prediction of whole, hippocampal, and amygdala regions and normalize to native space
    #
    ################################

    for( i in seq.int( 2, length( predictedData ) ) )
      {
      probabilityImagesList <- decodeUnet( predictedData[[i]], t1Cropped )
      for( j in seq.int( dim( predictedData[[i]] )[1] ) )
        {
        probabilityImage <- probabilityImagesList[[j]][[1]]
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )

        if( j == 2 ) # flipped
          {
          probabilityArray <- as.array( probabilityImage )
          probabilityArrayDimension <- dim( probabilityImage )
          probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
          probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
          }

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
              whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
          }

        if( j == 1 )  # not flipped
          {
          foregroundProbabilityImagesLeft <- append( foregroundProbabilityImagesLeft, probabilityImage )
          } else {    # flipped
          foregroundProbabilityImagesRight <- append( foregroundProbabilityImagesRight, probabilityImage )
          }
        }
      }

    ################################
    #
    # Right:  build model and load weights
    #
    ################################

    networkName <- 'deepFlash2RightT1Hierarchical'

    if( verbose )
      {
      cat( "DeepFlash: retrieving model weights (Right).\n" )
      }
    weightsFileName <- getPretrainedNetwork( networkName )
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Right:  do prediction and normalize to native space
    #
    ################################

    if( verbose )
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
    t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiRight, 255, 64, FALSE )
    batchX[1,,,,1] <- as.array( t1Cropped )
    if( useContralaterality )
      {
      t1Cropped <- cropIndices( t1PreprocessedFlipped, lowerBoundRight, upperBoundRight )
      t1Cropped <- histogramMatchImage( t1Cropped, t1TemplateRoiRight, 255, 64, FALSE )
      batchX[2,,,,1] <- as.array( t1Cropped )
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

    for( i in seq.int( 1 + length( labelsRight ) ) )
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

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
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
    # Right:  do prediction of whole, hippocampal, and amygdala regions and normalize to native space
    #
    ################################

    for( i in seq.int( 2, length( predictedData ) ) )
      {
      probabilityImagesList <- decodeUnet( predictedData[[i]], t1Cropped )
      for( j in seq.int( dim( predictedData[[i]] )[1] ) )
        {
        probabilityImage <- probabilityImagesList[[j]][[1]]
        probabilityImage <- decropImage( probabilityImage, t1Preprocessed * 0 )

        if( j == 2 ) # flipped
          {
          probabilityArray <- as.array( probabilityImage )
          probabilityArrayDimension <- dim( probabilityImage )
          probabilityArrayFlipped <- probabilityArray[probabilityArrayDimension[1]:1,,]
          probabilityImage <- as.antsImage( probabilityArrayFlipped, reference = probabilityImage )
          }

        if( doPreprocessing )
          {
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
              transformlist = templateTransforms$invtransforms,
              whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
          }

        if( j == 1 )  # not flipped
          {
          if( useContralaterality )
            {
            foregroundProbabilityImagesRight[[i-1]] <- ( foregroundProbabilityImagesRight[[i-1]] + probabilityImage ) / 2
            } else {
            foregroundProbabilityImagesRight <- append( foregroundProbabilityImagesRight, probabilityImage )
            }
          } else {    # flipped
          foregroundProbabilityImagesLeft[[i-1]] <- ( foregroundProbabilityImagesLeft[[i-1]] + probabilityImage ) / 2
          }
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

    ################################
    #
    # Convert probability images to segmentation
    #
    ################################

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

    foregroundProbabilityImages <- list()
    for( i in seq.int( length( foregroundProbabilityImagesLeft ) ) )
      {
      foregroundProbabilityImages[[i]] <- foregroundProbabilityImagesLeft[[i]] + foregroundProbabilityImagesRight[[i]]
      }

    results <- list( segmentationImage = relabeledImage,
                     probabilityImages = probabilityImages,
                     wholeProbabilityImage = foregroundProbabilityImages[[1]],
                     hippocampalProbabilityImage = foregroundProbabilityImages[[2]],
                     amygdalaProbabilityImage = foregroundProbabilityImages[[3]]
                  )
    return( results )
    } else {
    stop("Uncrecognized parcellation.")
    }
}

