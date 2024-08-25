#' Cortical and deep gray matter labeling using Desikan-Killiany-Tourville
#'
#' Perform DKT labeling using deep learning
#'
#' The labeling is as follows:
#' \itemize{
#'  \item{Label 0:}{background}
#'
#'  Inner labels:
#'  \item{Label 4:}{left lateral ventricle}
#'  \item{Label 5:}{left inferior lateral ventricle}
#'  \item{Label 6:}{left cerebellem exterior}
#'  \item{Label 7:}{left cerebellum white matter}
#'  \item{Label 10:}{left thalamus proper}
#'  \item{Label 11:}{left caudate}
#'  \item{Label 12:}{left putamen}
#'  \item{Label 13:}{left pallidium}
#'  \item{Label 14:}{3rd ventricle}
#'  \item{Label 15:}{4th ventricle}
#'  \item{Label 16:}{brain stem}
#'  \item{Label 17:}{left hippocampus}
#'  \item{Label 18:}{left amygdala}
#'  \item{Label 24:}{CSF}
#'  \item{Label 25:}{left lesion}
#'  \item{Label 26:}{left accumbens area}
#'  \item{Label 28:}{left ventral DC}
#'  \item{Label 30:}{left vessel}
#'  \item{Label 43:}{right lateral ventricle}
#'  \item{Label 44:}{right inferior lateral ventricle}
#'  \item{Label 45:}{right cerebellum exterior}
#'  \item{Label 46:}{right cerebellum white matter}
#'  \item{Label 49:}{right thalamus proper}
#'  \item{Label 50:}{right caudate}
#'  \item{Label 51:}{right putamen}
#'  \item{Label 52:}{right palladium}
#'  \item{Label 53:}{right hippocampus}
#'  \item{Label 54:}{right amygdala}
#'  \item{Label 57:}{right lesion}
#'  \item{Label 58:}{right accumbens area}
#'  \item{Label 60:}{right ventral DC}
#'  \item{Label 62:}{right vessel}
#'  \item{Label 72:}{5th ventricle}
#'  \item{Label 85:}{optic chasm}
#'  \item{Label 91:}{left basal forebrain}
#'  \item{Label 92:}{right basal forebrain}
#'  \item{Label 630:}{cerebellar vermal lobules I-V}
#'  \item{Label 631:}{cerebellar vermal lobules VI-VII}
#'  \item{Label 632:}{cerebellar vermal lobules VIII-X}
#'
#'  Outer labels:
#'  \item{Label 1002:}{left caudal anterior cingulate}
#'  \item{Label 1003:}{left caudal middle frontal}
#'  \item{Label 1005:}{left cuneus}
#'  \item{Label 1006:}{left entorhinal}
#'  \item{Label 1007:}{left fusiform}
#'  \item{Label 1008:}{left inferior parietal}
#'  \item{Label 1009:}{left inferior temporal}
#'  \item{Label 1010:}{left isthmus cingulate}
#'  \item{Label 1011:}{left lateral occipital}
#'  \item{Label 1012:}{left lateral orbitofrontal}
#'  \item{Label 1013:}{left lingual}
#'  \item{Label 1014:}{left medial orbitofrontal}
#'  \item{Label 1015:}{left middle temporal}
#'  \item{Label 1016:}{left parahippocampal}
#'  \item{Label 1017:}{left paracentral}
#'  \item{Label 1018:}{left pars opercularis}
#'  \item{Label 1019:}{left pars orbitalis}
#'  \item{Label 1020:}{left pars triangularis}
#'  \item{Label 1021:}{left pericalcarine}
#'  \item{Label 1022:}{left postcentral}
#'  \item{Label 1023:}{left posterior cingulate}
#'  \item{Label 1024:}{left precentral}
#'  \item{Label 1025:}{left precuneus}
#'  \item{Label 1026:}{left rostral anterior cingulate}
#'  \item{Label 1027:}{left rostral middle frontal}
#'  \item{Label 1028:}{left superior frontal}
#'  \item{Label 1029:}{left superior parietal}
#'  \item{Label 1030:}{left superior temporal}
#'  \item{Label 1031:}{left supramarginal}
#'  \item{Label 1034:}{left transverse temporal}
#'  \item{Label 1035:}{left insula}
#'  \item{Label 2002:}{right caudal anterior cingulate}
#'  \item{Label 2003:}{right caudal middle frontal}
#'  \item{Label 2005:}{right cuneus}
#'  \item{Label 2006:}{right entorhinal}
#'  \item{Label 2007:}{right fusiform}
#'  \item{Label 2008:}{right inferior parietal}
#'  \item{Label 2009:}{right inferior temporal}
#'  \item{Label 2010:}{right isthmus cingulate}
#'  \item{Label 2011:}{right lateral occipital}
#'  \item{Label 2012:}{right lateral orbitofrontal}
#'  \item{Label 2013:}{right lingual}
#'  \item{Label 2014:}{right medial orbitofrontal}
#'  \item{Label 2015:}{right middle temporal}
#'  \item{Label 2016:}{right parahippocampal}
#'  \item{Label 2017:}{right paracentral}
#'  \item{Label 2018:}{right pars opercularis}
#'  \item{Label 2019:}{right pars orbitalis}
#'  \item{Label 2020:}{right pars triangularis}
#'  \item{Label 2021:}{right pericalcarine}
#'  \item{Label 2022:}{right postcentral}
#'  \item{Label 2023:}{right posterior cingulate}
#'  \item{Label 2024:}{right precentral}
#'  \item{Label 2025:}{right precuneus}
#'  \item{Label 2026:}{right rostral anterior cingulate}
#'  \item{Label 2027:}{right rostral middle frontal}
#'  \item{Label 2028:}{right superior frontal}
#'  \item{Label 2029:}{right superior parietal}
#'  \item{Label 2030:}{right superior temporal}
#'  \item{Label 2031:}{right supramarginal}
#'  \item{Label 2034:}{right transverse temporal}
#'  \item{Label 2035:}{right insula}
#' }
#'
#' Performing the lobar parcellation is based on the FreeSurfer division
#' described here:
#'
#'  \url{https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation}
#'
#' \itemize{
#'  Frontal lobe:
#'  \item{Label 1002:}{left caudal anterior cingulate}
#'  \item{Label 1003:}{left caudal middle frontal}
#'  \item{Label 1012:}{left lateral orbitofrontal}
#'  \item{Label 1014:}{left medial orbitofrontal}
#'  \item{Label 1017:}{left paracentral}
#'  \item{Label 1018:}{left pars opercularis}
#'  \item{Label 1019:}{left pars orbitalis}
#'  \item{Label 1020:}{left pars triangularis}
#'  \item{Label 1024:}{left precentral}
#'  \item{Label 1026:}{left rostral anterior cingulate}
#'  \item{Label 1027:}{left rostral middle frontal}
#'  \item{Label 1028:}{left superior frontal}
#'  \item{Label 2002:}{right caudal anterior cingulate}
#'  \item{Label 2003:}{right caudal middle frontal}
#'  \item{Label 2012:}{right lateral orbitofrontal}
#'  \item{Label 2014:}{right medial orbitofrontal}
#'  \item{Label 2017:}{right paracentral}
#'  \item{Label 2018:}{right pars opercularis}
#'  \item{Label 2019:}{right pars orbitalis}
#'  \item{Label 2020:}{right pars triangularis}
#'  \item{Label 2024:}{right precentral}
#'  \item{Label 2026:}{right rostral anterior cingulate}
#'  \item{Label 2027:}{right rostral middle frontal}
#'  \item{Label 2028:}{right superior frontal}
#'
#'  Parietal:
#'  \item{Label 1008:}{left inferior parietal}
#'  \item{Label 1010:}{left isthmus cingulate}
#'  \item{Label 1022:}{left postcentral}
#'  \item{Label 1023:}{left posterior cingulate}
#'  \item{Label 1025:}{left precuneus}
#'  \item{Label 1029:}{left superior parietal}
#'  \item{Label 1031:}{left supramarginal}
#'  \item{Label 2008:}{right inferior parietal}
#'  \item{Label 2010:}{right isthmus cingulate}
#'  \item{Label 2022:}{right postcentral}
#'  \item{Label 2023:}{right posterior cingulate}
#'  \item{Label 2025:}{right precuneus}
#'  \item{Label 2029:}{right superior parietal}
#'  \item{Label 2031:}{right supramarginal}
#'
#'  Temporal:
#'  \item{Label 1006:}{left entorhinal}
#'  \item{Label 1007:}{left fusiform}
#'  \item{Label 1009:}{left inferior temporal}
#'  \item{Label 1015:}{left middle temporal}
#'  \item{Label 1016:}{left parahippocampal}
#'  \item{Label 1030:}{left superior temporal}
#'  \item{Label 1034:}{left transverse temporal}
#'  \item{Label 2006:}{right entorhinal}
#'  \item{Label 2007:}{right fusiform}
#'  \item{Label 2009:}{right inferior temporal}
#'  \item{Label 2015:}{right middle temporal}
#'  \item{Label 2016:}{right parahippocampal}
#'  \item{Label 2030:}{right superior temporal}
#'  \item{Label 2034:}{right transverse temporal}
#'
#'  Occipital:
#'  \item{Label 1005:}{left cuneus}
#'  \item{Label 1011:}{left lateral occipital}
#'  \item{Label 1013:}{left lingual}
#'  \item{Label 1021:}{left pericalcarine}
#'  \item{Label 2005:}{right cuneus}
#'  \item{Label 2011:}{right lateral occipital}
#'  \item{Label 2013:}{right lingual}
#'  \item{Label 2021:}{right pericalcarine}
#'
#'  Other outer labels:
#'  \item{Label 1035:}{left insula}
#'  \item{Label 2035:}{right insula}
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
#' @param returnProbabilityImages whether to return the two sets of probability images
#' for the inner and outer labels.
#' @param doLobarParcellation perform lobar parcellation (also divided by hemisphere).
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
#' results <- desikanKillianyTourvilleLabeling( image )
#' }
#' @export
desikanKillianyTourvilleLabeling <- function( t1, doPreprocessing = TRUE,
  returnProbabilityImages = FALSE, doLobarParcellation = FALSE,
  verbose = FALSE )
{

  if( t1@dimension != 3 )
    {
    stop( "Input image dimension must be 3." )
    }

  ################################
  #
  # Preprocess image
  #
  ################################

  t1Preprocessed <- t1
  if( doPreprocessing )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = c( 0.01, 0.99 ),
        brainExtractionModality = "t1",
        template = "croppedMni152",
        templateTransformType = "antsRegistrationSyNQuickRepro[a]",
        doBiasCorrection = TRUE,
        doDenoising = TRUE,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    }

  ################################
  #
  # Download spatial priors for outer model
  #
  ################################

  if( verbose )
    {
    cat( "DesikanKillianyTourville:  retrieving label spatial priors.\n" )
    }
  priorsFileNamePath <- getANTsXNetData( "priorDktLabels" )
  priorsImageList <- splitNDImageToList( antsImageRead( priorsFileNamePath ) )

  ################################
  #
  # Build outer model and load weights
  #
  ################################

  templateSize <- c( 96L, 112L, 96L )
  labels <- c( 0, 1002:1003, 1005:1031, 1034:1035, 2002:2003, 2005:2031, 2034:2035 )
  channelSize <- 1 + length( priorsImageList )

  unetModel <- createUnetModel3D( c( templateSize, channelSize ),
    numberOfOutputs = length( labels ), mode = 'classification',
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

  if( verbose )
    {
    cat( "DesikanKillianyTourville:  retrieving model weights.\n" )
    }
  weightsFileNamePath <- getPretrainedNetwork( "dktOuterWithSpatialPriors" )
  load_model_weights_hdf5( unetModel, filepath = weightsFileNamePath )

  ################################
  #
  # Do prediction and normalize to native space
  #
  ################################

  if( verbose )
    {
    cat( "DesikanKillianyTourville:  outer model prediction.\n" )
    }

  downsampledImage <- resampleImage( t1Preprocessed, templateSize, useVoxels = TRUE, interpType = 0 )
  imageArray <- as.array( downsampledImage )
  imageArray <- ( imageArray - mean( imageArray ) ) / sd( imageArray )

  batchX <- array( data = 0, dim = c( 1, templateSize, channelSize ) )
  batchX[1,,,,1] <- imageArray

  for( i in seq.int( length( priorsImageList ) ) )
    {
    priorImageArray <- as.array( resampleImage( priorsImageList[[i]], templateSize, useVoxels = TRUE, interpType = 0 ) )
    batchX[1,,,,i+1] <- priorImageArray
    }

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  probabilityImagesList <- decodeUnet( predictedData, downsampledImage )

  outerProbabilityImages <- list()
  for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
    {
    resampledImage <- resampleImage( probabilityImagesList[[1]][[i]], dim( t1Preprocessed ), useVoxels = TRUE, interpType = 0 )
    if( doPreprocessing )
      {
      outerProbabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = resampledImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      } else {
      outerProbabilityImages[[i]] <- resampledImage
      }
    }

  imageMatrix <- imageListToMatrix( outerProbabilityImages, t1 * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]]

  dktLabelImage <- antsImageClone( segmentationImage )

  for( i in seq.int( length( labels ) ) )
    {
    dktLabelImage[( segmentationImage == i )] <- labels[i]
    }

  ################################
  #
  # Build inner model and load weights
  #
  ################################

  templateSize <- c( 160L, 192L, 160L )
  labels <- c( 0, 4, 6, 7, 10:18, 24, 26, 28, 30, 43, 44:46, 49:54, 58, 60, 91:92, 630:632 )

  unetModel <- createUnetModel3D( c( templateSize, 1 ),
    numberOfOutputs = length( labels ), mode = 'classification',
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

  if( verbose )
    {
    cat( "DesikanKillianyTourville:  retrieving inner model weights.\n" )
    }
  weightsFileName <- getPretrainedNetwork( "dktInner" )
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )

  ################################
  #
  # Do inner model prediction and normalize to native space
  #
  ################################

  if( verbose )
    {
    cat( "Inner model prediction.\n" )
    }

  croppedImage <- cropIndices( t1Preprocessed, c( 13, 15, 1 ), c( 172, 206, 160 ) )
  imageArray <- as.array( croppedImage )

  batchX <- array( data = imageArray, dim = c( 1, templateSize, 1 ) )
  batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )
  probabilityImagesList <- decodeUnet( predictedData, croppedImage )

  innerProbabilityImages <- list()
  for( i in seq.int( length( probabilityImagesList[[1]] ) ) )
    {
    if( i > 1 )
      {
      decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 )
      } else {
      decroppedImage <- decropImage( probabilityImagesList[[1]][[i]], t1Preprocessed * 0 + 1 )
      }
    if( doPreprocessing )
      {
      innerProbabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
      } else {
      innerProbabilityImages[[i]] <- decroppedImage
      }
    }

  imageMatrix <- imageListToMatrix( innerProbabilityImages, t1 * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]]

  ################################
  #
  # Incorporate the inner model results into the final label image.
  # Note that we purposely prioritize the inner label results.
  #
  ################################

  for( i in seq.int( length( labels ) ) )
    {
    if( labels[i] > 0 )
      {
      dktLabelImage[( segmentationImage == i )] <- labels[i]
      }
    }

  if( doLobarParcellation )
    {
    if( verbose )
      {
      cat( "Doing lobar parcellation.\n" )
      }

    ################################
    #
    # Lobar/hemisphere parcellation
    #
    ################################

    # Consolidate lobar cortical labels

    if( verbose )
      {
      cat( "   Consolidating cortical labels.\n" )
      }

    frontalLabels <- c( 1002, 1003, 1012, 1014, 1017:1020, 1024, 1026:1028,
                        2002, 2003, 2012, 2014, 2017:2020, 2024, 2026:2028 )
    parietalLabels <- c( 1008, 1010, 1022, 1023, 1025, 1029, 1031,
                        2008, 2010, 2022, 2023, 2025, 2029, 2031 )
    temporalLabels <- c( 1006, 1007, 1009, 1015, 1016, 1030, 1034,
                        2006, 2007, 2009, 2015, 2016, 2030, 2034 )
    occipitalLabels <- c( 1005, 1011, 1013, 1021,
                          2005, 2011, 2013, 2021 )

    lobarLabels <- list( frontalLabels, parietalLabels, temporalLabels, occipitalLabels )

    dktLobes <- antsImageClone( dktLabelImage )
    dktLobes[dktLobes < 1000] <- 0

    for( i in seq.int( length( lobarLabels ) ) )
      {
      for( j in seq.int( length( lobarLabels[[i]] ) ) )
        {
        dktLobes[dktLobes == lobarLabels[[i]][j]] <- i
        }
      }
    dktLobes[dktLobes > length( lobarLabels )] <- 0

    sixTissue <- deepAtropos( t1Preprocessed, doPreprocessing = FALSE, verbose = verbose )
    atroposSeg <- sixTissue$segmentationImage
    if( doPreprocessing )
      {
      atroposSeg <- antsApplyTransforms( fixed = t1, moving = atroposSeg,
          transformlist = t1Preprocessing$templateTransforms$invtransforms,
          whichtoinvert = c( TRUE ), interpolator = "genericLabel", verbose = verbose )
      }

    brainMask <- antsImageClone( atroposSeg )
    brainMask[brainMask == 1 | brainMask == 5 | brainMask == 6] <- 0
    brainMask <- thresholdImage( brainMask, 0, 0, 0, 1 )

    lobarParcellation <- iMath( brainMask, "PropagateLabelsThroughMask", brainMask * dktLobes )

    lobarParcellation[atroposSeg == 5] <- 5
    lobarParcellation[atroposSeg == 6] <- 6

    # Do left/right

    if( verbose )
      {
      cat( "   Doing left/right hemispheres.\n" )
      }

    leftLabels <- c( 4:7, 10:13, 17, 18, 25, 26, 28, 30, 91, 1002, 1003, 1005:1031, 1034, 1035 )
    rightLabels <- c( 43:46, 49:54, 57, 58, 60, 62, 92, 2002, 2003, 2005:2031, 2034, 2035 )

    hemisphereLabels <- list( leftLabels, rightLabels )

    dktHemispheres <- antsImageClone( dktLabelImage )

    for( i in seq.int( length( hemisphereLabels ) ) )
      {
      for( j in seq.int( length( hemisphereLabels[[i]] ) ) )
        {
        dktHemispheres[dktHemispheres == hemisphereLabels[[i]][j]] <- i
        }
      }
    dktHemispheres[dktHemispheres > 2] <- 0

    atroposBrainMask <- thresholdImage( atroposSeg, 0, 0, 0, 1 )
    hemisphereParcellation <- iMath( atroposBrainMask, "PropagateLabelsThroughMask", atroposBrainMask * dktHemispheres )

    for( i in seq.int( 6 ) )
      {
      lobarParcellation[lobarParcellation == i & hemisphereParcellation == 2] <- 6 + i
      }
    }

  if( returnProbabilityImages && doLobarParcellation )
    {
    return( list(
            segmentationImage = dktLabelImage,
            lobarParcellation = lobarParcellation,
            innerProbabilityImages = innerProbabilityImages,
            outerProbabilityImages = outerProbabilityImages
            )
          )
    } else if( returnProbabilityImages && ! doLobarParcellation ) {
    return( list(
            segmentationImage = dktLabelImage,
            innerProbabilityImages = innerProbabilityImages,
            outerProbabilityImages = outerProbabilityImages
            )
          )
    } else if( ! returnProbabilityImages && doLobarParcellation ) {
    return( list(
            segmentationImage = dktLabelImage,
            lobarParcellation = lobarParcellation
            )
          )
    } else {
    return( dktLabelImage )
    }
}

