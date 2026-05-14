#' CIT168 Labeling
#'
#' Perform CIT168 segmentation in T1 images using Pauli atlas (CIT168)
#' labels described in https://pubmed.ncbi.nlm.nih.gov/29664465/
#'
#' The labeling is as follows:
#' \itemize{
#'   \item{Label 1: BN_STR_Pu_Left}
#'   \item{Label 2: BN_STR_Ca_Left}
#'   \item{Label 3: BN_STR_NAC_Left (not modeled)}
#'   \item{Label 4: EXA_Left (not modeled)}
#'   \item{Label 5: BN_GP_GPe_Left}
#'   \item{Label 6: BN_GP_GPi_Left}
#'   \item{Label 7: MTg_SN_SNc_Left}
#'   \item{Label 8: MTg_RN_Left}
#'   \item{Label 9: MTg_SN_SNc_Left}
#'   \item{Label 10: MTg_VTR_PBP_Left (not modeled)}
#'   \item{Label 11: MTg_VTR_VTA_Left (not modeled)}
#'   \item{Label 12: BN_GP_VeP_Left (not modeled)}
#'   \item{Label 13: THM_ETH_HN_Left (not modeled)}
#'   \item{Label 14: Die_HTH_Left (not modeled)}
#'   \item{Label 15: Die_HTH_MN_Left (not modeled)}
#'   \item{Label 16: Die_STH_Left (not modeled)}
#'   \item{Label 17: BN_STR_Pu_Right}
#'   \item{Label 18: BN_STR_Ca_Right}
#'   \item{Label 19: BN_STR_NAC_Right (not modeled)}
#'   \item{Label 20: EXA_Right (not modeled)}
#'   \item{Label 21: BN_GP_GPe_Right}
#'   \item{Label 22: BN_GP_GPi_Right}
#'   \item{Label 23: MTg_SN_SNc_Right}
#'   \item{Label 24: MTg_RN_Right}
#'   \item{Label 25: MTg_SN_SNr_Right}
#'   \item{Label 26: MTg_VTR_PBP_Right (not modeled)}
#'   \item{Label 27: MTg_VTR_VTA_Right (not modeled)}
#'   \item{Label 28: BN_GP_VeP_Right (not modeled)}
#'   \item{Label 29: THM_ETH_HN_Right (not modeled)}
#'   \item{Label 30: Die_HTH_Right (not modeled)}
#'   \item{Label 31: Die_HTH_MN_Right (not modeled)}
#'   \item{Label 32: Die_STH_Right (not modeled)}
#'   \item{Label 33: ReferenceRegion_Left}
#'   \item{Label 34: ReferenceRegion_Right}
#' }
#'
#' Preprocessing consists of:
#' \itemize{
#'   \item{n4 bias correction}
#'   \item{brain extraction}
#' }
#'
#' @param t1 input ANTs image (3D).
#' @param verbose Print progress to the screen.
#'
#' @return ANTs segmentation label image.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsR )
#' library( ANTsRNet )
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' seg <- cit168Labeling( image )
#' }
#' @import keras
#' @import tensorflow
#' @export
cit168Labeling <- function(t1, verbose = FALSE) {
  
  if (t1@dimension != 3) {
    stop("Image dimension must be 3.")
  }

  template <- antsImageRead(getANTsXNetData("CIT168_T1w_700um_pad_adni"))
  templateSmall <- resampleImage(template, c(2, 2, 2), useVoxels = FALSE, interpType = 0)
  templateLarge <- resampleImage(template, c(0.5, 0.5, 0.5), useVoxels = FALSE, interpType = 0)  
  templateSeg <- antsImageRead(getANTsXNetData("det_atlas_25_pad_LR_adni"))

  templateTransformType <- "antsRegistrationSyNQuickRepro[s]"
  croppedTemplateSize <- c(160, 160, 112)
  
  ################################
  # Preprocess images
  ################################
  t1 <- antsImageClone(t1, out_pixeltype = "float")
  
  t1Preprocessing <- preprocessBrainImage(t1,
                                          truncateIntensity = c(1e-4, 0.999),
                                          brainExtractionModality = "t1threetissue",
                                          template = NULL,
                                          doBiasCorrection = TRUE,
                                          doDenoising = FALSE,
                                          intensityNormalizationType = "01",
                                          verbose = verbose)
  
  t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
  
  reg <- antsRegistration(templateSmall, t1Preprocessed, 
                          typeofTransform = templateTransformType, verbose = verbose)
  
  t1Preprocessed <- antsApplyTransforms(fixed = templateLarge, moving = t1Preprocessed,
                                  transformlist = reg$fwdtransforms[2],
                                  interpolator = "linear", verbose = verbose)
  templatePriors <- antsApplyTransforms(fixed = t1Preprocessed, moving = templateSeg,
                                        transformlist = reg$invtransforms[2],
                                        interpolator = "nearestNeighbor", verbose = verbose)                                  
  
  templateBin <- thresholdImage(templatePriors, 0, 0, 0, 1)
  centerOfMass <- getCenterOfMass(templateBin)
  centerOfMass[2] <- centerOfMass[2] + 10.0 
  
  t1Cropped <- cropImageFromCenterPoint(t1Preprocessed, centerOfMass, croppedTemplateSize)
  templatePriorsCropped <- cropImageFromCenterPoint(templatePriors, centerOfMass, croppedTemplateSize)
  
  ################################
  # Build model and load weights
  ################################
  cit168SegmentationImage <- t1 * 0
  
  for (sn in c(TRUE, FALSE)) {
    if (verbose) cat("    CIT168: Creating model and loading weights. (sn=", sn, ")\n")
    
    if (sn) {
      groupLabels <- c(0, 7, 8, 9, 23, 24, 25, 33, 34)
      cit168Weights <- getPretrainedNetwork("deepCIT168_sn")
    } else {
      groupLabels <- c(0, 1, 2, 5, 6, 17, 18, 21, 22)
      cit168Weights <- getPretrainedNetwork("deepCIT168")
    }
    
    numberOfOutputs <- length(groupLabels)
    numberOfChannels <- length(groupLabels)
    
    unet0 <- createUnetModel3D(c(NA, NA, NA, numberOfChannels),
                               numberOfOutputs = 1, numberOfLayers = 4,
                               numberOfFiltersAtBaseLayer = 32,
                               convolutionKernelSize = c(3, 3, 3),
                               deconvolutionKernelSize = c(2, 2, 2),
                               poolSize = c(2, 2, 2), strides = c(2, 2, 2), dropoutRate = 0.0,
                               weightDecay = 0,
                               additionalOptions = "nnUnetActivationStyle",
                               mode = "sigmoid")
    
    unet1 <- createUnetModel3D(c(NA, NA, NA, 2),
                               numberOfOutputs = numberOfOutputs,
                               numberOfFilters = c(32, 64, 96, 128, 256),
                               convolutionKernelSize = c(3, 3, 3),
                               deconvolutionKernelSize = c(2, 2, 2),
                               dropoutRate = 0.0, weightDecay = 0,
                               additionalOptions = "nnUnetActivationStyle",
                               mode = "classification")
    
    splits <- tf$split(unet0$inputs[[1]], as.integer(9), axis = -1L)
    newmult <- layer_concatenate(list(splits[[1]], unet0$outputs[[1]]), axis = -1L)
    unetonnet <- unet1(newmult)
    unetModel <- keras_model(inputs = unet0$inputs, outputs = list(unetonnet, unet0$outputs[[1]]))
    
    load_model_weights_hdf5(unetModel, cit168Weights)
    
    priorsArray <- as.array(templatePriorsCropped)
    priorsOneHot <- array(0, dim = c(dim(priorsArray), length(groupLabels) - 1))
    for (i in 2:length(groupLabels)) {
      priorsOneHot[,,,i-1][priorsArray == groupLabels[i]] <- 1
    }
    
    t1Array <- as.array(t1Cropped)
    batchX <- array(0, dim = c(1, dim(t1Array), numberOfChannels))
    batchX[1,,,,1] <- t1Array
    batchX[1,,,,2:numberOfChannels] <- priorsOneHot
    
    if (verbose) cat("    CIT168: Model prediction.\n")
    predictedData <- predict(unetModel, batchX, verbose = as.integer(verbose))
    
    probabilityImages <- list()
    for (i in 2:length(groupLabels)) {
      probArray <- predictedData[[1]][1,,,,i]
      probImg <- as.antsImage(probArray, reference = t1Cropped)
      
      probImg <- antsApplyTransforms(fixed = t1, moving = probImg,
                                     transformlist = reg$invtransforms[1], whichtoinvert = c(TRUE),
                                     interpolator = "linear", verbose = verbose)
      probabilityImages[[i-1]] <- probImg
    }
    
    predMaskArray <- predictedData[[2]][1,,,,1]
    predMask <- as.antsImage(predMaskArray, reference = t1Cropped)
    predMask <- antsApplyTransforms(fixed = t1, moving = predMask,
                                    transformlist = reg$invtransforms[1], whichtoinvert = c(TRUE),
                                    interpolator = "linear", verbose = verbose)
    predMask <- thresholdImage(predMask, 0.5, 1, 1, 0)
    
    imageMatrix <- imageListToMatrix(probabilityImages, predMask)
    segmentationMatrix <- max.col(t(imageMatrix), ties.method = "first")
    segmentationImage <- matrixToImages(matrix(segmentationMatrix, nrow=1), predMask)[[1]]
    
    relabeledSegmentationImage <- segmentationImage * 0
    for (i in 2:length(groupLabels)) {
      singleLabelImage <- thresholdImage(segmentationImage, i-1, i-1, 1, 0)
      if (groupLabels[i] < 33) {
        singleLabelImage <- labelClusters(singleLabelImage, minClusterSize = 1)
        singleLabelImage <- thresholdImage(singleLabelImage, 1, 1, 1, 0)
      }
      relabeledSegmentationImage <- relabeledSegmentationImage + (singleLabelImage * groupLabels[i])
    }
    cit168SegmentationImage <- cit168SegmentationImage + relabeledSegmentationImage
  }
  
  return(cit168SegmentationImage)
}