#' Nucleus Basalis of Meynert (NBM) Labeling
#'
#' Perform CH13 and Nucleus Basalis of Meynert (NBM) segmentation in
#' T1 images using Avants labels.
#'
#' The labeling is as follows:
#' \itemize{
#'   \item{Label 1: CH13_left}
#'   \item{Label 2: CH13_right}
#'   \item{Label 3: NBM_left_ant}
#'   \item{Label 4: NBM_left_mid}
#'   \item{Label 5: NBM_left_pos}
#'   \item{Label 6: NBM_right_ant}
#'   \item{Label 7: NBM_right_mid}
#'   \item{Label 8: NBM_right_pos}
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
#' @return A list containing the ANTs segmentation label image and probability images.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsR )
#' library( ANTsRNet )
#' image <- antsImageRead( getANTsRData( "r16" ) )
#' seg <- nbmLabeling( image )
#' }
#' @import keras
#' @export
nbmLabeling <- function(t1, verbose = FALSE) {
  
  if (t1@dimension != 3) {
    stop("Image dimension must be 3.")
  }
  
  whichTemplate <- "CIT168_T1w_700um_pad_adni"
  templateTransformType <- "antsRegistrationSyNReproQuick[a]"
  
  template <- antsImageRead(getANTsXNetData(whichTemplate))
  template <- rankIntensity(template)
  
  templateSeg <- antsImageRead(getANTsXNetData("CIT168_basal_forebrain_adni"))
  centerOfMassLabels <- c(1, 2, 3, 4)
  croppedTemplateSize <- c(144, 96, 64)
  
  ################################
  # Preprocess images
  ################################
  t1 <- antsImageClone(t1, out_pixeltype = "float")
  t1 <- iMath(t1, "Normalize")
  
  t1Preprocessing <- preprocessBrainImage(t1,
                                          truncateIntensity = c(1e-4, 0.999),
                                          brainExtractionModality = "t1threetissue",
                                          template = whichTemplate,
                                          templateTransformType = templateTransformType,
                                          doBiasCorrection = TRUE,
                                          doDenoising = FALSE,
                                          verbose = verbose)
  
  t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
  t1Preprocessed <- rankIntensity(t1Preprocessed)
  
  templateSegMasked <- maskImage(templateSeg, templateSeg, centerOfMassLabels, binarize = TRUE)
  com <- getCenterOfMass(templateSegMasked)
  
  t1Cropped <- cropImageFromCenterPoint(t1Preprocessed, com, croppedTemplateSize)
  
  ################################
  # Build model and load weights
  ################################
  nbmLateralLabels <- c(0)
  nbmLateralLeftLabels <- c(1, 3, 4, 5)
  nbmLateralRightLabels <- c(2, 6, 7, 8)
  
  labels <- sort(c(nbmLateralLabels, nbmLateralLeftLabels, nbmLateralRightLabels))
  numberOfOutputs <- length(labels)
  channelSize <- 1
  
  if (verbose) cat("    NBM: Creating model and loading weights.\n")
  
  unet0 <- createUnetModel3D(c(NA, NA, NA, channelSize),
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
  
  nextin <- layer_concatenate(list(unet0$inputs[[1]], unet0$outputs[[1]]), axis = -1L)
  unetModel <- keras_model(inputs = unet0$inputs, outputs = list(unet1(nextin), unet0$outputs[[1]]))
  
  nbmWeights <- getPretrainedNetwork("deep_nbm_rank")
  load_model_weights_hdf5(unetModel, nbmWeights)
  
  ################################
  # Do prediction and normalize
  ################################
  if (verbose) cat("Model prediction using both original and contralaterally flipped version.\n")
  
  batchX <- array(0, dim = c(2, croppedTemplateSize, channelSize))
  t1Array <- as.array(t1Cropped)
  
  batchX[1,,,,1] <- t1Array
  batchX[2,,,,1] <- t1Array[nrow(t1Array):1, , ]
  
  predictedData <- predict(unetModel, batchX, verbose = as.integer(verbose))
  
  probabilityImages <- list()
  
  for (i in 1:length(labels)) {
    label <- labels[i]
    
    if (label == 0) {
      probArray <- predictedData[[1]][1,,,,1]
      probArrayFlipped <- predictedData[[1]][2,,,,1]
      probArrayFlipped <- probArrayFlipped[nrow(probArrayFlipped):1, , ]
    } else {
      labelIndex <- which(labels == label)
      probArray <- predictedData[[1]][1,,,,labelIndex]
      
      if (label %in% nbmLateralLeftLabels) {
        leftIndex <- which(nbmLateralLeftLabels == label)
        rightLabel <- nbmLateralRightLabels[leftIndex]
        labelFlippedIndex <- which(labels == rightLabel)
      } else {
        rightIndex <- which(nbmLateralRightLabels == label)
        leftLabel <- nbmLateralLeftLabels[rightIndex]
        labelFlippedIndex <- which(labels == leftLabel)
      }
      
      probArrayFlipped <- predictedData[[1]][2,,,,labelFlippedIndex]
      probArrayFlipped <- probArrayFlipped[nrow(probArrayFlipped):1, , ]
    }
    
    avgProbArray <- 0.5 * (probArray + probArrayFlipped)
    probImage <- as.antsImage(avgProbArray, reference = t1Cropped)
    
    probImage <- antsApplyTransforms(fixed = t1, moving = probImage,
                                     transformlist = t1Preprocessing$templateTransforms$invtransforms,
                                     whichtoinvert = TRUE, interpolator = "linear", verbose = verbose)
    probabilityImages[[i]] <- probImage
  }
  
  imageMatrix <- imageListToMatrix(probabilityImages, t1 * 0 + 1)
  segmentationMatrix <- max.col(t(imageMatrix), ties.method = "first") - 1
  segmentationImage <- matrixToImages(matrix(segmentationMatrix, nrow=1), t1 * 0 + 1)[[1]]
  
  return(list(
    segmentationImage = segmentationImage,
    probabilityImages = probabilityImages
  ))
}