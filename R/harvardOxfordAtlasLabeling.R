#' Subcortical and cerebellar labeling from a T1 image.
#'
#' Perform HOA labeling using deep learning and data from 
# "High Resolution, Comprehensive Atlases of the Human Brain 
#' Morphology" number: "NIH NIMH 5R01MH112748-04". Repository: 
#' https://github.com/HOA-2/SubcorticalParcellations'
#'
#' The labeling is as follows:
#' \itemize{
#' \item{Label 1:}{Lateral Ventricle Left}
#' \item{Label 2:}{Lateral Ventricle Right}
#' \item{Label 3:}{CSF}
#' \item{Label 4:}{Third Ventricle}
#' \item{Label 5:}{Fourth Ventricle}
#' \item{Label 6:}{5th Ventricle}
#' \item{Label 7:}{Nucleus Accumbens Left}
#' \item{Label 8:}{Nucleus Accumbens Right}
#' \item{Label 9:}{Caudate Left}
#' \item{Label 10:}{Caudate Right}
#' \item{Label 11:}{Putamen Left}
#' \item{Label 12:}{Putamen Right}
#' \item{Label 13:}{Globus Pallidus Left}
#' \item{Label 14:}{Globus Pallidus Right}
#' \item{Label 15:}{Brainstem}
#' \item{Label 16:}{Thalamus Left}
#' \item{Label 17:}{Thalamus Right}
#' \item{Label 18:}{Inferior Horn of the Lateral Ventricle Left}
#' \item{Label 19:}{Inferior Horn of the Lateral Ventricle Right}
#' \item{Label 20:}{Hippocampal Formation Left}
#' \item{Label 21:}{Hippocampal Formation Right}
#' \item{Label 22:}{Amygdala Left}
#' \item{Label 23:}{Amygdala Right}
#' \item{Label 24:}{Optic Chiasm}
#' \item{Label 25:}{VDC Anterior Left}
#' \item{Label 26:}{VDC Anterior Right}
#' \item{Label 27:}{VDC Posterior Left}
#' \item{Label 28:}{VDC Posterior Right}
#' \item{Label 29:}{Cerebellar Cortex Left}
#' \item{Label 30:}{Cerebellar Cortex Right}
#' \item{Label 31:}{Cerebellar White Matter Left}
#' \item{Label 32:}{Cerebellar White Matter Right}
#' }
#'
#' Preprocessing on the training data consisted of:
#'    * n4 bias correction,
#'    * brain extraction, and
#'    * affine registration to HCP.
#' The input T1 should undergo the same steps.  If the input T1 is the raw
#' T1, these steps can be performed by the internal preprocessing, i.e. set
#' \code{doPreprocessing = TRUE}
#'
#' @param t1 raw or preprocessed 3-D T1-weighted brain image.
#' @param doPreprocessing perform preprocessing.  See description above.
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
#' results <- harvardOxfordAtlasLabeling( image )
#' }
#' @export
harvardOxfordAtlasLabeling <- function( t1, doPreprocessing = TRUE,
  verbose = FALSE )
{

  if( t1@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  reshapeImage <- function( image, cropSize, interpType = "linear" )
    {
    imageResampled <- NULL
    if( interpType == "linear" )
      {
      imageResampled <- resampleImage( image, c( 1, 1, 1 ), useVoxels = FALSE, interpType = 0 )
      } else {
      imageResampled <- resampleImage( image, c( 1, 1, 1 ), useVoxels = FALSE, interpType = 1 )
      }
    imageCropped <- padOrCropImageToSize( imageResampled, cropSize ) 
    return( imageCropped )
    }

  whichTemplate <- "hcpyaT1Template"
  templateTransformType <- "antsRegistrationSyNQuick[a]"
  template <- antsImageRead( getANTsXNetData( whichTemplate ) )

  croppedTemplateSize <- c( 160, 176, 160 )

  ################################
  #
  # Preprocess image
  #
  ################################

  t1Preprocessed <- antsImageClone( t1, out_pixeltype = "float" )
  if( doPreprocessing )
    {
    t1Preprocessing <- preprocessBrainImage( t1,
        truncateIntensity = NULL,
        brainExtractionModality = "t1threetissue",
        template = whichTemplate,
        templateTransformType = templateTransformType,
        doBiasCorrection = TRUE,
        doDenoising = FALSE,
        verbose = verbose )
    t1Preprocessed <- t1Preprocessing$preprocessedImage * t1Preprocessing$brainMask
    t1Preprocessed <- reshapeImage( t1Preprocessed, cropSize = croppedTemplateSize )
    }

  ################################
  #
  # Build model and load weights
  #
  ################################

  hoaLateralLabels <- c( 0, 3, 4, 5, 6, 15, 24 )
  hoaLateralLeftLabels <- c( 1, 7, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31 )
  hoaLateralRightLabels <- c( 2, 8, 10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32 )
  hoaExtraLabels <- c( 33, 34, 35 )

  labels <- sort( c( hoaLateralLabels, hoaLateralLeftLabels, hoaExtraLabels ) )

  channelSize <- 1
  numberOfClassificationLabels <- length( labels )

  unetModelPre <- createUnetModel3D( c( croppedTemplateSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
    numberOfFilters = c( 16, 32, 64, 128 ), dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 0.0 )

  penultimateLayer <- unetModelPre$layers[[length( unetModelPre$layers ) - 1]]$output

  output2 <- penultimateLayer %>%
             keras::layer_conv_3d( filters = 1, 
                                   kernel_size = c( 1, 1, 1 ),
                                   activation = 'sigmoid',
                                   kernel_regularizer = keras::regularizer_l2( 0.0 ) )

  unetModel <- keras::keras_model( inputs = unetModelPre$input, 
                                   outputs = list( unetModelPre$output, output2 ) )
  weightsFileNamePath <- getPretrainedNetwork( "HarvardOxfordAtlasSubcortical" )
  keras::load_model_weights_hdf5( unetModel, filepath = weightsFileNamePath )

  ################################
  #
  # Do prediction and normalize to native space
  #
  ################################

  if( verbose )
    {
    cat( "Model prediction using both the original and contralaterally flipped version\n" )
    }

  batchX <- array( data = 0, dim = c( 2, croppedTemplateSize, channelSize ) )
  batchX[1,,,,1] <- as.array( iMath( t1Preprocessed, "Normalize" ) )
  batchX[2,,,,1] <- batchX[1,croppedTemplateSize[1]:1,,,1] 

  predictedData <- unetModel %>% predict( batchX, verbose = verbose )

  labels <- sort( c( hoaLateralLabels, hoaLateralLeftLabels ) )
  probabilityImages <- list()

  hoaLabels <- list()
  hoaLabels[[1]] <- hoaLateralLabels
  hoaLabels[[2]] <- hoaLateralLeftLabels
  # hoaLabels[[3]] <- hoaLateralRightLabels

  for( b in seq.int( 2 ) )
    {
    for( i in seq.int( length( hoaLabels ) ) ) 
      {
      for( j in seq.int( length( hoaLabels[[i]] ) ) )
        {
        label <- hoaLabels[[i]][j]   
        labelIndex <- which( labels == label )
        probabilityArray <- drop( predictedData[[1]][b,,,,labelIndex] )
        if( label == 0 )
          {
          label0_indices <- 21:dim(predictedData[[1]])[5]
          probabilityArray <- probabilityArray + drop( rowSums( predictedData[[1]][b,,,,label0_indices, drop=FALSE], dims = 4 ) )
          }
        if( b == 2 )
          {
          probabilityArray <- probabilityArray[dim( probabilityArray )[1]:1,,]
          }
        probabilityImage <- as.antsImage( probabilityArray, reference = t1Preprocessed )
        if( doPreprocessing )
          {
          probabilityImage <- padOrCropImageToSize( probabilityImage, dim( template ) )
          probabilityImage <- antsApplyTransforms( fixed = t1, moving = probabilityImage,
                  transformlist = t1Preprocessing$templateTransforms$invtransforms,
                    whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
          }
        if( b == 1 )  
          {
          probabilityImages[[labelIndex]] <- probabilityImage
          } else {
          probabilityImages[[labelIndex]] <- 0.5 * ( probabilityImages[[labelIndex]] + probabilityImage )
          }
        }
      }
    }

  if( verbose )
    {
    cat( "Constructing foreground probability image.\n" )
    }

  flippedPredictedData <- drop( predictedData[[2]][2,dim( predictedData[[2]] )[2]:1,,,] )
  foregroundProbabilityArray <- 0.5 * ( drop( predictedData[[2]][1,,,,] ) + flippedPredictedData )
  foregroundProbabilityImage <- as.antsImage( drop( foregroundProbabilityArray ), reference = t1Preprocessed )
  if( doPreprocessing )
    {
    foregroundProbabilityImage <- padOrCropImageToSize( foregroundProbabilityImage, dim( template ) )
    foregroundProbabilityImage <- antsApplyTransforms( fixed = t1, moving = foregroundProbabilityImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
              whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
    }

  for( i in seq.int( length( hoaLabels ) ) ) 
    {
    for( j in seq.int( length( hoaLabels[[i]] ) ) )
      {
      label <- hoaLabels[[i]][j]
      labelIndex <- which( labels == label ) 
      if( label == 0 )
        {
        probabilityImages[[labelIndex]] <- probabilityImages[[labelIndex]] * ( foregroundProbabilityImage * -1 + 1 )
        } else {
        probabilityImages[[labelIndex]] <- probabilityImages[[labelIndex]] * foregroundProbabilityImage
        }
      }  
    }

  labels <- sort( c( hoaLateralLabels, hoaLateralLeftLabels ) )

  bext <- brainExtraction( t1, modality = "t1hemi", verbose = verbose )

  probabilityAllImages <- list()
  count <- 1 
  for( i in seq.int( length( labels )  ) )
    {
    probabilityImage <- antsImageClone( probabilityImages[[i]], out_pixeltype = "float" )
    if( labels[i] %in% hoaLateralLeftLabels )
      {
      probabilityLeftImage <- probabilityImage * bext$probabilityImages[[2]]
      probabilityRightImage <- probabilityImage * bext$probabilityImages[[3]]
      probabilityAllImages[[count]] <- probabilityLeftImage
      count <- count + 1
      probabilityAllImages[[count]] <- probabilityRightImage
      count <- count + 1
      } else {
      probabilityAllImages[[count]] <- probabilityImage
      count <- count + 1
      }
    } 

  imageMatrix <- imageListToMatrix( probabilityAllImages, t1 * 0 + 1 )
  segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
  segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]] - 1

  hoaAllLabels <- sort( c( hoaLateralLabels, hoaLateralLeftLabels, hoaLateralRightLabels ) )   

  hoaLabelImage <- segmentationImage * 0
  for( i in seq.int( hoaAllLabels ) )
    {
    label <- hoaAllLabels[i] 
    labelIndex <- which( hoaAllLabels == label )
    hoaLabelImage[segmentationImage == labelIndex] <- label + 1
    }  

  return( list(
          segmentationImage = hoaLabelImage,
          probabilityImages = probabilityAllImages
          )
        )
}

