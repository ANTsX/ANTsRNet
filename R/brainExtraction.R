#' Brain extraction
#'
#' Perform T1, FA, or bold brain extraction using a U-net architecture
#' training data.  "NoBrainer" is also possible where
#' brain extraction uses U-net and FreeSurfer
#' training data ported from the
#'
#'  \url{https://github.com/neuronets/nobrainer-models}
#'
#' @param image input 3-D brain image (or list of images for multi-modal scenarios).
#' @param modality image type.  Options include:
#' \itemize{
#'   \item{"t1": }{T1-weighted MRI---ANTs-trained.  Previous versions are specified as "t1.v0", "t1.v1".}
#'   \item{"t1nobrainer": }{T1-weighted MRI---FreeSurfer-trained: h/t Satra Ghosh and Jakub Kaczmarzyk.}
#'   \item{"t1combined": }{Brian's combination of "t1" and "t1nobrainer".  One can also specify
#'                         "t1combined[X]" where X is the morphological radius.  X = 12 by default.}
#'   \item{"t1threetissue": }{T1-weighted MRI---originally developed from BrainWeb20 (and later expanded). 
#'                            Label 1: brain + subdural CSF, label 2: sinuses + skull, 
#'                            label 3: other head, face, neck tissue.}
#'   \item{"t1hemi": }{Label 1 of "t1threetissue" subdivided into left and right hemispheres.}
#'   \item{"flair": }{FLAIR MRI.}
#'   \item{"t2": }{T2-w MRI.}
#'   \item{"bold": }{3-D mean BOLD MRI.}
#'   \item{"fa": }{Fractional anisotropy.}
#'   \item{"t1t2infant": }{Combined T1-w/T2-w infant MRI h/t Martin Styner.}
#'   \item{"t1infant": }{T1-w infant MRI h/t Martin Styner.}
#'   \item{"t2infant": }{T2-w infant MRI h/t Martin Styner.}
#' }
#' @param verbose print progress.
#' @return brain probability mask (ANTsR image)
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' probabilityMask <- brainExtraction( image, modality = "t1" )
#' }
#' @export
brainExtraction <- function( image,
  modality = c( "t1", "t1.v0", "t1.v1", "t1nobrainer", "t1combined", "t1threetissue", 
                "t1hemi", "t1lobes", "t2", "t2.v0", "t2star", "flair", "flair.v0", 
                "bold", "bold.v0", "fa", "fa.v0", "mra", "t1t2infant", "t1infant", 
                "t2infant" ), 
  verbose = FALSE )
  {

  channelSize <- length( image )

  inputImages <- list()
  if( channelSize == 1 )
    {
    if( modality == "t1hemi" || modality == "t1lobes" ) 
      {
      bext <- brainExtraction( image, modality = "t1threetissue", verbose=verbose)
      mask <- thresholdImage( bext$segmentationImage, 1, 1, 1, 0 ) 
      inputImages[[1]] <- image * mask
      } else {
      inputImages[[1]] <- image
      }
    } else {
    inputImages <- image
    }

  if( inputImages[[1]]@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  inputImages <- antsImageTypeCast( inputImages, pixeltype = "float" )

  if( substr( modality, 1, 10 ) == "t1combined" )
    {

    # Need to change with voxel resolution
    morphologicalRadius <- 12
    if( grepl( '\\[', modality ) && grepl( '\\]', modality ) )
      {
      morphologicalRadius <- as.numeric( strsplit( strsplit( modality, "\\[" )[[1]][2], "\\]" )[[1]][1] )
      }

    brainExtraction_t1 <- brainExtraction( image, modality = "t1", verbose = verbose )
    brainMask <- thresholdImage( brainExtraction_t1, 0.5, Inf ) %>%
      morphology("close",morphologicalRadius) %>%
      iMath("FillHoles") %>%
      iMath( "GetLargestComponent" )

    brainExtraction_t1nobrainer <- brainExtraction( image * iMath( brainMask, "MD", morphologicalRadius ),
      modality = "t1nobrainer", verbose = verbose )
    brainExtraction_combined <- iMath( brainExtraction_t1nobrainer * brainMask, "GetLargestComponent" ) %>% iMath( "FillHoles" )

    brainExtraction_combined <- brainExtraction_combined + iMath( brainMask, "ME", morphologicalRadius ) + brainMask

    return( brainExtraction_combined )
    }

  if( modality != "t1nobrainer" )
    {

    #####################
    #
    # ANTs-based
    #
    #####################

    weightsFilePrefix <- ''
    isStandardNetwork <- FALSE

    if( modality == "t1.v0" )
      {
      weightsFilePrefix <- "brainExtraction"
      } else if( modality == "t1.v1" ) {
      weightsFilePrefix <- "brainExtractionT1v1"
      } else if( modality == "t1" ) {
      weightsFilePrefix <- "brainExtractionRobustT1"
      isStandardNetwork <- TRUE
      } else if( modality == "t2.v0" ) {
      weightsFilePrefix <- "brainExtractionT2"
      } else if( modality == "t2" ) {
      weightsFilePrefix <- "brainExtractionRobustT2"
      isStandardNetwork <- TRUE
      } else if( modality == "t2star" ) {
      weightsFilePrefix <- "brainExtractionRobustT2Star"
      isStandardNetwork <- TRUE
      } else if( modality == "flair.v0" ) {
      weightsFilePrefix <- "brainExtractionFLAIR"
      } else if( modality == "flair" ) {
      weightsFilePrefix <- "brainExtractionRobustFLAIR"
      isStandardNetwork <- TRUE
      } else if( modality == "bold.v0" ) {
      weightsFilePrefix <- "brainExtractionBOLD"
      } else if( modality == "bold" ) {
      weightsFilePrefix <- "brainExtractionRobustBOLD"
      isStandardNetwork <- TRUE
      } else if( modality == "fa.v0" ) {
      weightsFilePrefix <- "brainExtractionFA"
      } else if( modality == "fa" ) {
      weightsFilePrefix <- "brainExtractionRobustFA"
      isStandardNetwork <- TRUE
      } else if( modality == "mra" ) {
      weightsFilePrefix <- "brainExtractionMra"
      isStandardNetwork <- TRUE
      } else if( modality == "t1t2infant" ) {
      weightsFilePrefix <- "brainExtractionInfantT1T2"
      } else if( modality == "t1infant" ) {
      weightsFilePrefix <- "brainExtractionInfantT1"
      } else if( modality == "t2infant" ) {
      weightsFilePrefix <- "brainExtractionInfantT2"
      } else if( modality == "t1threetissue" ) {
      weightsFilePrefix <- "brainExtractionBrainWeb20"
      isStandardNetwork <- TRUE
      } else if( modality == "t1hemi" ) {
      weightsFilePrefix <- "brainExtractionT1Hemi"
      isStandardNetwork <- TRUE
      } else if( modality == "t1lobes" ) {
      weightsFilePrefix <- "brainExtractionT1Lobes"
      isStandardNetwork <- TRUE
      } else {
      stop( "Unknown modality type." )
      }

    if( verbose )
      {
      cat( "Brain extraction:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( weightsFilePrefix )

    if( verbose )
      {
      cat( "Brain extraction:  retrieving template.\n" )
      }
     
    if( modality == "t1threetissue" ) 
      {
      reorientTemplate <- antsImageRead( getANTsXNetData( "nki" ) )
      } else if( modality == "t1hemi" || modality == "t1lobes" ) {
      reorientTemplate <- antsImageRead( getANTsXNetData( "hcpyaT1Template" ) )
      reorientTemplateMask <- antsImageRead( getANTsXNetData( "hcpyaTemplateBrainMask" ) )
      reorientTemplate <- reorientTemplate * reorientTemplateMask
      reorientTemplate <- resampleImage( reorientTemplate, c( 1, 1, 1 ), useVoxels = FALSE, interpType = 0 )
      reorientTemplate <- padOrCropImageToSize( reorientTemplate, c( 160, 176, 160 ) )
      xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = getCenterOfMass( reorientTemplate ), translation = c( 0, -10, -15 ) )
      reorientTemplate <- applyAntsrTransformToImage( xfrm, reorientTemplate, reorientTemplate )
      } else {
      reorientTemplate <- antsImageRead( getANTsXNetData( "S_template3" ) )
      if( isStandardNetwork && ( modality != "t1.v1" && modality != "mra" ) )
        {
        antsSetSpacing( reorientTemplate, c( 1.5, 1.5, 1.5 ) )
        }
      } 
    resampledImageSize <- dim( reorientTemplate )

    numberOfFilters <- c( 8, 16, 32, 64 )
    mode <- "classification"
    numberOfClassificationLabels <- 2
    if( isStandardNetwork )
      {
      numberOfFilters <- c( 16, 32, 64, 128 )
      numberOfClassificationLabels <- 1
      mode <- "sigmoid"
      }

    unetModel <- NULL
    if( modality == "t1threetissue" || modality == "t1hemi" || modality == "t1lobes" )
      {
      mode <- "classification"
      if( modality == "t1threetissue" )
        {
        numberOfClassificationLabels <- 4 # background, brain, meninges/csf, misc. head
        } else if( modality == "t1hemi" ) {
        numberOfClassificationLabels <- 3 # background, left, right
        } else if( modality == "t1lobes" ) {
        numberOfClassificationLabels <- 6 # background, frontal, parietal, temporal, occipital, misc
        }
      unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
        numberOfOutputs = numberOfClassificationLabels, mode = mode,
        numberOfFilters = numberOfFilters, dropoutRate = 0.0,
        convolutionKernelSize = 3, deconvolutionKernelSize = 2,
        weightDecay = 0 )
      } else {
      unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
        numberOfOutputs = numberOfClassificationLabels, mode = mode,
        numberOfFilters = numberOfFilters, dropoutRate = 0.0,
        convolutionKernelSize = 3, deconvolutionKernelSize = 2,
        weightDecay = 1e-5 )
      }

    unetModel$load_weights( weightsFileName )

    if( verbose )
      {
      cat( "Brain extraction:  normalizing image to the template.\n" )
      }

    centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
    centerOfMassImage <- getCenterOfMass( inputImages[[1]] )
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )

    batchX <- array( data = 0, dim = c( 1, resampledImageSize, channelSize ) )

    for( i in seq.int( length( inputImages ) ) )
      {
      warpedImage <- applyAntsrTransformToImage( xfrm, inputImages[[i]], reorientTemplate )
      if( isStandardNetwork && modality != "t1.v1" )
        {
        batchX[1,,,,i] <- as.array( iMath( warpedImage, "Normalize" ) )
        } else {
        warpedArray <- as.array( warpedImage )
        batchX[1,,,,i] <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )
        }
      }

    if( verbose )
      {
      cat( "Brain extraction:  prediction and decoding.\n" )
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )
    probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

    if( verbose )
      {
      cat( "Brain extraction:  renormalize probability mask to native space.\n" )
      }

    xfrmInv <- invertAntsrTransform( xfrm )

    if( modality == "t1threetissue" || modality == "t1hemi" || modality == "t1lobes" )
      {
      probabilityImagesWarped <- list()
      for( i in seq.int( numberOfClassificationLabels ) )
        {
        probabilityImagesWarped[[i]] <- applyAntsrTransformToImage( xfrmInv,
                     probabilityImagesArray[[1]][[i]], inputImages[[1]] )               
        }
      imageMatrix <- imageListToMatrix( probabilityImagesWarped, inputImages[[1]] * 0 + 1 )
      segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
      segmentationImage <- matrixToImages( segmentationMatrix, inputImages[[1]] * 0 + 1 )[[1]] - 1
      
      results <- list( segmentationImage = segmentationImage,
                       probabilityImages = probabilityImagesWarped )

      return( results )
      } else {
      probabilityImage <- applyAntsrTransformToImage( xfrmInv,
        probabilityImagesArray[[1]][[numberOfClassificationLabels]], inputImages[[1]] )

      return( probabilityImage )
      }

    } else {

    #####################
    #
    # NoBrainer
    #
    #####################

    if( verbose )
      {
      cat( "NoBrainer:  generating network.\n")
      }
    model <- createNoBrainerUnetModel3D( list( NULL, NULL, NULL, 1 ) )

    if( verbose )
      {
      cat( "NoBrainer:  retrieving model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "brainExtractionNoBrainer" )
    model$load_weights( weightsFileName )

    if( verbose )
      {
      cat( "NoBrainer:  preprocessing (intensity truncation and resampling).\n" )
      }
    imageArray <- as.array( image )
    imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
    thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
    thresholdedMask <- thresholdImage( image, -10000, thresholdValue, 0, 1 )
    thresholdedImage <- image * thresholdedMask

    imageResampled <- resampleImage( image, rep( 256, 3 ), useVoxels = TRUE )
    imageArray <- array( as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )

    if( verbose )
      {
      cat( "NoBrainer:  predicting mask.\n" )
      }
    brainMaskArray <- predict( model, imageArray )
    brainMaskResampled <- as.antsImage( brainMaskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
    brainMaskImage = resampleImage( brainMaskResampled, dim( image ),
      useVoxels = TRUE, interpType = "nearestneighbor" )
    minimumBrainVolume <- round( 649933.7 / prod( antsGetSpacing( image ) ) )
    brainMaskLabeled = labelClusters( brainMaskImage, minimumBrainVolume )

    return( brainMaskLabeled )
    }
  }
