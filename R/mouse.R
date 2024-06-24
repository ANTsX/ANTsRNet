#' Mouse brain extraction
#'
#' Perform brain extraction of mouse MRI
#'
#' @param image input 3-D brain image (or list of images for multi-modal scenarios).
#' @param modality image type.  Options include:
#' \itemize{
#'   \item{"t2": }{T2-weighted MRI}
#'   \item{"ex5coronal": }{E13.5 and E15.5 mouse embroyonic histology data.}
#'   \item{"ex5sagittal": }{E13.5 and E15.5 mouse embroyonic histology data.}
#' }
#' @param returnIsotropicOutput The network actually learns an interpolating 
#' function specific to the mouse brain.  Setting this to true, the output 
#' images are returned isotropically resampled.
#' @param whichAxis Specify direction for ex5 modalities..
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return brain probability mask 
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "brain.nii.gz" )
#' probabilityMask <- mouseBrainExtraction( image, modality = "t2" )
#' }
#' @export
mouseBrainExtraction <- function( image,
  modality = c( "t2", "ex5coronal", "ex5sagittal" ),
  returnIsotropicOutput = FALSE, whichAxis = 2,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( whichAxis < 1 || whichAxis > 3 )
    {
    stop( "Chosen axis not supported." )

    }

  if( modality == "t2" )
    {
    templateShape <- c( 176, 176, 176 ) 

    template <- antsImageRead( getANTsXNetData( "bsplineT2MouseTemplate" ) )
    template <- resampleImage( template, templateShape, useVoxels = TRUE, interpType = 0 )
    templateMask <- antsImageRead( getANTsXNetData( "bsplineT2MouseTemplateBrainMask" ) )
    templateMask <- resampleImage( template, templateShape, useVoxels = TRUE, interpType = 1 )

    if( verbose )
      {
      message( "Preprocessing:  Warping to B-spline T2w mouse template." )
      }

    centerOfMassReference <- getCenterOfMass( templateMask )
    centerOfMassImage <- getCenterOfMass( image )
    translation <- as.array( centerOfMassImage ) - as.array( centerOfMassReference )
    xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassReference,
      translation = centerOfMassImage - centerOfMassReference )

    imageWarped <- applyAntsrTransformToImage( xfrm, image, templateMask, interpolation = "linear" )
    imageWarped <- iMath( imageWarped, "Normalize" )

    unetModel <- createUnetModel3D( c( templateShape, 1 ),
      numberOfOutputs = 1, mode = "sigmoid",
      numberOfFilters = c( 16, 32, 64, 128 ), 
      convolutionKernelSize = 3, deconvolutionKernelSize = 2 )
    weightsFileName <- getPretrainedNetwork( "mouseT2wBrainExtraction3D" )
    unetModel$load_weights( weightsFileName )

    batchX <- array( data = 0, dim = c( 1, templateShape, 1 ) )
    batchX[1,,,,1] = as.array( imageWarped )

    if( verbose )
      {
      message( "Prediction." )
      }
    predictedData <- drop( unetModel$predict( batchX, verbose = verbose ) )
    predictedImage <- as.antsImage( predictedData, reference = template )

    probabilityMask <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      predictedImage, image )

    return( probabilityMask )
    }
  }


#' Mouse brain parcellation
#'
#' Perform brain extraction of mouse T2 MRI
#'
#' @param image input 3-D brain image (or list of images for multi-modal scenarios).
#' @param mask Brain mask.  If not specified, one is estimated using ANTsXNet mouse 
#' brain extraction.
#' @param returnIsotropicOutput The network actually learns an interpolating 
#' function specific to the mouse brain.  Setting this to true, the output 
#' images are returned isotropically resampled.
#' @param whichParcellation Brain parcellation type:
#' \itemize{
#'   \item{"nick":
#'     \itemize{
#'       \item{Label 0:}{background}
#'       \item{Label 1:}{cerebral cortex}
#'       \item{Label 2:}{cerebral nuclei}
#'       \item{Label 3:}{brain stem}
#'       \item{Label 4:}{cerebellum}
#'       \item{Label 5:}{main olfactory bulb}
#'       \item{Label 6:}{hippocampal formation}
#'     }}
#'   }
#' \itemize{
#'   \item{"tct":
#'     \itemize{
#'       \item{Label 0:}{}
#'       \item{Label 1:}{background}
#'       \item{Label 2:}{Infralimbic area}
#'       \item{Label 3:}{Prelimbic area}
#'       \item{Label 4:}{Medial group of the dorsal thalamus}
#'       \item{Label 5:}{Reticular nucleus of the thalamus}
#'       \item{Label 6:}{Hippocampal formation}
#'       \item{Label 7:}{Cerebellum}
#'     }}
#'   }
#' \itemize{
#'   \item{"jay":
#'     \itemize{
#'       \item{Label 0:}{background}
#'       \item{Label 1:}{}
#'       \item{Label 2:}{}
#'       \item{Label 3:}{}
#'       \item{Label 4:}{}
#'     }}
#'   }
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return brain probability mask 
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "brain.nii.gz" )
#' parcellation <- mouseBrainParcellation( image, modality = "t2" )
#' }
#' @export
mouseBrainParcellation <- function( image,
  mask = NULL, returnIsotropicOutput = FALSE,
  whichParcellation = c( "nick", "jay", "tct" ),
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {
  if( whichParcellation == "nick" || whichParcellation == "tct" || whichParcellation == "jay" )
    {
    templateSpacing <- c( 0.075, 0.075, 0.075 ) 
    templateCropSize <- c( 176, 176, 176 ) 

    if( whichParcellation == "nick" )
      {
      templateString <- "DevCCF P56 T2w"
      template <- antsImageRead( getANTsXNetData( "DevCCF_P56_MRI_T2_50um" ) )
      templateMatch <- rankIntensity( template )
      templateMask <- antsImageRead( getANTsXNetData( "DevCCF_P56_MRI_T2_50um_BrainParcellationNickMask" ) )
      weightsFileName <- getPretrainedNetwork( "mouseT2wBrainParcellation3DNick" )
      } else if( whichParcellation == "tct" ) {
      templateString <- "DevCCF P56 T2w"
      template <- antsImageRead( getANTsXNetData( "DevCCF_P56_MRI_T2_50um" ) )
      templateMatch <- rankIntensity( template )
      templateMask <- antsImageRead( getANTsXNetData( "DevCCF_P56_MRI_T2_50um_BrainParcellationTctMask" ) )
      weightsFileName <- getPretrainedNetwork( "mouseT2wBrainParcellation3DTct" )
      } else if( whichParcellation == "jay" ) {
      templateString <- "DevCCF P04 STPT"
      template <- antsImageRead( getANTsXNetData( "DevCCF_P04_STPT_50um" ) )
      templateMatch <- histogramEqualizeImage( template )
      templateMask <- antsImageRead( getANTsXNetData( "DevCCF_P04_STPT_50um_BrainParcellationJayMask" ) )
      weightsFileName <- getPretrainedNetwork( "mouseSTPTBrainParcellation3DJay" )
      }
    templateMatch <- (( templateMatch - ANTsR::min( templateMatch ) ) / 
                      ( ANTsR::max( templateMatch ) - ANTsR::min( templateMatch ) ))

    antsSetSpacing( template, c( 0.05, 0.05, 0.05 ) )
    template <- resampleImage( template, templateSpacing, useVoxels = FALSE, interpType = 4 )
    template <- padOrCropImageToSize( template, templateCropSize )

    antsSetSpacing( templateMask, c( 0.05, 0.05, 0.05 ) )
    templateMask <- resampleImage( templateMask, templateSpacing, useVoxels = FALSE, interpType = 1 )
    templateMask <- padOrCropImageToSize( templateMask, templateCropSize )

    numberOfNonzeroLabels <- length( sort( unique( as.vector( as.array( templateMask ) ) ) ) ) - 1

    templatePriors <- list()
    for( i in seq.int( numberOfNonzeroLabels ) )
      {
      singleLabel <- thresholdImage( templateMask, i, i, 1, 0 )
      prior <- smoothImage( singleLabel, sigma = 0.003, sigmaInPhysicalCoordinates = TRUE )
      templatePriors[[i]] <- prior
      }

    if( is.null( mask ) )
      {
      if( verbose )
        {
        message( "Preprocessing:  Brain extraction." )
        }
      mask <- mouseBrainExtraction( image, modality = "t2", 
                                    antsxnetCacheDirectory = antsxnetCacheDirectory, 
                                    verbose = verbose )
      mask <- thresholdImage( mask, 0.5, 1.1, 1, 0 )
      mask <- labelClusters( mask, fullyConnected = TRUE )
      mask <- thresholdImage( mask, 1, 1, 1, 0 )
      }

    imageBrain <- image * mask

    if ( verbose )
      {
      message( paste0( "Preprocessing:  Warping to ", templateString, " mouse template." ) )
      }

    reg <- antsRegistration( template, imageBrain, 
                             typeofTransform = "antsRegistrationSyNQuickRepro[a]", 
                             verbose = verbose )
    
    imageWarped <- NULL
    if ( whichParcellation == "nick" || whichParcellation == "tct" ) 
      {
      imageWarped <- rankIntensity( reg$warpedmovout )
      } else {
      imageWarped <- antsImageClone( reg$warpedmovout )
      }
    imageWarped <- histogramMatchImage( imageWarped, templateMatch )
    imageWarped <- iMath( imageWarped, "Normalize" )

    numberOfFilters <- c( 16, 32, 64, 128, 256 )
    numberOfClassificationLabels <- numberOfNonzeroLabels + 1
    channelSize = 1 + numberOfNonzeroLabels

    unetModel <- createUnetModel3D( c( dim( template ), channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = "classification",
      numberOfFilters = numberOfFilters, 
      convolutionKernelSize = 3, deconvolutionKernelSize = 2 )
    unetModel$load_weights( weightsFileName )

    batchX <- array( data = 0, dim = c( 1, dim( template ), channelSize ) )
    batchX[1,,,,1] = as.array( imageWarped )
    for( i in seq.int( length( templatePriors ) ) )
      {
      batchX[1,,,,i+1] <- as.array( templatePriors[[i]] )
      }

    if( verbose )
      {
      message( "Prediction." )
      }
    predictedData <- drop( unetModel$predict( batchX, verbose = verbose ) )

    referenceImage <- image
    if( returnIsotropicOutput )
      {
      newSpacing <- rep( min( antsGetSpacing( image ) ), 3 )
      referenceImage <- resampleImage( image, newSpacing, useVoxels = FALSE, interpType = 0 )
      }

    probabilityImages <- list()
    for( i in seq.int( numberOfClassificationLabels ) )
      {
      if( verbose )
        {
        message( "Reconstructing image ", i, "\n" )
        }
      probabilityImage <- as.antsImage( predictedData[,,,i], reference = template )
      probabilityImages[[i]] <- antsApplyTransforms( fixed = referenceImage, 
                            moving = probabilityImage, transformlist = reg$invtransforms,
                            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )                                 
      }

    imageMatrix <- imageListToMatrix( probabilityImages, referenceImage * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
    segmentationImage <- matrixToImages( segmentationMatrix, referenceImage * 0 + 1 )[[1]] - 1

    results <- list( segmentationImage = segmentationImage,
                      probabilityImages = probabilityImages )
    return( results )
    } else {
    stop( "Unrecognized parcellation." ) 
    }
  }


#' Mouse brain cortical thickness using deep learning
#'
#' Perform KellyKapowski cortical thickness
#'
#' @param t2 input 3-D unprocessed T2-weighted mouse brain image
#' @param mask  Brain mask.  If not specified, one is estimated using ANTsXNet mouse brain 
#' extraction.
#' @param returnIsotropicOutput The network actually learns an interpolating 
#' function specific to the mouse brain.  Setting this to true, the output 
#' images are returned isotropically resampled.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return Cortical thickness image and segmentation probability images.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1w_image.nii.gz" )
#' kk <- corticalThickness( image )
#' }
#' @export
mouseCorticalThickness <- function( t2, mask = NULL, 
  returnIsotropicOutput = FALSE, antsxnetCacheDirectory = NULL, 
  verbose = FALSE )
{

  parcellation <- mouseBrainParcellation( t2, mask = mask, whichParcellation = "nick",
                                          returnIsotropicOutput = TRUE, 
                                          antsxnetCacheDirectory = antsxnetCacheDirectory,
                                          verbose = verbose )

  # Kelly Kapowski cortical thickness

  kkSegmentation <- antsImageClone(parcellation$segmentationImage)
  kkSegmentation[kkSegmentation == 2] <- 3
  kkSegmentation[kkSegmentation == 1] <- 2
  kkSegmentation[kkSegmentation == 6] <- 2
  corticalMatter <- parcellation$probabilityImages[[2]] + parcellation$probabilityImages[[7]]
  otherMatter <- parcellation$probabilityImages[[3]] + parcellation$probabilityImages[[4]]

  kk <- kellyKapowski( s = kkSegmentation, g = corticalMatter, w = otherMatter,
                      its = 45, r = 0.0025, m = 1.5, x = 0, t = 10, verbose = verbose )

  if( ! returnIsotropicOutput )
    {
    kk <- resampleImage( kk, antsGetSpacing( t2 ), useVoxels = FALSE, interpType = 0 ) 
    parcellation$segmentationImage <- resampleImage( parcellation$segmentationImage, 
                                                     antsGetSpacing( t2 ), useVoxels = FALSE, 
                                                     interpType = 1 )
    for( i in seq.int( length( parcellation$probabilityImages ) ) )
      {
      parcellation$probabilityImages[[i]] <- resampleImage( parcellation$probabilityImages[[i]], 
                                                          antsGetSpacing( t2 ), useVoxels = FALSE, 
                                                          interpType = 0 )
      }                                                                                                        
    }

  return( list(
          thicknessImage = kk,
          parcellation = parcellation
        ) )
}
