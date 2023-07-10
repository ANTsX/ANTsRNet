#' MRI modality classification
#'
#' Predict MRI modality type (whole-head only).
#' Modalities:
#' \itemize{
#'     \item{T1}
#'     \item{T2}
#'     \item{FLAIR}
#'     \item{T2Star}
#'     \item{Mean DWI}
#'     \item{Mean Bold}
#'     \item{ASL perfusion}
#' }
#'
#' @param image raw 3-D MRI whole head image.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return classification data frame
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#'
#' image <- antsImageRead( getANTsXNetData( "mprageHippmap3r" ) )
#' classification <- mriModalityClassification( image )
#' }
#' @export
mriModalityClassification <- function( image,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( image@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  ################################
  #
  # Normalize to template
  #
  ################################

  imageSize <- c( 112, 112, 112 )
  resampleSize <- c( 2, 2, 2 )

  template <- antsImageRead( getANTsXNetData( "kirby" ) )
  template <- resampleImage( template, resampleSize )
  template <- padOrCropImageToSize( template, imageSize )
  direction <- antsGetDirection( template )
  direction[1, 1] <- 1.0
  antsSetDirection( template, direction )
  antsSetOrigin( template, c( 0, 0, 0 ) )

  centerOfMassTemplate <- getCenterOfMass( template*0 + 1 )
  centerOfMassImage <- getCenterOfMass( image*0 + 1 )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
    center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate )
  image <- applyAntsrTransformToImage( xfrm, image, template )  

  image <- ( image - min( image ) ) / ( max( image ) - min( image ) )

  ################################
  #
  # Load model and weights
  #
  ################################

  weightsFileName <- getPretrainedNetwork( "mriModalityClassification", 
                                           antsxnetCacheDirectory = antsxnetCacheDirectory )

  modalityTypes <- c( "T1", "T2", "FLAIR", "T2Star", "Mean DWI", "Mean Bold", "ASL Perfusion" )

  numberOfClassificationLabels <- length( modalityTypes )
  channelSize <- 1

  model <- createResNetModel3D( c( imageSize, channelSize ),
                                numberOfClassificationLabels = numberOfClassificationLabels,
                                mode = "classification",
                                layers = c( 1, 2, 3, 4 ),
                                residualBlockSchedule = c( 3, 4, 6, 3 ),
                                lowestResolution = 64,
                                cardinality = 1,
                                squeezeAndExcite = FALSE )

    model$load_weights( weightsFileName )

    batchX <- array( data = 0, dim = c( 1, imageSize, channelSize ) )
    batchX[1,,,,1] <- as.array( image )

    batchY <- model %>% predict( batchX, verbose = verbose )
    colnames( batchY ) <- modalityTypes

    return( batchY )
  }
