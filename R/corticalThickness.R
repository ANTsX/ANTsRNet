#' Cortical thickness using deep learning
#'
#' Perform KellyKapowski cortical thickness using \code{deepAtropos} for
#' segmentation.  Description concerning implementaiton and evaluation:
#'
#'  \url{https://www.medrxiv.org/content/10.1101/2020.10.19.20215392v1}
#'
#' @param t1 input 3-D unprocessed T1-weighted brain image
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
corticalThickness <- function( t1, antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  if( t1@dimension != 3 )
    {
    stop( "Input image must be 3-D" )
    }

  atropos <- deepAtropos( t1, doPreprocessing = TRUE,
    antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )

  # Kelly Kapowski cortical thickness

  kkSegmentation <- atropos$segmentationImage
  kkSegmentation[kkSegmentation == 4] <- 3
  grayMatter <- atropos$probabilityImages[[3]]
  whiteMatter <- atropos$probabilityImages[[4]] + atropos$probabilityImages[[5]]
  kk <- kellyKapowski( s = kkSegmentation, g = grayMatter, w = whiteMatter,
                      its = 45, r = 0.025, m = 1.5, x = 0, t = 10, verbose = verbose )

  return( list(
          thicknessImage = kk,
          csfProbabilityImage = atropos$probabilityImages[[2]],
          grayMatterProbabilityImage = atropos$probabilityImages[[3]],
          whiteMatterProbabilityImage = atropos$probabilityImages[[4]],
          deepGrayMatterProbabilityImage = atropos$probabilityImages[[5]],
          brainStemProbabilityImage = atropos$probabilityImages[[6]],
          cerebellumProbabilityImage = atropos$probabilityImages[[7]]
        ) )
  }


#' Longitudinal cortical thickness using deep learning
#'
#' Perform KellyKapowski cortical thickness longitudinally using \code{deepAtropos}
#' for segmentation of the derived single-subject template.  It takes inspiration from
#' the work described here:
#'
#' \url{https://pubmed.ncbi.nlm.nih.gov/31356207/}
#'
#' @param t1s input list of 3-D unprocessed T1-weighted brain images from a single subject
#' @param template input image to define the orientation of the SST.  Can be a string
#' (see \code{getANTsXNetData}) or a specified template.
#' @param numberOfAffineRefinements Defines the number of sets of affine iterations
#' for refining the SST.
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @param verbose print progress.
#' @return List consisting of the SST, and a (sub-)list for each subject consisting of
#' the preprocessed image, cortical thickness image, segmentation probability images,
#' and affine mapping to the SST.
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
longitudinalCorticalThickness <- function( t1s, template = "oasis", numberOfAffineRefinements = 1,
  antsxnetCacheDirectory = NULL, verbose = FALSE )
  {

  ###################
  #
  #  Initial SST + optional affine refinement
  #
  ##################

  sst <- NULL
  if( is.character( template ) )
    {
    templateFileNamePath <- getANTsXNetData( template, antsxnetCacheDirectory = antsxnetCacheDirectory )
    sst <- antsImageRead( templateFileNamePath )
    } else {
    sst <- template
    }

  brainsOnly <- list()
  for( s in seq.int( numberOfAffineRefinements + 1 ) )
    {
    sstTmp <- antsImageClone( templateImage ) * 0

    if( verbose )
      {
      cat( "Affine refinement", i, "( out of", numberOfAffineRefinements, ")" )
      }

    if( s == 1 )
      {
      for( i in seq.int( length( t1s ) ) )
        {
        if( verbose )
          {
          cat( "Processing image", i, "( out of", length( t1s ), ")" )
          }
        t1Preprocessed <- preprocessBrainImage( t1s[[i]], truncateIntensity = c( 0.01, 0.99 ),
          doBrainExtraction = TRUE, templateTransformType = "antsRegistrationSyNQuick[r]",
          template = sst, doBiasCorrection = FALSE, returnBiasField = FALSE,
          doDenoising = FALSE, intensityNormalizationType == "01",
          antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )
        brainsOnly[[i]] <- t1sPreprocessed$preprocessedImage * t1sPreprocessed$brainMask
        sstTmp <- sstTmp + t1sPreprocessed$preprocessedImage
        }
      } else {
      sstMask <- brainExtraction( sst,
        antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )
      sstBrainOnly <- sst * sstMask

      for( i in seq.int( length( t1s ) ) )
        {
        if( verbose )
          {
          cat( "Processing image", i, "( out of", length( t1s ), ")" )
          }
        sstRegistration <- antsRegistration( fixed = sstBrainOnly, moving = brainsOnly[[i]],
          typeofTransform = "antsRegistrationSyNQuick[a]" )
        sstTmp <- sstTmp + sstRegistration$warpedmovout
        }
      }
    sst <- sstTmp / length( t1s )
    }

  ###################
  #
  #  Preprocessing and affine transform to final SST
  #
  ##################

  t1sPreprocessed <- list()
  for( i in seq.int( length( t1s ) ) )
    {
    if( verbose )
      {
      cat( "Processing image", i, "( out of", length( t1s ), ")" )
      }
    t1sPreprocessed[[i]] <- preprocessBrainImage( t1s[[i]], truncateIntensity = c( 0.01, 0.99 ),
      doBrainExtraction = TRUE, templateTransformType = "antsRegistrationSyNQuick[a]",
      template = sst, doBiasCorrection = TRUE, returnBiasField = FALSE,
      doDenoising = TRUE, intensityNormalizationType == "01",
      antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )
    }

  ###################
  #
  #  Deep Atropos of SST for priors
  #
  ##################

  sstAtropos <- deepAtropos( sst, doPreprocessing = TRUE,
    antsxnetCacheDirectory = antsxnetCacheDirectory, verbose = verbose )

  ###################
  #
  #  Traditional atropos + KK for each image
  #
  ##################

  returnList <- list( )
  for( i in seq.int( length( t1sPreprocessed ) ) )
    {
    if( verbose )
      {
      cat( "Atropos for image", i, "( out of", length( t1s ), ")" )
      }
    atroposOutput <- atropos( t1sPreprocessed[[i]]$preprocessedImage,
      x = t1sPreprocessed[[i]]$brainMask, i = sstAtropos$probabilityImages[2:7],
      m = "[0.1,1x1x1]", c = "[5,0]", priorweight = 0.25, p = "Socrates[1]",
      verbose = verbose )

    kkSegmentation <- atroposOutput$segmentationImage
    kkSegmentation[kkSegmentation == 4] <- 3
    grayMatter <- atroposOutput$probabilityImages[[2]]
    whiteMatter <- atroposOutput$probabilityImages[[3]] + atroposOutput$probabilityImages[[4]]
    kk <- kellyKapowski( s = kkSegmentation, g = grayMatter, w = whiteMatter,
                        its = 45, r = 0.025, m = 1.5, x = 0, t = 10, verbose = verbose )

    returnList[[i]] <- list(
          preprocessedImage = t1sPreprocessed[[i]]$preprocessedImage,
          thicknessImage = kk,
          csfProbabilityImage = atroposOutput$probabilityImages[[1]],
          grayMatterProbabilityImage = atropos$probabilityImages[[2]],
          whiteMatterProbabilityImage = atropos$probabilityImages[[3]],
          deepGrayMatterProbabilityImage = atropos$probabilityImages[[4]],
          brainStemProbabilityImage = atropos$probabilityImages[[5]],
          cerebellumProbabilityImage = atropos$probabilityImages[[6]],
          templateTransforms = t1sPreprocessed[[i]]$templateTransforms
        )
    }
  returnList[["singleSubjectTemplate"]] <- sst

  return( returnList )
  }
