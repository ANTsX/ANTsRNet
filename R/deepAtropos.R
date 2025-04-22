#' Six tissue segmentation
#'
#' Perform Atropos-style six tissue segmentation using deep learning
#'
#' The labeling is as follows:
#' \itemize{
#'   \item{Label 0:}{background}
#'   \item{Label 1:}{CSF}
#'   \item{Label 2:}{gray matter}
#'   \item{Label 3:}{white matter}
#'   \item{Label 4:}{deep gray matter}
#'   \item{Label 5:}{brain stem}
#'   \item{Label 6:}{cerebellum}
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
#' @param useSpatialPriors Use MNI spatial tissue priors (0 or 1).  Currently,
#' only '0' (no priors) and '1' (cerebellar prior only) are the only two options.
#' Default is 1.
#' @param verbose print progress.
#' @param debug return feature images in the last layer of the u-net model.
#' @return list consisting of the segmentation image and probability images for
#' each label.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' library( keras )
#'
#' image <- antsImageRead( "t1.nii.gz" )
#' results <- deepAtropos( image )
#' }
#' @export
deepAtropos <- function( t1, doPreprocessing = TRUE, useSpatialPriors = 1,
  verbose = FALSE, debug = FALSE )
{

  if( ! is.list( t1 ) )
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
    # Build model and load weights
    #
    ################################

    patchSize <- c( 112L, 112L, 112L )
    strideLength <- dim( t1Preprocessed ) - patchSize

    classes <- c( "background", "csf", "gray matter", "white matter",
      "deep gray matter", "brain stem", "cerebellum" )

    mniPriors <- NULL
    channelSize <- 1
    if( useSpatialPriors != 0 )
      {
      mniPriors <- splitNDImageToList( antsImageRead( getANTsXNetData( "croppedMni152Priors" ) ) )
      for( i in seq.int( length( mniPriors ) ) )
        {
        mniPriors[[i]] <- antsCopyImageInfo( t1Preprocessed, mniPriors[[i]] )
        }
      channelSize <- 2
      }

    unetModel <- createUnetModel3D( c( patchSize, channelSize ),
      numberOfOutputs = length( classes ), mode = 'classification',
      numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

    if( verbose )
      {
      cat( "DeepAtropos:  retrieving model weights.\n" )
      }
    weightsFileName <- ''
    if( useSpatialPriors == 0 )
      {
      weightsFileName <- getPretrainedNetwork( "sixTissueOctantBrainSegmentation" )
      } else if( useSpatialPriors == 1 ) {
      weightsFileName <- getPretrainedNetwork( "sixTissueOctantBrainSegmentationWithPriors1" )
      } else {
      stop( "useSpatialPriors must be a 0 or 1" )
      }
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose )
      {
      message( "Prediction.\n" )
      }

    t1Preprocessed <- ( t1Preprocessed - mean( t1Preprocessed ) ) / sd( t1Preprocessed )
    imagePatches <- extractImagePatches( t1Preprocessed, patchSize, maxNumberOfPatches = "all",
                                         strideLength = strideLength, returnAsArray = TRUE )
    batchX <- array( data = 0, dim = c( dim( imagePatches ), channelSize ) )
    batchX[,,,,1] <- imagePatches
    if( channelSize > 1 )
      {
      priorPatches <- extractImagePatches( mniPriors[[7]], patchSize, maxNumberOfPatches = "all",
                        strideLength = strideLength, returnAsArray = TRUE )
      batchX[,,,,2] <- priorPatches
      }
    predictedData <- unetModel %>% predict( batchX, verbose = verbose )

    probabilityImages <- list()
    for( i in seq.int( dim( predictedData )[5] ) )
      {
      if( verbose )
        {
        cat( "Reconstructing image ", classes[i], "\n" )
        }
      reconstructedImage <- reconstructImageFromPatches( predictedData[,,,,i],
          domainImage = t1Preprocessed, strideLength = strideLength )
      if( doPreprocessing )
        {
        probabilityImages[[i]] <- antsApplyTransforms( fixed = t1, moving = reconstructedImage,
            transformlist = t1Preprocessing$templateTransforms$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        } else {
        probabilityImages[[i]] <- reconstructedImage
        }
      }

    imageMatrix <- imageListToMatrix( probabilityImages, t1 * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
    segmentationImage <- matrixToImages( segmentationMatrix, t1 * 0 + 1 )[[1]] - 1

    results <- list( segmentationImage = segmentationImage,
                     probabilityImages = probabilityImages )

    # debugging

    if( debug )
      {
      inputImage <- unetModel$input
      featureLayer <- unetModel$layers[[length( unetModel$layers ) - 1]]
      featureFunction <- keras::backend()$`function`( list( inputImage ), list( featureLayer$output ) )
      featureBatch <- featureFunction( list( batchX[1,,,,,drop = FALSE] ) )

      featureImagesList <- decodeUnet( featureBatch[[1]], croppedImage )

      featureImages <- list()
      for( i in seq.int( length( featureImagesList[[1]] ) ) )
        {
        decroppedImage <- decropImage( featureImagesList[[1]][[i]], t1Preprocessed * 0 )
        if( doPreprocessing )
          {
          featureImages[[i]] <- antsApplyTransforms( fixed = t1, moving = decroppedImage,
              transformlist = t1Preprocessing$templateTransforms$invtransforms,
              whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
          } else {
          featureImages[[i]] <- decroppedImage
          }
        }
      results[['featureImagesLastLayer']] <- featureImages
      }
    return( results )

    } else {

    if( length( t1 ) != 3 )
      {
      stop( paste0( "Length of input list must be 3.  Input images are (in order): [T1, T2, FA].", 
                    "If a particular modality or modalities is not available, use NULL as a placeholder." ) )
      }

    if( is.null( t1[[1]] ) )
      {
      stop( "T1 modality must be specified." )
      }

    whichNetwork <- ""
    inputImages <- list()
    inputImages[[1]] <- t1[[1]]
    if( ! is.null( t1[[2]] ) && ! is.null( t1[[3]] ) )
      {
      whichNetwork = "t1_t2_fa" 
      inputImages[[2]] <- t1[[2]]
      inputImages[[3]] <- t1[[3]]
      } else if( ! is.null( t1[[2]] ) ) {
      whichNetwork = "t1_t2" 
      inputImages[[2]] <- t1[[2]]
      } else if( ! is.null( t1[[3]] ) ) {
      whichNetwork = "t1_fa" 
      inputImages[[2]] <- t1[[3]]
      } else {
      whichNetwork = "t1" 
      }

    if( verbose )
      {
      cat( "Prediction using ", whichNetwork ) 
      }  

    ################################
    #
    # Preprocess image
    #
    ################################

    truncateImageIntensity <- function( image, truncatedValues = c( 0.01, 0.99 ) )
      {
      truncatedImage <- antsImageClone( image )
      quantiles <- quantile( truncatedImage, truncatedValues )
      truncatedImage[image < quantiles[1]] <- quantiles[1]
      truncatedImage[image > quantiles[2]] <- quantiles[2]
      return( truncatedImage )
      }

    hcpT1Template <- antsImageRead( getANTsXNetData( "hcpinterT1Template" ) )
    hcpTemplateBrainMask <- antsImageRead( getANTsXNetData( "hcpinterTemplateBrainMask" ) )
    hcpTemplateBrainSegmentation <- antsImageRead( getANTsXNetData( "hcpinterTemplateBrainSegmentation" ) )

    hcpT1Template <- hcpT1Template * hcpTemplateBrainMask

    reg <- NULL
    t1Mask <- NULL
    preprocessedImages <- list()
    for( i in seq.int( length( inputImages ) ) )
      {
      n4 <- n4BiasFieldCorrection( truncateImageIntensity( inputImages[[i]] ), 
                                   mask = inputImages[[i]] * 0 + 1, 
                                   convergence = list( iters = c( 50, 50, 50, 50 ), tol = 0.0 ),
                                   rescaleIntensities = TRUE,
                                   verbose = verbose )
      if( i == 1 )
        {
        t1Bext <- brainExtraction( inputImages[[1]], modality = "t1threetissue", verbose = verbose )
        t1Mask <- thresholdImage(t1Bext$segmentationImage, 1, 1, 1, 0 )
        n4 <- n4 * t1Mask 
        reg <- antsRegistration( hcpT1Template, n4, 
                                 typeofTransform = "antsRegistrationSyNQuick[a]",
                                 verbose = verbose )
        preprocessedImages[[i]] <- antsImageClone( reg$warpedmovout )                         
        } else {
        n4 <- n4 * t1Mask 
        n4 <- antsApplyTransforms( hcpT1Template, n4, 
                                   transformlist = reg$fwdtransforms,
                                   verbose = verbose )
        preprocessedImages[[i]] <- n4
        }
      preprocessedImages[[i]] <- iMath( preprocessedImages[[i]], "Normalize" ) 
      }
     
    ################################
    #
    # Build model and load weights
    #
    ################################

    patchSize <- c( 192L, 224L, 192L )
    strideLength <- c( dim( hcpT1Template )[1] - patchSize[1],
                       dim( hcpT1Template )[2] - patchSize[2],  
                       dim( hcpT1Template )[3] - patchSize[3] )

    hcpTemplatePriors <- list()
    for( i in seq.int( 6 ) )
      {
      prior <- thresholdImage( hcpTemplateBrainSegmentation, i, i, 1, 0 ) 
      priorSmooth <- smoothImage( prior, 1.0 )
      hcpTemplatePriors[[i]] <- priorSmooth
      }

    classes <- c( "background", "csf", "gray matter", "white matter",
      "deep gray matter", "brain stem", "cerebellum" )
    numberOfClassificationLabels <- length( classes )  
    channelSize <- length( inputImages ) + length( hcpTemplatePriors )

    unetModel <- createUnetModel3D( c( patchSize, channelSize ),
      numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
      numberOfFilters = c( 16, 32, 64, 128 ), dropoutRate = 0.0,
      convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
      weightDecay = 0.0 )

    if( verbose )
      {
      cat( "DeepAtropos:  retrieving model weights.\n" )
      }
    weightsFileName <- ''
    if( whichNetwork == "t1" )
      {
      weightsFileName <- getPretrainedNetwork( "DeepAtroposHcpT1Weights" )
      } else if( whichNetwork == "t1_t2" ) {
      weightsFileName <- getPretrainedNetwork( "DeepAtroposHcpT1T2Weights" )
      } else if( whichNetwork == "t1_fa" ) {
      weightsFileName <- getPretrainedNetwork( "DeepAtroposHcpT1FAWeights" )
      } else if( whichNetwork == "t1_t2_fa" ) {
      weightsFileName <- getPretrainedNetwork( "DeepAtroposHcpT1T2FAWeights" )
      }
    load_model_weights_hdf5( unetModel, filepath = weightsFileName )

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if( verbose )
      {
      message( "Prediction.\n" )
      }

    predictedData <- array( data = 0, dim = c( 8, patchSize, numberOfClassificationLabels ) )
 
    batchX <- array( data = 0, dim = c( 1, patchSize, channelSize ) )

    for( h in seq.int( 8 ) )
      {
      index <- 1
      for( i in seq.int( length( preprocessedImages ) ) )
        {
        patches <- extractImagePatches( preprocessedImages[[i]],
                                        patchSize = patchSize,
                                        maxNumberOfPatches = "all", 
                                        strideLength = strideLength,
                                        returnAsArray = TRUE )
        batchX[1,,,,index] <- patches[h,,,]
        index <- index + 1
        }
      for( i in seq.int( length( hcpTemplatePriors ) ) )
        {
        patches <- extractImagePatches( hcpTemplatePriors[[i]],
                                        patchSize = patchSize,
                                        maxNumberOfPatches = "all", 
                                        strideLength = strideLength,
                                        returnAsArray = TRUE )
        batchX[1,,,,index] <- patches[h,,,]
        index <- index + 1
        }

      predictedData[h,,,,] <- unetModel %>% predict( batchX, verbose = verbose )
      }

    probabilityImages <- list()
    for( i in seq.int( dim( predictedData )[5] ) )
      {
      if( verbose )
        {
        cat( "Reconstructing image ", classes[i], "\n" )
        }
      reconstructedImage <- reconstructImageFromPatches( predictedData[,,,,i],
          domainImage = hcpT1Template, strideLength = strideLength )
      if( doPreprocessing )
        {
        probabilityImages[[i]] <- antsApplyTransforms( fixed = inputImages[[1]], 
            moving = reconstructedImage,
            transformlist = reg$invtransforms,
            whichtoinvert = c( TRUE ), interpolator = "linear", verbose = verbose )
        } else {
        probabilityImages[[i]] <- reconstructedImage
        }
      }

    imageMatrix <- imageListToMatrix( probabilityImages, inputImages[[1]] * 0 + 1 )
    segmentationMatrix <- matrix( apply( imageMatrix, 2, which.max ), nrow = 1 )
    segmentationImage <- matrixToImages( segmentationMatrix, inputImages[[1]] * 0 + 1 )[[1]] - 1

    results <- list( segmentationImage = segmentationImage,
                     probabilityImages = probabilityImages )

    return( results )
    }
}
