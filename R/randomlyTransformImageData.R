#' Randomly transform image data (optional: with corresponding segmentations). 
#'
#' Apply rigid, affine and/or deformable maps to an input set of training 
#' images.  The reference image domain defines the space in which this happens.
#'
#' @param referenceImage defines the spatial domain for all output images.  If 
#' the input images do not match the spatial domain of the reference image, we
#' internally resample the target to the reference image.  This could have 
#' unexpected consequences.  Resampling to the reference domain is performed by
#' testing using \code{antsImagePhysicalSpaceConsistency} then calling
#' \code{resampleImageToTarget} upon failure.
#' @param inputImageList list of lists of input images to warp.  The internal
#' list sets contains one or more images (per subject) which are assumed to be
#' mutually aligned.  The outer list contains multiple subject lists which are
#' randomly sampled to produce output image list.  
#' @param segmentationImageList list of segmentation images corresponding to the
#' input image list (optional).
#' @param numberOfSimulations number of output images.  Default = 10.
#' @param transformType one of the following options 
#' \code{c( "translation", "rigid", "scaleShear", "affine"," deformation" ,
#'   "affineAndDeformation" )}.  Default = \"affine\".
#' @param inputImageInterpolator one of the following options 
#' \code{ c( "linear", "gaussian", "bspline" )}.  Default = \"linear\".
#' @param segmentationImageInterpolator one of the following options 
#' \code{ c( "nearestNeighbor", "genericLabel" )}.  Default = 
#' \"nearestNeighbor\".
#' @param sdAffine parameter dictating deviation amount from identity for 
#' random linear transformations.  Default = 1.0.
#' @param numberOfControlPoints number of control points for simulated 
#' deformations. Default = 100.
#' @param spatialSmoothing amount of spatial smoothing for simulated 
#' deformations.  Default = 3.0.
#' @param sdNoise randomization parameter for simulated deformations.  
#' Default = 1.0.
#' @return list (if no directory set) or boolean for success, failure
#' @author Avants BB
#' @seealso \code{\link{randomImageTransformBatchGenerator}}
#' @importFrom ANTsRCore getAntsrTransformFixedParameters iMath resampleImageToTarget applyAntsrTransform antsImagePhysicalSpaceConsistency antsrTransformFromDisplacementField makeImage smoothImage setAntsrTransformFixedParameters getAntsrTransformParameters setAntsrTransformParameters readAntsrTransform createAntsrTransform randomMask mergeChannels 
#' @importFrom ANTsR composeTransformsToField
#' @importFrom stats rnorm
#' @examples
#'
#' library( ANTsR )
#' image1 <- antsImageRead( getANTsRData( "r16" ) )
#' image2 <- antsImageRead( getANTsRData( "r64" ) )
#' segmentation1 <- thresholdImage( image1, "Otsu", 3 )
#' segmentation2 <- thresholdImage( image2, "Otsu", 3 )
#' data <- randomlyTransformImageData( image1,
#'   list( list( image1 ), list( image2 ) ),  
#'   list( segmentation1, segmentation2 ) )
#' @export randomlyTransformImageData

randomlyTransformImageData <- function( referenceImage,
  inputImageList, segmentationImageList = NA, numberOfSimulations = 10,
  transformType = 'affine', inputImageInterpolator = 'linear',
  segmentationImageInterpolator = 'nearestNeighbor',
  sdAffine = 1.0, numberOfControlPoints = 100, 
  spatialSmoothing = 3.0, sdNoise = 1.0 )
{

  polarDecomposition <- function( X )
    {
    svdX <- svd( X )
    P <- svdX$u %*% diag( svdX$d ) %*% t( svdX$u )
    Z <- svdX$u %*% t( svdX$v )
    if ( det( Z ) < 0 ) 
      {
      Z <- Z * ( -1 )
      }
    return( list( P = P, Z = Z, Xtilde = P %*% Z ) )
    }

  createRandomLinearTransform <- function( 
    image, fixedParameters, transformType = 'affine', sdAffine = 1.0 ) 
    {
    transform <- createAntsrTransform( precision = "float", 
      type = "AffineTransform", dimension = image@dimension )
    setAntsrTransformFixedParameters( transform, fixedParameters )
    identityParameters <- getAntsrTransformParameters( transform )

    randomEpsilon <- stats::rnorm( length( identityParameters ), mean = 0, 
      sd = sdAffine )
    if ( transformType == 'translation' )
      {
      randomEpsilon[1:( length( identityParameters ) - image@dimension )] <- 0
      }

    randomParameters <- identityParameters + randomEpsilon
    randomMatrix <- matrix( randomParameters[
      1:( length( randomParameters ) - image@dimension )], 
        ncol = image@dimension )
    decomposition <- polarDecomposition( randomMatrix )

    if( transformType == "rigid" )
      { 
      randomMatrix <- decomposition$Z
      }
    if( transformType == "affine" ) 
      {
      randomMatrix <- decomposition$Xtilde
      }
    if( transformType == "scaleShear" ) 
      {
      randomMatrix <- decomposition$P
      }

    randomParameters[1:( length( identityParameters ) - image@dimension )] <- 
      as.numeric( randomMatrix )
    setAntsrTransformParameters( transform, randomParameters )
    return( transform )
    }

  createRandomDisplacementFieldTransform <- function( image, 
    numberOfControlPoints = 100, spatialSmoothing = 3.0, 
    sdNoise = 1.0, dilationRadius = 5, numberOfCompositions = 10, 
    gradientStep = 0.1 )
    {
    maskImage <- image * 0.0 + 1.0

    spacing <- antsGetSpacing( image )

    displacementFieldComponents <- list()
    for( d in 1:image@dimension ) 
      {
      randomMaskForTransform <- 
        randomMask( maskImage, numberOfControlPoints )
      voxelValues <- rnorm( numberOfControlPoints, sd = sdNoise * spacing[d] )
      minimumValue <- min( voxelValues ) 
      randomImage <- makeImage( randomMaskForTransform, ( voxelValues - minimumValue ) )
      fieldComponent <- iMath( randomImage, "GD", dilationRadius ) %>% 
        smoothImage( spatialSmoothing ) * gradientStep
      displacementFieldComponents[[d]] <- fieldComponent - 
        0.5 * ( max( fieldComponent ) - min( fieldComponent ) )
      }
    displacementField <- mergeChannels( displacementFieldComponents )
    displacementFieldTransform <- antsrTransformFromDisplacementField( displacementField )

    transforms <- list()
    for ( i in 1:numberOfCompositions ) 
      {
      transforms[[i]] <- displacementFieldTransform
      }

    composedTransformField <- antsrTransformFromDisplacementField( 
      composeTransformsToField( image, transforms ) )
    return( composedTransformField )
    }

  admissibleTransforms <- c( "translation", "rigid", "scaleShear", "affine", 
    "affineAndDeformation", "deformation" )
  if( !( transformType %in% admissibleTransforms ) )
    {
    stop( paste0( "The specified transform, ", transformType, 
      "is not a possible option.  Please see help menu." ) )    
    }

  # Get the fixed parameters from the reference image.

  referenceRegistration <- antsRegistration( referenceImage, referenceImage, 
    typeofTransform = 'Rigid', regIterations = c( 1, 0 ) )
  referenceTransform <- readAntsrTransform( 
    referenceRegistration$fwdtransforms[1], referenceImage@dimension )
  fixedParameters = getAntsrTransformFixedParameters( referenceTransform )

  numberOfSubjects <- length( inputImageList )   
  numberOfImagesPerSubject <- length( inputImageList[[1]] )

  randomIndices <- sample( numberOfSubjects, size = numberOfSimulations, 
    replace = TRUE )

  simulatedImageList <- list()
  simulatedSegmentationImageList <- list()
  for( i in seq_len( numberOfSimulations ) )
    {
    singleSubjectImageList <- inputImageList[[randomIndices[i]]] 
    singleSubjectSegmentationImage <- NA
    if( !is.na( segmentationImageList[[1]] ) )
      {
      singleSubjectSegmentationImage <- segmentationImageList[[randomIndices[i]]]
      }

    if( !antsImagePhysicalSpaceConsistency( referenceImage, singleSubjectImageList[[1]] ) )
      {
      for( j in 1:length( singleSubjectImageList ) )
        {
        singleSubjectImageList[[j]] <- resampleImageToTarget( 
          singleSubjectImageList[[j]], referenceImage, 
            interpType = inputImageInterpolator )
        }
      if( !is.na( singleSubjectSegmentationImage ) )
        {
        singleSubjectSegmentationImage <- resampleImageToTarget( 
          singleSubjectSegmentationImage, referenceImage,
            interpType = segmentationImageInterpolator )
        }    
      }

    transforms <- list()
    if( transformType == 'deformation' )
      {
      deformableTransform <- createRandomDisplacementFieldTransform(     
        referenceImage, numberOfControlPoints, spatialSmoothing, sdNoise )
      transforms <- list( deformableTransform )
      }  
    if( transformType == 'affineAndDeformation' )
      {
      deformableTransform <- createRandomDisplacementFieldTransform(     
        referenceImage, numberOfControlPoints, spatialSmoothing, sdNoise )
      linearTransform <- createRandomLinearTransform( referenceImage, 
        fixedParameters, 'affine', sdAffine )  
      transforms <- list( deformableTransform, linearTransform )
      }  
    if( transformType %in% admissibleTransforms[1:4] )
      {
      linearTransform <- createRandomLinearTransform( referenceImage, 
        fixedParameters, transformType, sdAffine )
      transforms <- list( linearTransform )
      }

    antsrTransform <- composeAntsrTransforms( transforms )

    singleSubjectSimulatedImageList <- list()
    for( j in 1:length( singleSubjectImageList ) )
      {
      singleSubjectSimulatedImageList[[j]] <- applyAntsrTransform( 
        antsrTransform, singleSubjectImageList[[j]], referenceImage, 
        interpolation = inputImageInterpolator )
      }
    simulatedImageList[[i]] <- singleSubjectSimulatedImageList

    if( !is.na( singleSubjectSegmentationImage ) )
      {
      simulatedSegmentationImageList[[i]] <- applyAntsrTransform( transforms, 
        singleSubjectSegmentationImage, referenceImage, 
          interpolation = segmentationImageInterpolator )
      }    
    }

  if( is.na( segmentationImageList[[1]] ) )
    {
    return( list( simulatedImageList = simulatedImageList ) )  
    }
  else 
    {
    return( list( simulatedImageList = simulatedImageList, 
      simulatedSegmentationImageList = simulatedSegmentationImageList ) )  
    }
}