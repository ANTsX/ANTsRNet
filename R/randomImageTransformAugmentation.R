#' Apply random transforms to a predictor / outcome training image set
#'
#' The function will apply rigid, affine or deformable maps to an input set of
#' training images.  The reference image domain defines the space in which this
#' happens.
#'
#' @param imageDomain defines the spatial domain for all images.  NOTE: if the
#' input images do not match the spatial domain of the domain image, we
#' internally resample the target to the domain.  This may have unexpected
#' consequences if you are not aware of this.  This operation will test
#' \code{antsImagePhysicalSpaceConsistency} then call
#' \code{resampleImageToTarget} upon failure.
#' @param predictorImageList list of lists of image predictors
#' @param outcomeImageList optional list of image outcomes
#' @param n number of simulations to run
#' @param typeOfTransform one of the following options
#'   \code{c("Translation","Rigid","ScaleShear","Affine","Deformation",
#'   "AffineAndDeformation", "DeformationBasis")}
#' @param interpolator nearestNeighbor or linear (string) for predictor and
#' outcome images respectively
#' @param sdAffine roughly controls deviation from identity matrix
#' @param nControlPoints number of control points for simulated deformation
#' @param spatialSmoothing spatial smoothing for simulated deformation
#' @param composeToField defaults to FALSE, will return deformation fields
#' otherwise i.e. maps any transformation to a single deformation field.
#' @param numberOfCompositions integer greater than or equal to one
#' @param deformationBasis list containing deformationBasis set
#' @param directoryName where to write to disk (optional)
#' @return list (if no directory set) or boolean for success, failure
#' @author Avants BB
#' @seealso \code{\link{randomImageTransformBatchGenerator}}
#' @importFrom ANTsRCore getAntsrTransformFixedParameters iMath resampleImageToTarget applyAntsrTransform antsImagePhysicalSpaceConsistency antsrTransformFromDisplacementField makeImage smoothImage setAntsrTransformFixedParameters getAntsrTransformParameters setAntsrTransformParameters readAntsrTransform createAntsrTransform randomMask mergeChannels
#' @importFrom ANTsR composeTransformsToField
#' @importFrom stats rnorm
#' @examples
#'
#' library( ANTsR )
#' i1 = antsImageRead( getANTsRData( "r16" ) )
#' i2 = antsImageRead( getANTsRData( "r64" ) )
#' s1 = thresholdImage( i1, "Otsu", 3 )
#' s2 = thresholdImage( i2, "Otsu", 3 )
#' rand = randomImageTransformAugmentation( i1,
#'   list( list(i1), list(i2) ),  list( s1, s2 ) )
#'
#' @export randomImageTransformAugmentation
randomImageTransformAugmentation <- function(
  imageDomain,
  predictorImageList,
  outcomeImageList,
  n = 8,
  typeOfTransform = 'Affine',
  interpolator = c('linear','nearestNeighbor'),
  sdAffine = 1, # for rigid, affine trasformations, deviance from identity
  nControlPoints = 100, # for deformation
  spatialSmoothing = 3, # for deformation
  composeToField = FALSE, # maps any transformation to single deformation field
  numberOfCompositions = 4,
  deformationBasis,
  directoryName )
{

  admissibleTx = c(
    "Translation","Rigid","ScaleShear","Affine","Deformation",
    "AffineAndDeformation", "DeformationBasis" )
  if ( missing( directoryName ) ) returnList = TRUE else returnList = FALSE
  if ( !(typeOfTransform %in% admissibleTx ) )
    stop("!(typeOfTransform %in% admissibleTx ) - see help")
  if ( returnList ) {
    outputPredictorList = list()
    outputOutcomeList = list()
    outputRandomTransformList = list()
  }

  # get some reference parameters by a 0 iteration mapping
  refreg = antsRegistration( imageDomain, imageDomain,
    typeofTransform='Rigid', regIterations=c(1,0) )
  reftx = readAntsrTransform( refreg$fwdtransforms[1], imageDomain@dimension )
  fxparam = getAntsrTransformFixedParameters( reftx )

  # polar decomposition of X
  polarX <- function( X )
    {
    x_svd <- svd( X )
    P <- x_svd$u %*% diag(x_svd$d) %*% t(x_svd$u)
    Z <- x_svd$u %*% t(x_svd$v)
    if ( det( Z ) < 0 ) Z = Z * (-1)
    return(list(P = P, Z = Z, Xtilde = P %*% Z))
    }

  randAff <- function( img, fixedParams, txtype = 'Affine', sdAffine ) {
    loctx <- createAntsrTransform(
      precision="float", type="AffineTransform", dimension=img@dimension )
    setAntsrTransformFixedParameters( loctx, fixedParams )
    idparams = getAntsrTransformParameters( loctx )
    noisemat = stats::rnorm( length(idparams), mean=0, sd=sdAffine )
    if ( txtype == 'Translation' )
      noisemat[ 1:(length(idparams) - img@dimension ) ] = 0
    idparams = idparams + noisemat
    idmat = matrix( idparams[ 1:(length(idparams) - img@dimension ) ],
      ncol = img@dimension )
    idmat = polarX( idmat )
    if ( txtype == "Rigid" ) idmat = idmat$Z
    if ( txtype == "Affine" ) idmat = idmat$Xtilde
    if ( txtype == "ScaleShear" ) idmat = idmat$P
    idparams[ 1:(length(idparams) - img@dimension ) ] = as.numeric( idmat )
    setAntsrTransformParameters( loctx, idparams )
    return( loctx )
    }

  randWarp <- function(  img,
    nsamples = nControlPoints,
    mdval = 5, # dilation of point image
    sdval = 5, # standard deviation for noise image
    smval = spatialSmoothing, # smoothing of each component
    ncomp = numberOfCompositions,  # number of compositions,
    deformationBasis
    ) {
    # generate random points within the image domain
    mymask = img * 0 + 1
    fields = list( )
    for ( k in 1:img@dimension ) {
      randMask = randomMask( mymask, nsamples )
      aaDist = iMath( randMask, "MaurerDistance") +
        makeImage( mymask, rnorm( sum( mymask  ), sd=sdval ))
      bb = iMath( randMask, "MD", mdval ) %>% smoothImage( smval )
      fields[[ k ]] =  bb
      # * ( iMath( img, "Grad", 1.5 ) %>% iMath("Normalize") )
      }
    warpTx = antsrTransformFromDisplacementField( mergeChannels( fields ) )
    fields = list( )
    for ( i in 1:ncomp ) fields[[ i ]] = warpTx
    return( fields )
    # return( applyAntsrTransform( fields, data = img, reference = img ) )
    #   plot( randWarp( r16, mdval=5, sdval=5, smval=6, ncomp=6 ) )
  }

  for ( i in 1:n ) {
    # for each run, randomly select an input image
    selimg = sample( 1:length(predictorImageList) )[1]
    locimgpredictors = predictorImageList[ selimg ][[1]]
    locimgoutcome = outcomeImageList[ selimg ]
    if (  ! antsImagePhysicalSpaceConsistency(imageDomain,locimgoutcome[[1]]) |
     ! antsImagePhysicalSpaceConsistency(imageDomain,locimgpredictors[[1]])  ) {
      for ( kk in 1:length( locimgpredictors ) )
        locimgpredictors[[kk]] =
          resampleImageToTarget( locimgpredictors[[kk]], imageDomain,
            interpType =  interpolator[1] )
      locimgoutcome[[1]] =
        resampleImageToTarget( locimgoutcome[[1]], imageDomain,
        interpType = interpolator[2] )
      }
    # get simulated data
    if ( typeOfTransform == 'Deformation' )
      loctx = randWarp( imageDomain )
    if ( typeOfTransform == 'AffineAndDeformation' )
        loctx = c( randWarp( imageDomain ),
        randAff( imageDomain, fxparam,  'Affine', sdAffine ) )
    if ( typeOfTransform %in% admissibleTx[1:4] )
      loctx = randAff( imageDomain, fxparam,  typeOfTransform, sdAffine )
    # pass to output
    if ( returnList ) {
      for ( kk in 1:length( locimgpredictors ) )
        locimgpredictors[[kk]] =
          applyAntsrTransform( loctx, locimgpredictors[[kk]], imageDomain,
            interpolation =  interpolator[1] )
      locimgoutcome =
        applyAntsrTransform( loctx, locimgoutcome[[1]], imageDomain,
                  interpolation =  interpolator[2] )
      outputPredictorList[[i]] = locimgpredictors
      outputOutcomeList[[i]] = locimgoutcome
      outputRandomTransformList[[i]] = loctx
      } else { # write to disk with some specified naming convention
      # if ( i ==  1 ) derka create directory if needed
      stop("not yet implemented")
      }
    }
  if ( composeToField ) {
    for ( k in 1:length( outputRandomTransformList ) ) {
      if ( is.list(outputRandomTransformList[[ k ]]) )
        outputRandomTransformList[[ k ]] =
          composeTransformsToField( imageDomain,
            outputRandomTransformList[[ k ]] ) else {
              outputRandomTransformList[[ k ]] =
                composeTransformsToField( imageDomain,
                  list(outputRandomTransformList[[ k ]]) )
            }
      }
  }
  if ( returnList ) return(
    list(
      outputPredictorList = outputPredictorList,
      outputOutcomeList = outputOutcomeList,
      outputRandomTransformList = outputRandomTransformList
      )
    )
}











#' Generate transform parameters and transformed images
#'
#' The function will apply rigid, affine or deformable maps to an input set of
#' training images.  The reference image domain defines the space in which this
#' happens.  The outcome here is the transform parameters themselves.  This
#' is intended for use with low-dimensional transformations.
#'
#' @param imageDomain defines the spatial domain for all images.  NOTE: if the
#' input images do not match the spatial domain of the domain image, we
#' internally resample the target to the domain.  This may have unexpected
#' consequences if you are not aware of this.  This operation will test
#' \code{antsImagePhysicalSpaceConsistency} then call
#' \code{resampleImageToTarget} upon failure.
#' @param predictorImageList list of lists of image predictors
#' @param n number of simulations to run
#' @param typeOfTransform one of the following options
#'   \code{c("Translation","Rigid","ScaleShear","Affine", "DeformationBasis")}
#' @param interpolator nearestNeighbor or linear (string) for predictor images
#' @param sdTransform roughly controls deviation from identity parameters
#' @param spatialSmoothing spatial smoothing for simulated deformation
#' @param numberOfCompositions integer greater than or equal to one
#' @param deformationBasis list containing deformationBasis set
#' @param deformationBasisMeans list containing deformationBasis set means
#' @param deformationBasisSDs list containing deformationBasis standard deviations
#' @return list of transformed images and transform parameters
#' @author Avants BB
#' @seealso \code{\link{randomImageTransformParametersBatchGenerator}}
#' @examples
#'
#' library( ANTsR )
#' i1 = antsImageRead( getANTsRData( "r16" ) )
#' i2 = antsImageRead( getANTsRData( "r64" ) )
#' s1 = thresholdImage( i1, "Otsu", 3 )
#' s2 = thresholdImage( i2, "Otsu", 3 )
#' rand = randomImageTransformParametersAugmentation( i1,
#'   list( i1, i2 ) )
#'
#' @export randomImageTransformParametersAugmentation
randomImageTransformParametersAugmentation <- function(
  imageDomain,
  predictorImageList,
  n = 8,
  typeOfTransform = 'Affine',
  interpolator = 'linear',
  sdTransform = 1, # deviance from identity
  spatialSmoothing = 3, # for deformation
  numberOfCompositions = 4,
  deformationBasis,
  deformationBasisMeans, # mean values
  deformationBasisSDs  )  # sd values
{
  admissibleTx = c(
    "Translation","Rigid","ScaleShear","Affine", "DeformationBasis" )
  if ( !(typeOfTransform %in% admissibleTx ) )
    stop("!(typeOfTransform %in% admissibleTx ) - see help")
  if ( typeOfTransform == "DeformationBasis" & missing( deformationBasis ) )
    stop("deformationBasis is missing")
  outputPredictorList = list()
  outputParameterList = list()

  if ( ! missing( deformationBasis ) & missing( deformationBasisMeans ) )
    deformationBasisMeans = rep( 0, length( DeformationBasis ))

  if ( ! missing( deformationBasis ) & missing( deformationBasisSDs ) )
    deformationBasisSDs = rep( sdTransform, length( DeformationBasis ))

  # get some reference parameters by a 0 iteration mapping
  refreg = antsRegistration( imageDomain, imageDomain,
    typeofTransform='Rigid', regIterations=c(1,0) )
  reftx = readAntsrTransform( refreg$fwdtransforms[1], imageDomain@dimension )
  fxparam = getAntsrTransformFixedParameters( reftx )

  # polar decomposition of X
  polarX <- function( X )
    {
    x_svd <- svd( X )
    P <- x_svd$u %*% diag(x_svd$d) %*% t(x_svd$u)
    Z <- x_svd$u %*% t(x_svd$v)
    if ( det( Z ) < 0 ) Z = Z * (-1)
    return(list(P = P, Z = Z, Xtilde = P %*% Z))
    }

  randAff <- function( img, fixedParams, txtype = 'Affine', sdAffine ) {
    loctx <- createAntsrTransform(
      precision="float", type="AffineTransform", dimension=img@dimension )
    setAntsrTransformFixedParameters( loctx, fixedParams )
    idparams = getAntsrTransformParameters( loctx )
    noisemat = stats::rnorm( length(idparams), mean=0, sd=sdAffine )
    if ( txtype == 'Translation' )
      noisemat[ 1:(length(idparams) - img@dimension ) ] = 0
    idparams = idparams + noisemat
    idmat = matrix( idparams[ 1:(length(idparams) - img@dimension ) ],
      ncol = img@dimension )
    idmat = polarX( idmat )
    if ( txtype == "Rigid" ) idmat = idmat$Z
    if ( txtype == "Affine" ) idmat = idmat$Xtilde
    if ( txtype == "ScaleShear" ) idmat = idmat$P
    idparams[ 1:(length(idparams) - img@dimension ) ] = as.numeric( idmat )
    setAntsrTransformParameters( loctx, idparams )
    return( loctx )
    }

  basisParameters <- function(
    deformationBasisMeans, # mean values
    deformationBasisSDs    # sd values
    ) {
    parameters = rep( 0, length( deformationBasisMeans ) )
    for ( k in 1:length( parameters ) ) {
      parameters[ k ] = rnorm( 1, deformationBasisMeans[k],
        deformationBasisSDs[k] )
      }
    return( parameters )
  }

  basisWarp <- function(
    smval = spatialSmoothing, # smoothing of each component
    ncomp = numberOfCompositions,  # number of compositions,
    deformationBasis,  # basis set of deformations
    betaParameters    # beta values
    ) {
    # generate random points within the image domain
    mergedField = deformationBasis[[1]] * betaParameters[1]
    for ( k in 2:length( deformationBasis ) ) {
      mergedField = mergedField + deformationBasis[[k]] * betaParameters[k]
      }
    if ( smval > 0 )
      mergedField = smoothImage( mergedField, smval )
    warpTx = antsrTransformFromDisplacementField( mergedField )
    fields = list( )
    for ( i in 1:ncomp ) fields[[ i ]] = warpTx
    return( fields )
  }

  for ( i in 1:n ) {
    # for each run, randomly select an input image
    selimg = sample( 1:length(predictorImageList) )[1]
    locimgpredictors = predictorImageList[[ selimg ]]
    if (
     ! antsImagePhysicalSpaceConsistency(imageDomain, locimgpredictors )  ) {
      locimgpredictors =
          resampleImageToTarget( locimgpredictors, imageDomain,
            interpType =  interpolator[1] )
      }
    # get simulated data
    if ( typeOfTransform == 'DeformationBasis' ) {
      params = basisParameters( deformationBasisMeans, deformationBasisSDs )
      loctx = basisWarp( spatialSmoothing, numberOfCompositions,
        deformationBasis, params  )
      }
    if ( typeOfTransform %in% admissibleTx[1:4] ) {
      loctx = randAff( imageDomain, fxparam,  typeOfTransform, sdTransform )
      params = getAntsrTransformParameters( loctx )
      }
    # pass to output
    locimgpredictors =
      applyAntsrTransform( loctx, locimgpredictors, imageDomain,
        interpolation =  interpolator[1] )
    outputPredictorList[[i]] = locimgpredictors
    outputParameterList[[i]] = params
    }

  return(
    list(
      outputPredictorList = outputPredictorList,
      outputParameterList = outputParameterList
      )
    )
}
