#' Apply random transforms to a predictor / outcome training image set
#'
#' The function will apply rigid, affine or deformable maps to an input set of
#' training images.  The reference image domain defines the space in which this
#' happens.
#'
#' @param imageDomain defines the spatial domain for all images.  NOTE: if the
#' input images do not match the spatial domain of the domain image, we
#' internally resample the target to the domain.  This may have unexpected
#' consequences if you are now aware of this.  This operation will test
#' \code{antsImagePhysicalSpaceConsistency} then call
#' \code{resampleImageToTarget} upon failure.
#' @param predictorImageList list of lists of image predictors
#' @param outcomeImageList list of image outcomes
#' @param n number of simulations to run
#' @param typeOfTransform one of the following options
#'   \code{c("Translation","Rigid","ScaleShear","Affine","Deformation",
#'   "AffineAndDeformation")}
#' @param interpolator nearestNeighbor or linear (string) for predictor and
#' outcome images respectively
#' @param sdAffine roughly controls deviation from identity matrix
#' @param nControlPoints number of control points for simulated deformation
#' @param spatialSmoothing spatial smoothing for simulated deformation
#' @param composeToField defaults to FALSE, will return deformation fields
#' otherwise i.e. maps any transformation to a single deformation field.
#' @param directoryName where to write to disk (optional)
#' @return list (if no directory set) or boolean for success, failure
#' @author Avants BB
#' @examples
#'
#' i1 = antsImageRead( getnANTsRData( "r16" ) )
#' i2 = antsImageRead( getnANTsRData( "r64" ) )
#' s1 = thresholdImage( i1, "Otsu", 3 )
#' s2 = thresholdImage( i2, "Otsu", 3 )
#' rand = randomImageTransformAugmentation( i1,
#'   list( i1, i2 ),  list( s1, s2 ) )
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
  directoryName )
{

  admissibleTx = c(
    "Translation","Rigid","ScaleShear","Affine","Deformation",
    "AffineAndDeformation")
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
      precision="float", type="AffineTransform", dim=img@dimension )
    setAntsrTransformFixedParameters( loctx, fixedParams )
    idparams = getAntsrTransformParameters( loctx )
    noisemat = rnorm( length(idparams), mean=0, sd=sdAffine )
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
    ncomp = 6  # number of compositions
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
      }
    warpTx = antsrTransformFromDisplacementField( mergeChannels( fields ) )
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
