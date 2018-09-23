#'
#' Random image transformation batch generator
#'
#' This R6 class can be used to generate affine and other transformations of an
#' input image predictor and outcome population.  It currently works for single
#' predictor and single outcome modalities but will be extended in the future.
#' The class calls \code{\link{randomImageTransformAugmentation}}.
#'
#' @section Usage:
#' \preformatted{
#' bgen = randomImageTransformBatchGenerator$new( ... )
#'
#' bgen$generate( batchSize = 32L )
#'
#' }
#'
#' @section Arguments:
#' \code{imageList} List of lists where the embedded list contains k images.
#'
#' \code{outcomeImageList} List of outcome images.
#'
#' \code{transformType} random transform type to generate;
#' one of the following options
#'   \code{c("Translation","Rigid","ScaleShear","Affine","Deformation",
#'   "AffineAndDeformation")}
#'
#' \code{imageDomain} defines the spatial domain for all images.
#' NOTE: if the input images do not match the spatial domain of the domain
#' image, we internally resample the target to the domain.  This may have
#' unexpected consequences if you are not aware of this.
#' This operation will test
#' \code{antsImagePhysicalSpaceConsistency} then call
#' \code{resampleImageToTarget} upon failure.
#'
#' \code{sdAffine} roughly controls deviation from identity matrix
#'
#' \code{nControlPoints} number of control points for simulated deformation
#'
#' \code{spatialSmoothing} spatial smoothing for simulated deformation
#'
#' \code{toCategorical} boolean vector denoting whether the outcome class is categorical or not
#'
#' @section Methods:
#' \code{$new()} Initialize the class in default empty or filled form.
#'
#' \code{$generate} generate the batch of samples with given batch size
#'
#' @name randomImageTransformBatchGenerator
#' @seealso \code{\link{randomImageTransformAugmentation}}
#' @examples
#'
#' library( ANTsR )
#' i1 = antsImageRead( getANTsRData( "r16" ) )
#' i2 = antsImageRead( getANTsRData( "r64" ) )
#' s1 = thresholdImage( i1, "Otsu", 3 )
#' s2 = thresholdImage( i2, "Otsu", 3 )
#' # see ANTsRNet randomImageTransformAugmentation
#' predictors = list( list(i1), list(i2), list(i1), list(i2) )
#' outcomes = list( s1, s2,  s1, s2 )
#' trainingData <- randomImageTransformBatchGenerator$new(
#'   imageList = predictors,
#'   outcomeImageList = outcomes,
#'   transformType = "Affine",
#'   imageDomain = i1,
#'   toCategorical = TRUE
#'   )
#' testBatchGenFunction = trainingData$generate( 2 )
#' myout = testBatchGenFunction( )
#'
NULL




#' @importFrom R6 R6Class
#' @importFrom keras to_categorical
#' @export
randomImageTransformBatchGenerator <- R6::R6Class(
  "randomImageTransformBatchGenerator",

public = list(

    imageList = NULL,

    outcomeImageList = NULL,

    transformType = NULL,

    imageDomain = NULL,

    sdAffine = 1,

    nControlPoints = 100,

    spatialSmoothing = 3,

    toCategorical = FALSE,

    initialize = function( imageList = NULL, outcomeImageList = NULL,
      transformType = NULL, imageDomain = NULL, sdAffine = 1,
      nControlPoints = 100, spatialSmoothing = 3, toCategorical = FALSE )
      {

      self$sdAffine <- sdAffine
      self$nControlPoints <- nControlPoints
      self$spatialSmoothing <- spatialSmoothing
      self$toCategorical <- toCategorical

      if( !usePkg( "ANTsR" ) )
        {
        stop( "Please install the ANTsR package." )
        }

      if( !is.null( imageList ) )
        {
        self$imageList <- imageList
        } else {
        stop( "Input feature images must be specified." )
        }

      if( !is.null( outcomeImageList ) )
        {
        self$outcomeImageList <- outcomeImageList
        } else {
        stop( "Input outcome images must be specified." )
        }

      if( !is.null( transformType ) )
        {
        self$transformType <- transformType
        } else {
        stop( "Input transform type must be specified." )
        }

      if( is.null( imageDomain ) )
        {
        self$imageDomain <- imageList
        } else {
        self$imageDomain <- imageDomain
        }
      },

    generate = function( batchSize = 32L )
      {

      currentPassCount <- 1L

      function()
        {

        randITX = randomImageTransformAugmentation(
          self$imageDomain,
          self$imageList,
          self$outcomeImageList,
          n = batchSize, # generates n random samples from the inputs
          typeOfTransform = self$transformType,
          interpolator = c("linear", "nearestNeighbor"),
          sdAffine = self$sdAffine,
          nControlPoints = self$nControlPoints,
          spatialSmoothing = self$spatialSmoothing,
          composeToField = FALSE )
        gc()
        imageSize <- dim( randITX$outputPredictorList[[1]][[1]] )
        imageDim = length( imageSize )
        nChannels = length( self$imageList[[1]] ) # FIXME make work for multiple input features and multiple output features
        nChannelsY = 1
        xdims = c( batchSize, imageSize, nChannels )
        ydims = c( batchSize, imageSize )
        if ( ! self$toCategorical[ 1 ]  )
          ydims = c( batchSize, imageSize, nChannelsY )
 	      batchX <- array( data = 0, dim = xdims )
        batchY <- array( data = 0, dim = ydims )

        currentPassCount <<- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          # FIXME - make this work for multiple feature inputs
          # and multiple target outputs
          for ( chan in 1:nChannels ) {
            warpedArrayX <- as.array( randITX$outputPredictorList[[i]][[chan]] )
            warpedArrayY <- as.array( randITX$outputOutcomeList[[i]] )
            if ( imageDim == 3 ) {
              batchX[i,,,, chan] <- warpedArrayX # FIXME make work for multiple channels
            if ( ! self$toCategorical[ 1 ]  ) batchY[i,,,,1] <- warpedArrayY else batchY[i,,,] <- warpedArrayY
  	        }

            if ( imageDim == 2 ) {
              batchX[i,,,chan] <- warpedArrayX # FIXME make work for multiple channels
              if ( ! self$toCategorical[ 1 ]  ) batchY[i,,,1] <- warpedArrayY else batchY[i,,] <- warpedArrayY
              }
            }
          }

        if ( self$toCategorical[ 1 ] ) {
     	    segmentationLabels <- sort( unique( as.vector( batchY ) ) )
          outlist = list(  batchX, encodeUnet( batchY, segmentationLabels ) )
          return( outlist )
	        } else {
          return( list(  batchX, batchY ) )
	        }
        }
      }
    )
  )



#
#'
#' Random image transform parameters batch generator
#'
#' This R6 class can be used to generate parameters to affine and other
#' transformations applied to an input image population.
#' The class calls \code{\link{randomImageTransformParametersAugmentation}}.
#'
#' @section Usage:
#' \preformatted{
#' bgen = randomImageTransformParametersBatchGenerator$new( ... )
#'
#' bgen$generate( batchSize = 32L )
#'
#' }
#'
#' @section Arguments:
#' \code{imageDomain} defines the spatial domain for all images.
#' \code{imageList} List contains k images.
#'
#' \code{transformType} random transform type to generate;
#' one of the following options
#'   \code{c("Translation","Rigid","ScaleShear","Affine","DeformationBasis" ) }
#'
#' NOTE: if the input images do not match the spatial domain of the domain
#' image, we internally resample the target to the domain.  This may have
#' unexpected consequences if you are not aware of this.
#' This operation will test
#' \code{antsImagePhysicalSpaceConsistency} then call
#' \code{resampleImageToTarget} upon failure.
#'
#' \code{spatialSmoothing} spatial smoothing for simulated deformation
#'
#' \code{numberOfCompositions} number of compositions
#'
#' \code{deformationBasis} list of basis deformations
#' \code{txParamMeans} vector of basis deformations means
#' \code{txParamSDs} vector of basis deformations sds
#'

#' @section Methods:
#' \code{$new()} Initialize the class in default empty or filled form.
#'
#' \code{$generate} generate the batch of samples with given batch size
#'
#' @name randomImageTransformParametersBatchGenerator
#' @seealso \code{\link{randomImageTransformParametersAugmentation}}
#' @examples
#'
#' library( ANTsR )
#' i1 = antsImageRead( getANTsRData( "r16" ) )
#' i2 = antsImageRead( getANTsRData( "r64" ) )
#' s1 = thresholdImage( i1, "Otsu", 3 )
#' s2 = thresholdImage( i2, "Otsu", 3 )
#' # see ANTsRNet randomImageTransformAugmentation
#' predictors = list( i1, i2, i2, i1 )
#' trainingData <- randomImageTransformParametersBatchGenerator$new(
#'   imageList = predictors,
#'   transformType = "Affine",
#'   imageDomain = i1, txParamMeans=c(1,0,0,1,0,0), txParamSDs=diag(6)*0.01
#'   )
#' testBatchGenFunction = trainingData$generate( 2 )
#' myout = testBatchGenFunction( )
#'
NULL




#' @importFrom R6 R6Class
#' @export
randomImageTransformParametersBatchGenerator <- R6::R6Class(
  "randomImageTransformParametersBatchGenerator",

public = list(

    imageDomain = NULL,
    imageList = NULL,
    transformType = NULL,
    spatialSmoothing = 3,
    numberOfCompositions = 4,
    deformationBasis = NULL,
    txParamMeans = NULL,
    txParamSDs = NULL,

    initialize = function(
      imageDomain = NULL,
      imageList = NULL,
      transformType = NULL,
      spatialSmoothing = 3,
      numberOfCompositions = 4,
      deformationBasis = NULL,
      txParamMeans = NULL,
      txParamSDs = NULL
      )
      {
      self$imageDomain <- imageDomain
      self$imageList <- imageList
      self$transformType <- transformType
      self$spatialSmoothing <- spatialSmoothing
      self$numberOfCompositions <- numberOfCompositions
      self$deformationBasis <- deformationBasis
      self$txParamMeans <- txParamMeans
      self$txParamSDs <- txParamSDs

      if( !usePkg( "ANTsR" ) )
        {
        stop( "Please install the ANTsR package." )
        }

      if( !is.null( imageList ) )
        {
        self$imageList <- imageList
        } else {
        stop( "Input feature images must be specified." )
        }

      if( !is.null( transformType ) )
        {
        self$transformType <- transformType
        } else {
        stop( "Input transform type must be specified." )
        }

      if( is.null( imageDomain ) )
        {
        self$imageDomain <- imageList
        } else {
        self$imageDomain <- imageDomain
        }
      },

    generate = function( batchSize = 8L )
      {

      currentPassCount <- 1L

      function()
        {

        randITX = randomImageTransformParametersAugmentation(
          self$imageDomain,
          self$imageList,
          n = batchSize, # generates n random samples from the inputs
          typeOfTransform = self$transformType,
          interpolator = 'linear',
          spatialSmoothing = self$spatialSmoothing,
          numberOfCompositions = self$numberOfCompositions,
          deformationBasis = self$deformationBasis,
          txParamMeans = self$txParamMeans,
          txParamSDs = self$txParamSDs )
        gc()
        imageSize <- dim( randITX$outputPredictorList[[1]] )
        paramsSize = length( randITX$outputParameterList[[1]] )
        imageDim = length( imageSize )
        nChannels = 1 # FIXME make work for multiple input features and multiple output features
        xdims = c( batchSize, imageSize, nChannels )
        ydims = c( batchSize, paramsSize )
        batchX <- array( data = 0, dim = xdims )
        batchY <- array( data = 0, dim = ydims )

        currentPassCount <<- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {

          warpedArrayX <- as.array( randITX$outputPredictorList[[i]] )
#          warpedArrayX <- ( warpedArrayX - min( as.vector( warpedArrayX ) ) ) /
#            ( max( as.vector( warpedArrayX ) ) - min( as.vector( warpedArrayX ) ) )
          if ( imageDim == 3 ) {
            batchX[i,,,, 1] <- warpedArrayX # FIXME make work for multiple channels
            }

          if ( imageDim == 2 ) {
            batchX[i,,,1] <- warpedArrayX # FIXME make work for multiple channels
#            batchX[i,,,1] <- warpedArrayX - as.array( self$imageDomain )# FIXME make work for multiple channels
            }
          batchY[i,] <- randITX$outputParameterList[[i]]

          }

        return( list(  batchX, batchY ) )
        }
      }
    )
  )
