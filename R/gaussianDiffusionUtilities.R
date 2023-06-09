#' Custom Gaussian Diffusion
#'
#' Adapted from
#'
#' https://github.com/keras-team/keras-io/blob/master/examples/generative/ddpm.py
#'
#' @docType class
#'
#' @section Arguments:
#' \describe{
#'  \item{betaStart}{}
#'  \item{betaEnd}{}
#'  \item{timeSteps}{}
#'  \item{clipMin}{}
#'  \item{clipMax}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @name GaussianDiffusion
NULL

#' @export
GaussianDiffusion <- R6::R6Class("GaussianDiffusion",

  lock_objects = FALSE,

  public = list(

    initialize = function( betaStart = 1e-4, betaEnd = 0.02,
                           timeSteps = 1000, clipMin = -1.0, clipMax = 1.0 )
    {
      self$betaStart <- betaStart
      self$betaEnd <- betaEnd
      self$timeSteps <- timeSteps
      self$clipMin <- -1.0
      self$clipMax <- 1.0

      # Define the linear variance schedule
      betas <- seq( from = betaStart, to = betaEnd, length.out = timeSteps )
      self$numberOfTimeSteps <- timeSteps

      alphas <- 1.0 - betas
      self$alphasCumProd <- cumprod( alphas )
      self$alphasCumProdPrev <- c( 1.0, self$alphasCumProd[1:( length( self$alphasCumProd ) - 1 )] )

      # Calculations for diffusion q(x_t | x_{t-1}) and others
      self$sqrtAlphasCumprod <- sqrt( self$alphasCumProd )

      self$sqrtOneMinusAlphasCumProd <- sqrt( 1.0 - self$alphasCumProd )
      self$logOneMinusAlphasCumProd <- log( 1.0 - self$alphasCumProd )
      self$sqrtRecipAlphasCumProd <- sqrt( 1.0 / self$alphasCumProd )
      self$sqrtRecipM1AlphasCumProd <- sqrt( 1.0 / self$alphasCumProd - 1.0 )

      # Calculations for posterior q(x_{t-1} | x_t, x_0)
      self$posteriorVariance <- self$betas * ( 1.0 - self$alphasCumProdPrev ) / ( 1.0 - self$alphasCumProd )

      # Log calculation clipped because the posterior variance is 0 at the beginning
      # of the diffusion chain
      self$posteriorLogVarianceClipped <- log( max( self$posteriorVariance, 1e-20 ) )
      self$posteriorMeanCoef1 <- self$betas * sqrt( self$alphasCumProdPrev ) / ( 1.0 - self$alphasCumProd )
      self$posteriorMeanCoef2 <- ( 1.0 - self$alphasCumProdPrev ) * sqrt( alphas ) / ( 1.0 - self$alphasCumProd )
    },

    extract = function( a, t, xShape )
    {
      # Extract some coefficients at specified time steps, then reshape to
      # [batchSize, 1, 1, 1, 1, ...] for broadcasting purposes
      # Args:
      #     a: Tensor to extract from
      #     t: Time step for which the coefficients are to be extracted
      #     x_shape: Shape of the current batched samples

      batchSize <- xShape[1]
      out <- tensorflow::tf$gather( a, t )
      return( tensorflow::tf$reshape( out, list( batchSize, 1, 1, 1 ) ) )
    },

    qMeanVariance = function( xStart, t )
    {
      # Extracts the mean, and the variance at current time step.

      # Args:
      #     x_start: Initial sample (before the first diffusion step)
      #     t: Current time step

      xStartShape <- tensorflow::tf$shape( xStart )
      mean <- self$extract( self$sqrtAlphasCumprod, t, xStartShape ) * xStart
      variance <- self$extract( 1.0 - self$alphasCumProd, t, xStartShape )
      logVariance <- self$extract( self$logOneMinusAlphasCumProd, t, xStartShape )
      return( list( mean = mean, variance = variance, logVariance = logVariance ) )
    },

    qSample = function( xStart, t, noise )
    {
      # Diffuse the data.

      # Args:
      #     x_start: Initial sample (before the first diffusion step)
      #     t: Current time step
      #     noise: Gaussian noise to be added at the current time step
      # Returns:
      #     Diffused samples at time step `t`

      xStartShape <- tensorflow::tf$shape( xStart )
      return( self$extract( self$sqrtAlphasCumprod, t, tensorflow::tf$shape( xStart ) ) * xStart +
              self$extract( self$sqrtOneMinusAlphasCumProd, t, xStartShape ) * noise )
    },

    predictStartFromNoise = function( xT, t, noise )
    {
      xTShape <- tensorflow::tf$shape( xT )
      return( self$extract( self$sqrtRecipAlphasCumProd, t, tensorflow::tf$shape( xTShape ) ) * xT -
              self$extract( self$sqrtRecipM1AlphasCumProd, t, xTShape ) * noise )
    },

    qPosterior = function( xStart, xT, t )
    {
      # Compute the mean and variance of the diffusion
      # posterior q(x_{t-1} | x_t, x_0).

      # Args:
      #     x_start: Stating point(sample) for the posterior computation
      #     x_t: Sample at time step `t`
      #     t: Current time step
      # Returns:
      #     Posterior mean and variance at current time step

      xTShape <- tensorflow::tf$shape( xT )
      posteriorMean <- ( self$extract(self$posteriorMeanCoef1, t, xTShape ) * xStart +
                         self$extract( self$posteriorMeanCoef2, t, xTShape ) * xT )
      posteriorVariance <- self$extract( self$posteriorVariance, t, xTShape )
      posteriorLogVarianceClipped <- self$extract( self$posteriorLogVarianceClipped, t, xTShape )
      return( list( posteriorMean = posteriorMean, posteriorVariance = posteriorVariance,
                    posteriorLogVarianceClipped, posteriorLogVarianceClipped ) )
    },

    pMeanVariance = function( predNoise, x, t, clipDenoised = TRUE )
    {
      xRecon <- predictStartFromNoise( x, t = t, noise = predNoise )
      if( clipDenoised )
        {
        xRecon <- tensorflow::tf$clip_by_value( xRecon, self$clipMin, self$clipMax )
        }
      qp <- self$qPosterior( xStart = xRecon, xT = x, t = t )
      return( list( modelMean = qp$posteriorMean,
                    posteriorVariance = qp$posteriorVariance,
                    posteriorLogVariance = qp$posteriorLogVariance ) )
    },

    pSample = function( predNoise, x, t, clipDenoised = TRUE )
    {
      # Sample from the diffusion model.

      # Args:
      #     pred_noise: Noise predicted by the diffusion model
      #     x: Samples at a given time step for which the noise was predicted
      #     t: Current time step
      #     clip_denoised (bool): Whether to clip the predicted noise
      #         within the specified range or not.

      pmv <- self$pMeanVariance( predNoise, x = x, t = t, clipDenoised = clipDenoised )
      noise <- tensorflow::tf$random$normal( shape = dim( x ) )
      # No noise when t == 0
      nonzeroMask <- tensorflow::tf$reshape(
        1.0 - tensorflow$tf$cast( tensorflow$tf$equal( t, 0.0 ), tensorflow$tf$float32 ),
        list( tensorflow$shape( x )[0], 1, 1, 1 ) )
    }

  )
)

