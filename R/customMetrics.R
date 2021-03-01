#' Dice function for multilabel segmentation problems
#'
#' Note:  Assumption is that y_true is a one-hot representation
#' of the segmentation batch.  The background (label 0) should
#' be included but is not used in the calculation.
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @param dimensionality image dimension.
#' @param smoothingFactor parameter for smoothing the metric.
#' @return Dice value (negative)
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsR )
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ) )
#'
#' dice_loss <- multilabel_dice_coefficient( smoothingFactor = 0.1 )
#'
#' model %>% compile( loss = dice_loss,
#'   optimizer = optimizer_adam( lr = 0.0001 ) )
#'
#' ########################################
#' #
#' # Run in isolation
#' #
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( "r16" ) )
#' r16seg <- kmeansSegmentation( r16, 3 )$segmentation
#' r16array <- array( data = as.array( r16seg ), dim = c( 1, dim( r16seg ) ) )
#' r16tensor <- tensorflow::tf$convert_to_tensor( encodeUnet( r16array, c( 0, 1, 2, 3 ) ) )
#'
#' r64 <- antsImageRead( getANTsRData( "r64" ) )
#' r64seg <- kmeansSegmentation( r64, 3 )$segmentation
#' r64array <- array( data = as.array( r64seg ), dim = c( 1, dim( r64seg ) ) )
#' r64tensor <- tensorflow::tf$convert_to_tensor( encodeUnet( r64array, c( 0, 1, 2, 3 ) ) )
#'
#' dice_loss <- multilabel_dice_coefficient( r16tensor, r64tensor )
#' loss_value <- dice_loss( r16tensor, r64tensor )$numpy()
#'
#' # Compare with
#' # overlap_value <- labelOverlapMeasures( r16seg, r64seg )$MeanOverlap[1]
#'
#' rm(model); gc()
#' @import keras
#' @export

multilabel_dice_coefficient <- function( y_true, y_pred, smoothingFactor = 0.0 )
{
  multilabel_dice_coefficient_fixed <- function( y_true, y_pred )
    {
    K <- tensorflow::tf$keras$backend

    y_dims <- unlist( K$int_shape( y_pred ) )

    dimensionality <- 3L
    if( length( y_dims ) == 4 )
      {
      dimensionality <- 2L
      }

    numberOfLabels <- as.integer( y_dims[length( y_dims )] )

    # Unlike native R, indexing starts at 0.  However, we are
    # assuming the background is 0 so we skip index 0.

    if( dimensionality == 2L )
      {
      # 2-D image
      y_true_permuted <- K$permute_dimensions(
        y_true, pattern = c( 3L, 0L, 1L, 2L ) )
      y_pred_permuted <- K$permute_dimensions(
        y_pred, pattern = c( 3L, 0L, 1L, 2L ) )
      } else if( dimensionality == 3L ) {
      # 3-D image
      y_true_permuted <- K$permute_dimensions(
        y_true, pattern = c( 4L, 0L, 1L, 2L, 3L ) )
      y_pred_permuted <- K$permute_dimensions(
        y_pred, pattern = c( 4L, 0L, 1L, 2L, 3L ) )
      } else {
      stop( "Specified dimensionality not implemented." )
      }
    y_true_label <- K$gather( y_true_permuted, indices = c( 1L ) )
    y_pred_label <- K$gather( y_pred_permuted, indices = c( 1L ) )

    y_true_label_f <- K$flatten( y_true_label )
    y_pred_label_f <- K$flatten( y_pred_label )
    intersection <- y_true_label_f * y_pred_label_f
    union <- y_true_label_f + y_pred_label_f - intersection

    numerator <- K$sum( intersection )
    denominator <- K$sum( union )

    if( numberOfLabels > 2L )
      {
      for( i in seq.int( from = 2L, to = numberOfLabels - 1L ) )
        {
        y_true_label <- K$gather( y_true_permuted, indices = c( i ) )
        y_pred_label <- K$gather( y_pred_permuted, indices = c( i ) )
        y_true_label_f <- K$flatten( y_true_label )
        y_pred_label_f <- K$flatten( y_pred_label )
        intersection <- y_true_label_f * y_pred_label_f
        union <- y_true_label_f + y_pred_label_f - intersection

        numerator <- numerator + K$sum( intersection )
        denominator <- denominator + K$sum( union )
        }
      }
    unionOverlap <- numerator / denominator

    return ( -1.0 * ( 2.0 * unionOverlap + smoothingFactor ) /
      ( 1.0 + unionOverlap + smoothingFactor ) )
    }
  return( multilabel_dice_coefficient_fixed )
}

#' Function to calculate peak-signal-to-noise ratio.
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @return PSNR value
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ) )
#'
#' metric_peak_signal_to_noise_ratio <-
#'   custom_metric( "peak_signal_to_noise_ratio",
#'     peak_signal_to_noise_ratio )
#'
#' model %>% compile( loss = loss_categorical_crossentropy,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( metric_peak_signal_to_noise_ratio ) )
#'
#' @import keras
#' @export
peak_signal_to_noise_ratio <- function( y_true, y_pred )
{
  K <- keras::backend()

  return( -10.0 * K$log( K$mean( K$square( y_pred - y_true ) ) ) / K$log( 10.0 ) )
}

#' Function for Pearson correlation coefficient.
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @return Correlation
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ) )
#'
#' metric_pearson_correlation_coefficient <-
#'   custom_metric( "pearson_correlation_coefficient",
#'     pearson_correlation_coefficient )
#'
#' model %>% compile( loss = loss_categorical_crossentropy,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( metric_pearson_correlation_coefficient ) )
#'
#' @import keras
#' @export
pearson_correlation_coefficient <- function( y_true, y_pred )
{
  K <- keras::backend()

  N <- K$sum( K$ones_like( y_true ) )

  sum_x <- K$sum( y_true )
  sum_y <- K$sum( y_pred )
  sum_x_squared <- K$sum( K$square( y_true ) )
  sum_y_squared <- K$sum( K$square( y_pred ) )
  sum_xy <- K$sum( y_true * y_pred )

  numerator <- sum_xy - ( sum_x * sum_y / N )
  denominator <- K$sqrt( ( sum_x_squared - K$square( sum_x ) / N ) *
    ( sum_y_squared - K$square( sum_y ) / N ) )

  coefficient <- numerator / denominator

  return( coefficient )
}

#' Function for categorical focal gain
#'
#' The negative of the categorical focal loss discussed
#' in this paper:
#'
#'   \url{https://arxiv.org/pdf/1708.02002.pdf}
#'
#' and ported from this implementation:
#'
#'   \url{https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py}
#'
#' Used to handle imbalanced classes.
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @param gamma focusing parameter for modulating factor (1-p). Default = 2.0.
#' @param alpha weighing factor in balanced cross entropy. Default = 0.25.
#'
#' @return function value
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ) )
#'
#' metric_categorical_focal_gain <-
#'   custom_metric( "categorical_focal_gain",
#'     categorical_focal_gain( alpha = 0.25, gamma = 2.0 ) )
#'
#' model %>% compile( loss = categorical_focal_loss( alpha = 0.25, gamma = 2.0 ),
#'  optimizer = optimizer_adam( lr = 0.0001 ),
#'    metrics = c( metric_categorical_focal_gain ) )
#'
#' @import keras
#' @export
categorical_focal_gain <- function( y_true, y_pred, gamma = 2.0, alpha = 0.25 )
{
  categorical_focal_gain_fixed <- function( y_true, y_pred )
    {
    K <- keras::backend()

    y_pred <- y_pred / K$sum( y_pred, axis = -1L, keepdims = TRUE )
    y_pred <- K$clip( y_pred, K$epsilon(), 1.0 - K$epsilon() )
    cross_entropy = y_true * K$log( y_pred )
    gain <- alpha * K$pow( 1.0 - y_pred, gamma ) * cross_entropy
    return( K$sum( gain, axis = -1L ) )
    }

  return( categorical_focal_gain_fixed )
}

#' Function for categorical focal loss
#'
#' The categorical focal loss discussed in this paper:
#'
#'   \url{https://arxiv.org/pdf/1708.02002.pdf}
#'
#' and ported from this implementation:
#'
#'   \url{https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py}
#'
#' Used to handle imbalanced classes .
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @param gamma focusing parameter for modulating factor (1-p). Default = 2.0.
#' @param alpha weighing factor in balanced cross entropy. Default = 0.25.
#'
#' @return function value
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ) )
#'
#' metric_categorical_focal_gain <-
#'   custom_metric( "categorical_focal_gain",
#'     categorical_focal_gain( alpha = 0.25, gamma = 2.0 ) )
#'
#' model %>% compile( loss = categorical_focal_loss( alpha = 0.25, gamma = 2.0 ),
#'  optimizer = optimizer_adam( lr = 0.0001 ),
#'    metrics = c( metric_categorical_focal_gain ) )
#'
#' @import keras
#' @export
categorical_focal_loss <- function( y_true, y_pred, gamma = 2.0, alpha = 0.25 )
{
  categorical_focal_loss_fixed <- function( y_true, y_pred )
    {
    K <- keras::backend()

    y_pred <- y_pred / K$sum( y_pred, axis = -1L, keepdims = TRUE )
    y_pred <- K$clip( y_pred, K$epsilon(), 1.0 - K$epsilon() )
    cross_entropy = y_true * K$log( y_pred )
    gain <- alpha * K$pow( 1.0 - y_pred, gamma ) * cross_entropy
    return( -K$sum( gain, axis = -1L ) )
    }

  return( categorical_focal_loss_fixed )
}

#' Function for weighted categorical cross entropy
#'
#' ported from this implementation:
#'
#'    \url{https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d}
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @param weights weights for each class
#'
#' @return function value
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ), numberOfOutputs = 2 )
#'
#' model %>% compile( loss = weighted_categorical_crossentropy( weights = c( 1, 1 ) ),
#'  optimizer = optimizer_adam( lr = 0.0001 ),
#'    metrics = "accuracy" )
#'
#' @import keras
#' @export
weighted_categorical_crossentropy <- function( y_true, y_pred, weights )
{
  K <- keras::backend()

  weightsTensor <- K$variable( weights )

  weighted_categorical_crossentropy_fixed <- function( y_true, y_pred )
    {
    y_pred <- y_pred / K$sum( y_pred, axis = -1L, keepdims = TRUE )
    y_pred <- K$clip( y_pred, K$epsilon(), 1.0 - K$epsilon() )
    loss <- y_true * K$log( y_pred ) * weightsTensor
    loss <- -K$sum( loss, axis = -1L )
    return( loss )
    }
  return( weighted_categorical_crossentropy_fixed )
}

#' Function for surface loss
#'
#'  \url{https://pubmed.ncbi.nlm.nih.gov/33080507/}
#'
#' ported from this implementation:
#'
#'  \url{https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py}
#'
#' Note:  Assumption is that y_true is a one-hot representation
#' of the segmentation batch.  The background (label 0) should
#' be included but is not used in the calculation.
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#'
#' @return function value
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ), numberOfOutputs = 2 )
#'
#' model %>% compile( loss = multilabel_surface_loss,
#'  optimizer = optimizer_adam( lr = 0.0001 ),
#'    metrics = "accuracy" )
#'
#' ########################################
#' #
#' # Run in isolation
#' #
#'
#' library( ANTsR )
#'
#' r16 <- antsImageRead( getANTsRData( "r16" ) )
#' r16seg <- kmeansSegmentation( r16, 3 )$segmentation
#' r16array <- array( data = as.array( r16seg ), dim = c( 1, dim( r16seg ) ) )
#' r16tensor <- tensorflow::tf$convert_to_tensor( encodeUnet( r16array, c( 0, 1, 2, 3 ) ) )
#'
#' r64 <- antsImageRead( getANTsRData( "r64" ) )
#' r64seg <- kmeansSegmentation( r64, 3 )$segmentation
#' r64array <- array( data = as.array( r64seg ), dim = c( 1, dim( r64seg ) ) )
#' r64tensor <- tensorflow::tf$convert_to_tensor( encodeUnet( r64array, c( 0, 1, 2, 3 ) ) )
#'
#' loss_value <- multilabel_surface_loss( r16tensor, r64tensor )$numpy()
#'
#' @import keras
#' @export
multilabel_surface_loss <- function( y_true, y_pred )
{
  np <- reticulate::import( "numpy", convert = FALSE )
  scipy <- reticulate::import( "scipy" )
  tf <- tensorflow::tf
  K <- tensorflow::tf$keras$backend

  calculateResidualDistanceMap <- function( segmentation )
    {
    distance <- np$zeros_like( segmentation )

    positiveMask <- segmentation$astype( np$bool )
    if( reticulate::py_to_r( positiveMask$any() ) )
      {
      negativeMask <- np$logical_not( positiveMask )
      residualDistance <- scipy$ndimage$distance_transform_edt( negativeMask ) *
                  reticulate::py_to_r( negativeMask$astype( np$float32 ) ) -
                  ( scipy$ndimage$distance_transform_edt( positiveMask ) - 1 ) *
                  reticulate::py_to_r( positiveMask$astype( np$float32 ) )
      }
    return( residualDistance )
    }

  calculateBatchWiseResidualDistanceMaps <- function( y )
    {
    y_dims = unlist( K$int_shape( y ) )

    dimensionality <- 3L
    if( length( y_dims ) == 4 )
      {
      dimensionality <- 2L
      }

    batchSize <- as.integer( y_dims[1] )
    numberOfLabels <- as.integer( y_dims[length( y_dims )] )

    y_distance <- array( data = 0, dim = y_dims )

    for( i in seq.int( batchSize ) )
      {
      y_batch <- K$gather( y, indices = c( as.integer( i - 1 ) ) )
      for( j in seq.int( from = 2L, to = numberOfLabels ) )
        {
        if( dimensionality == 2L )
          {
          y_batch_permuted <- K$permute_dimensions(
            y_batch, pattern = c( 2L, 0L, 1L ) )
          } else if( dimensionality == 3L ) {
          y_batch_permuted <- K$permute_dimensions(
            y_batch, pattern = c( 3L, 0L, 1L, 2L ) )
          } else {
          stop( "Specified dimensionality not implemented." )
          }
        y_batch_channel <- K$gather( y_batch_permuted, indices = c( as.integer( j - 1 ) ) )
        y_batch_distance <- calculateResidualDistanceMap( reticulate::r_to_py( y_batch_channel$numpy() ) )
        if( dimensionality == 2L )
          {
          y_distance[i,,,j] <- y_batch_distance
          }
        }
      }
    return( reticulate::r_to_py( y_distance )$astype( np$float32 ) )
    }

  y_true_distance_map = tf$py_function( func = reticulate::py_func( calculateBatchWiseResidualDistanceMaps ),
                                        inp = list( y_true ),
                                        Tout = tf$float32 )
  product <- y_pred * y_true_distance_map
  return( K$mean( product ) )
}

#' Loss function for the SSD deep learning architecture.
#'
#' Creates an R6 class object for use with the SSD deep learning architecture
#' based on the paper
#'
#' W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, A. Berg.
#'     SSD: Single Shot MultiBox Detector.
#'
#' available here:
#'
#'      \url{https://arxiv.org/abs/1512.02325}
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{ssdLoss <- LossSSD$new( dimension = 2L, backgroundRatio = 3L,
#'   minNumberOfBackgroundBoxes = 0L,  alpha = 1.0,
#'   numberOfClassificationLabels )
#'
#' ssdLoss$smooth_l1_loss( y_true, y_pred )
#' ssdLoss$log_loss( y_true, y_pred )
#' ssdLoss$compute_loss( y_true, y_pred )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{ssdLoss}{A \code{process} object.}
#'  \item{dimension}{image dimensionality.}
#'  \item{backgroundRatio}{The maximum ratio of background to foreround
#'    for weighting in the loss function.  Is rounded to the nearest integer.
#'    Default is 3.}
#'  \item{minNumberOfBackgroundBoxes}{The minimum number of background boxes
#'    to use in loss computation *per batch*.  Should reflect a value in
#'    proportion to the batch size.  Default is 0.}
#'  \item{alpha}{Weighting factor for the localization loss in total loss
#'    computation.}
#'  \item{numberOfClassificationLabels}{number of classes including background.}
#' }
#'
#' @section Details:
#'   \code{$smooth_l1_loss} smooth loss
#'
#'   \code{$log_loss} log loss
#'
#'   \code{$compute_loss} computes total loss.
#'
#' @author Tustison NJ
#'
#' @return an SSD loss function
#'
#' @name LossSSD
NULL

#' @export
LossSSD <- R6::R6Class( "LossSSD",

  public = list(

    dimension = 2L,

    backgroundRatio = 3L,

    minNumberOfBackgroundBoxes = 0L,

    alpha = 1.0,

    numberOfClassificationLabels = NULL,

    tf = tensorflow::tf,

    initialize = function( dimension = 2L, backgroundRatio = 3L,
      minNumberOfBackgroundBoxes = 0L, alpha = 1.0,
      numberOfClassificationLabels = NULL )
      {
      self$dimension <- as.integer( dimension )
      self$backgroundRatio <- self$tf$constant( backgroundRatio )
      self$minNumberOfBackgroundBoxes <-
        self$tf$constant( minNumberOfBackgroundBoxes )
      self$alpha <- self$tf$constant( alpha )
      self$numberOfClassificationLabels <-
        as.integer( numberOfClassificationLabels )
      },

    smooth_l1_loss = function( y_true, y_pred )
      {
      y_true <- self$tf$cast( y_true, dtype = "float32" )
      absoluteLoss <- self$tf$abs( y_true - y_pred )
      squareLoss <- 0.5 * ( y_true - y_pred )^2
      l1Loss <- self$tf$where( self$tf$less( absoluteLoss, 1.0 ),
        squareLoss, absoluteLoss - 0.5 )
      return( self$tf$reduce_sum( l1Loss, axis = -1L, keepdims = FALSE ) )
      },

    log_loss = function( y_true, y_pred )
      {
      y_true <- self$tf$cast( y_true, dtype = "float32" )
      y_pred <- self$tf$maximum( y_pred, 1e-15 )
      logLoss <- -self$tf$reduce_sum( y_true * self$tf$log( y_pred ),
        axis = -1L, keepdims = FALSE )
      return( logLoss )
      },

    compute_loss = function( y_true, y_pred )
      {
      y_true$set_shape( y_pred$get_shape() )
      batchSize <- self$tf$shape( y_pred )[1]
      numberOfBoxesPerCell <- self$tf$shape( y_pred )[2]

      indices <- 1:self$numberOfClassificationLabels
      classificationLoss <- self$tf$to_float( self$log_loss(
         y_true[,, indices], y_pred[,, indices] ) )

      indices <- self$numberOfClassificationLabels + 1:( 2 * self$dimension )
      localizationLoss <- self$tf$to_float( self$smooth_l1_loss(
        y_true[,, indices], y_pred[,, indices] ) )

      backgroundBoxes <- y_true[,, 1]

      if( self$numberOfClassificationLabels > 2 )
        {
        foregroundBoxes <- self$tf$to_float( self$tf$reduce_max(
          y_true[,, 2:self$numberOfClassificationLabels],
          axis = -1L, keepdims = FALSE ) )
        } else {
        foregroundBoxes <- self$tf$to_float( self$tf$reduce_max(
          y_true[,, 2:self$numberOfClassificationLabels],
          axis = -1L, keepdims = TRUE ) )
        }

      numberOfForegroundBoxes <- self$tf$reduce_sum(
        foregroundBoxes, keepdims = FALSE )

      if( self$numberOfClassificationLabels > 2 )
        {
        foregroundClassLoss <- self$tf$reduce_sum(
          classificationLoss * foregroundBoxes, axis = -1L, keepdims = FALSE )
        } else {
        foregroundClassLoss <- self$tf$reduce_sum(
          classificationLoss * foregroundBoxes, axis = -1L, keepdims = TRUE )
        }

      backgroundClassLossAll <- classificationLoss * backgroundBoxes
      nonZeroIndices <-
        self$tf$count_nonzero( backgroundClassLossAll, dtype = self$tf$int32 )

      numberOfBackgroundBoxesToKeep <- self$tf$minimum( self$tf$maximum(
        self$backgroundRatio * self$tf$to_int32( numberOfForegroundBoxes ),
        self$minNumberOfBackgroundBoxes ), nonZeroIndices )

      f1 = function()
        {
        return( self$tf$zeros( list( batchSize ) ) )
        }

      f2 = function()
        {
        backgroundClassLossAll1d <-
          self$tf$reshape( backgroundClassLossAll, list( -1L ) )
        topK <- self$tf$nn$top_k(
          backgroundClassLossAll1d, numberOfBackgroundBoxesToKeep, FALSE )
        values <- topK$values
        indices <- topK$indices

        backgroundBoxesToKeep <- self$tf$scatter_nd(
          self$tf$expand_dims( indices, axis = 1L ),
          updates = self$tf$ones_like( indices, dtype = self$tf$int32 ),
          shape = self$tf$shape( backgroundClassLossAll1d ) )
        backgroundBoxesToKeep <- self$tf$to_float(
          self$tf$reshape( backgroundBoxesToKeep,
          list( batchSize, numberOfBoxesPerCell ) ) )

        return( self$tf$reduce_sum( classificationLoss * backgroundBoxesToKeep,
          axis = -1L, keepdims = FALSE ) )
        }

      backgroundClassLoss <- self$tf$cond( self$tf$equal(
        nonZeroIndices, self$tf$constant( 0L ) ), f1, f2 )

      classLoss <- foregroundClassLoss + backgroundClassLoss

      localizationLoss <- self$tf$reduce_sum(
        localizationLoss * foregroundBoxes, axis = -1L, keepdims = FALSE )

      totalLoss <- ( classLoss + self$alpha * localizationLoss ) /
        self$tf$maximum( 1.0, numberOfForegroundBoxes )

      return( totalLoss )
      }
    )
  )
