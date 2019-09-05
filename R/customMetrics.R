#' Dice function for multilabel problems
#'
#' @param y_true True labels (Tensor)
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' @return Dice value
#' @author Tustison NJ
#'
#' @examples
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createUnetModel2D( c( 64, 64, 1 ) )
#'
#' metric_multilabel_dice_coefficient <-
#'   custom_metric( "multilabel_dice_coefficient",
#'     multilabel_dice_coefficient )
#'
#' loss_dice <- function( y_true, y_pred ) {
#'   -multilabel_dice_coefficient(y_true, y_pred)
#' }
#' attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"
#'
#' model %>% compile( loss = loss_dice,
#'   optimizer = optimizer_adam( lr = 0.0001 ),
#'   metrics = c( metric_multilabel_dice_coefficient,
#'     metric_categorical_crossentropy ) )
#'
#' @import keras
#' @export
multilabel_dice_coefficient <- function( y_true, y_pred )
{
  smoothingFactor <- 1.0

  K <- keras::backend()

  K$set_image_data_format( 'channels_last' )

  y_dims <- unlist( K$int_shape( y_pred ) )
  numberOfLabels <- y_dims[length( y_dims )]

  # Unlike native R, indexing starts at 0.  However, we are
  # assuming the background is 0 so we skip index 0.

  if( length( y_dims ) == 3 )
    {
    # 2-D image
    y_true_permuted <- K$permute_dimensions(
      y_true, pattern = c( 3L, 0L, 1L, 2L ) )
    y_pred_permuted <- K$permute_dimensions(
      y_pred, pattern = c( 3L, 0L, 1L, 2L ) )
    } else {
    # 3-D image
    y_true_permuted <- K$permute_dimensions(
      y_true, pattern = c( 4L, 0L, 1L, 2L, 3L ) )
    y_pred_permuted <- K$permute_dimensions(
      y_pred, pattern = c( 4L, 0L, 1L, 2L, 3L ) )
    }
  y_true_label <- K$gather( y_true_permuted, indices = c( 1L ) )
  y_pred_label <- K$gather( y_pred_permuted, indices = c( 1L ) )

  y_true_label_f <- K$flatten( y_true_label )
  y_pred_label_f <- K$flatten( y_pred_label )
  intersection <- y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f - intersection

  numerator <- K$sum( intersection )
  denominator <- K$sum( union )

  if( numberOfLabels > 2 )
    {
    for( j in 2L:( numberOfLabels - 1L ) )
      {
      y_true_label <- K$gather( y_true_permuted, indices = c( j ) )
      y_pred_label <- K$gather( y_pred_permuted, indices = c( j ) )
      y_true_label_f <- K$flatten( y_true_label )
      y_pred_label_f <- K$flatten( y_pred_label )
      intersection <- y_true_label_f * y_pred_label_f
      union <- y_true_label_f + y_pred_label_f - intersection

      numerator <- numerator + K$sum( intersection )
      denominator <- denominator + K$sum( union )
      }
    }
  unionOverlap <- numerator / denominator

  return ( ( 2.0 * unionOverlap + smoothingFactor ) /
    ( 1.0 + unionOverlap + smoothingFactor ) )
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
#'         \url{https://arxiv.org/abs/1512.02325}
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
