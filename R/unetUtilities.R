#' Model loss function for multilabel problems--- multilabel dice coefficient
#'  
#' Based on the keras loss function (losses.R):
#' 
#'    \url{https://github.com/rstudio/keras/blob/master/R/losses.R}
#' 
#' @param y_true True labels (Tensor) 
#' @param y_pred Predictions (Tensor of the same shape as \code{y_true})
#' 
#' @details Loss functions are to be supplied in the loss parameter of the 
#' \code{compile()} function.
#' 
#' Loss functions can be specified either using the name of a built in loss
#' function (e.g. \code{loss = binary_crossentropy}), a reference to a built in loss
#' function (e.g. \code{loss = binary_crossentropy()}) or by passing an
#' arbitrary function that returns a scalar for each data-point.
#' The actual optimized objective is the mean of the output array across all
#' datapoints.
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
attr( multilabel_dice_coefficient, "py_function_name" ) <- 
  "multilabel_dice_coefficient"

#' Multilabel dice loss function.
#' 
#' @param y_true true encoded labels
#' @param y_pred predicted encoded labels
#'
#' @rdname loss_multilabel_dice_coefficient_error 
#' @export
loss_multilabel_dice_coefficient_error <- function( y_true, y_pred )
{
  return( -multilabel_dice_coefficient( y_true, y_pred ) )
}
attr( loss_multilabel_dice_coefficient_error, "py_function_name" ) <- 
  "multilabel_dice_coefficient_error"

#' Encoding function for Y_train
#'
#' Function for translating the segmentations to something readable by the 
#' optimization process.
#'
#' @param groundTruthSegmentations an array of shape (\code{batchSize}, \code{width}, 
#' \code{height}, \code{<depth>})
#' @param segmentationLabels vector of segmentation labels.  Note that a
#' background label (typically 0) needs to be included.
#'
#' @return an n-D array of shape 
#' \eqn{ batchSize \times width \times height \times <depth> \times numberOfSegmentationLabels }
#'
#' @author Tustison NJ
#' @export
encodeUnet <- function( groundTruthSegmentations, segmentationLabels )
  {
  numberOfLabels <- length( segmentationLabels )

  dimSegmentations <- dim( groundTruthSegmentations )

  imageDimension <- 2
  if( length( dimSegmentations ) == 4 )
    {
    imageDimension <- 3     
    }

  if( numberOfLabels < 2 )
    {
    stop( "At least two segmentation labels need to be specified." )    
    }

  yEncoded <- array( 0, dim = c( dimSegmentations, numberOfLabels ) )
  for( i in 1:numberOfLabels )
    {
    labelY <- groundTruthSegmentations
    labelY[which( groundTruthSegmentations == segmentationLabels[i] )] <- 1
    labelY[which( groundTruthSegmentations != segmentationLabels[i] )] <- 0
    if( imageDimension == 2 )
      {
      yEncoded[,,,i] <- labelY
      } else {
      yEncoded[,,,,i] <- labelY
      }
    }

  return( yEncoded ) 
  }


#' Decoding function for Y_predicted
#'
#' Function for translating the U-net predictions to ANTsR probability 
#' images.
#'
#' @param yPredicted an array of shape (\code{batchSize}, \code{width}, 
#' \code{height}, \code{<depth>}, \code{numberOfSegmentationLabels})
#'
#' @param domainImage image definining the geometry of the returned probability
#' images.
#'
#' @return a list of list of probability images.
#'
#' @author Tustison NJ
#' @importFrom utils tail
#' @importFrom ANTsRCore as.antsImage
#' @export
decodeUnet <- function( yPredicted, domainImage )
  {
  batchSize <- dim( yPredicted )[1]
  numberOfSegmentationLabels <- tail( dim( yPredicted ), 1 )

  imageDimension <- 2
  if( length( dim( yPredicted ) ) == 5 )
    {
    imageDimension <- 3    
    }

  batchProbabilityImages <- list()
  for( i in 1:batchSize )
    {
    probabilityImages <- list()    
    for( j in 1:numberOfSegmentationLabels )
      {
      if( imageDimension == 2 )
        {
        imageArray <- yPredicted[i,,,j]
        } else {
        imageArray <- yPredicted[i,,,,j]
        }
      probabilityImages[[j]] <- as.antsImage( imageArray, 
        reference = domainImage )  
      }
    batchProbabilityImages[[i]] <- probabilityImages
    }
  return( batchProbabilityImages ) 
  }




