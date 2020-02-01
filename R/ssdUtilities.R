#' Convert coordinates to/from min/max representation from/to centroids/width
#'
#' Function for converting box coordinates to/from min/max representation
#' from/to centroids/width
#'
#' @param boxes A vector or 2-D array where each row corresponds to a single box
#' consisting of the format (xmin,xmax,ymin,ymax) or (centerx,centery,width,height)
#' for 2-D vs. (xmin,xmax,ymin,ymax,zmin,zmax) or
#' (centerx,centery,centerz,width,height,depth) for 3-D.
#' @param type either \verb{'minmax2centroids'} or \verb{'centroids2minmax'}
#'
#' @return a vector or 2-D array with the converted coordinates
#' @author Tustison NJ
#' @export
convertCoordinates <- function( boxes, type = 'minmax2centroids' )
{
  convertedBoxes <- boxes

  if( is.array( boxes ) )
  {
    if( length( dim( boxes ) ) == 2 )
    {
      if( type == 'centroids2minmax' )
      {
        if( ncol( boxes ) == 4 )
        {
          convertedBoxes[, 1] <- boxes[, 1] - 0.5 * boxes[, 3]
          convertedBoxes[, 2] <- boxes[, 1] + 0.5 * boxes[, 3]
          convertedBoxes[, 3] <- boxes[, 2] - 0.5 * boxes[, 4]
          convertedBoxes[, 4] <- boxes[, 2] + 0.5 * boxes[, 4]
        } else {
          convertedBoxes[, 1] <- boxes[, 1] - 0.5 * boxes[, 4]
          convertedBoxes[, 2] <- boxes[, 1] + 0.5 * boxes[, 4]
          convertedBoxes[, 3] <- boxes[, 2] - 0.5 * boxes[, 5]
          convertedBoxes[, 4] <- boxes[, 2] + 0.5 * boxes[, 5]
          convertedBoxes[, 5] <- boxes[, 3] - 0.5 * boxes[, 6]
          convertedBoxes[, 6] <- boxes[, 3] + 0.5 * boxes[, 6]
        }
      } else if( type == 'minmax2centroids' ) {
        if( ncol( boxes ) == 4 )
        {
          convertedBoxes[, 1] <- 0.5 * ( boxes[, 1] + boxes[, 2] )
          convertedBoxes[, 2] <- 0.5 * ( boxes[, 3] + boxes[, 4] )
          convertedBoxes[, 3] <- boxes[, 2] - boxes[, 1]
          convertedBoxes[, 4] <- boxes[, 4] - boxes[, 3]
        } else {
          convertedBoxes[, 1] <- 0.5 * ( boxes[, 1] + boxes[, 2] )
          convertedBoxes[, 2] <- 0.5 * ( boxes[, 3] + boxes[, 4] )
          convertedBoxes[, 3] <- 0.5 * ( boxes[, 5] + boxes[, 6] )
          convertedBoxes[, 4] <- boxes[, 2] - boxes[, 1]
          convertedBoxes[, 5] <- boxes[, 4] - boxes[, 3]
          convertedBoxes[, 6] <- boxes[, 6] - boxes[, 5]
        }
      } else {
        stop( "Unrecognized conversion type." )
      }
    } else if( length( dim( boxes ) ) == 3 ) {
      if( type == 'centroids2minmax' )
      {
        if( dim( boxes )[3] == 4 )
        {
          convertedBoxes[,, 1] <- boxes[,, 1] - 0.5 * boxes[,, 3]
          convertedBoxes[,, 2] <- boxes[,, 1] + 0.5 * boxes[,, 3]
          convertedBoxes[,, 3] <- boxes[,, 2] - 0.5 * boxes[,, 4]
          convertedBoxes[,, 4] <- boxes[,, 2] + 0.5 * boxes[,, 4]
        } else {
          convertedBoxes[,, 1] <- boxes[,, 1] - 0.5 * boxes[,, 4]
          convertedBoxes[,, 2] <- boxes[,, 1] + 0.5 * boxes[,, 4]
          convertedBoxes[,, 3] <- boxes[,, 2] - 0.5 * boxes[,, 5]
          convertedBoxes[,, 4] <- boxes[,, 2] + 0.5 * boxes[,, 5]
          convertedBoxes[,, 5] <- boxes[,, 3] - 0.5 * boxes[,, 6]
          convertedBoxes[,, 6] <- boxes[,, 3] + 0.5 * boxes[,, 6]
        }
      } else if( type == 'minmax2centroids' ) {
        if( dim( boxes )[3] == 4 )
        {
          convertedBoxes[,, 1] <- 0.5 * ( boxes[,, 1] + boxes[,, 2] )
          convertedBoxes[,, 2] <- 0.5 * ( boxes[,, 3] + boxes[,, 4] )
          convertedBoxes[,, 3] <- boxes[,, 2] - boxes[,, 1]
          convertedBoxes[,, 4] <- boxes[,, 4] - boxes[,, 3]
        } else {
          convertedBoxes[,, 1] <- 0.5 * ( boxes[,, 1] + boxes[,, 2] )
          convertedBoxes[,, 2] <- 0.5 * ( boxes[,, 3] + boxes[,, 4] )
          convertedBoxes[,, 3] <- 0.5 * ( boxes[,, 5] + boxes[,, 6] )
          convertedBoxes[,, 4] <- boxes[,, 2] - boxes[,, 1]
          convertedBoxes[,, 5] <- boxes[,, 4] - boxes[,, 3]
          convertedBoxes[,, 6] <- boxes[,, 6] - boxes[,, 5]
        }
      } else {
        stop( "Unrecognized conversion type." )
      }
    } else {
      stop( "Wrong dimensionality for input." )
    }
  } else {
    if( type == 'centroids2minmax' )
    {
      if( length( boxes ) == 4 )
      {
        convertedBoxes[1] <- boxes[1] - 0.5 * boxes[3]
        convertedBoxes[2] <- boxes[1] + 0.5 * boxes[3]
        convertedBoxes[3] <- boxes[2] - 0.5 * boxes[4]
        convertedBoxes[4] <- boxes[2] + 0.5 * boxes[4]
      } else {
        convertedBoxes[1] <- boxes[1] - 0.5 * boxes[4]
        convertedBoxes[2] <- boxes[1] + 0.5 * boxes[4]
        convertedBoxes[3] <- boxes[2] - 0.5 * boxes[5]
        convertedBoxes[4] <- boxes[2] + 0.5 * boxes[5]
        convertedBoxes[5] <- boxes[3] - 0.5 * boxes[6]
        convertedBoxes[6] <- boxes[3] + 0.5 * boxes[6]
      }
    } else if( type == 'minmax2centroids' ) {
      if( length( boxes ) == 4 )
      {
        convertedBoxes[1] <- 0.5 * ( boxes[1] + boxes[2] )
        convertedBoxes[2] <- 0.5 * ( boxes[3] + boxes[4] )
        convertedBoxes[3] <- boxes[2] - boxes[1]
        convertedBoxes[4] <- boxes[4] - boxes[3]
      } else {
        convertedBoxes[1] <- 0.5 * ( boxes[1] + boxes[2] )
        convertedBoxes[2] <- 0.5 * ( boxes[3] + boxes[4] )
        convertedBoxes[3] <- 0.5 * ( boxes[5] + boxes[6] )
        convertedBoxes[4] <- boxes[2] - boxes[1]
        convertedBoxes[5] <- boxes[4] - boxes[3]
        convertedBoxes[6] <- boxes[6] - boxes[5]
      }
    } else {
      stop( "Unrecognized conversion type." )
    }
  }
  return( convertedBoxes )
}

#' Jaccard similarity between two sets of boxes.
#'
#' Function for determinining the Jaccard or iou (intersection over union)
#' similarity measure between two sets of boxes.
#'
#' @param boxes1 A 2-D array where each row corresponds to a single box
#' consisting of the format (xmin,xmax,ymin,ymax) or
#' (xmin,xmax,ymin,ymax,zmin,zmax)
#' @param boxes2 A 2-D array where each row corresponds to a single box
#' consisting of the format (xmin,xmax,ymin,ymax) or
#' (xmin,xmax,ymin,ymax,zmin,zmax)
#'
#' @return the Jaccard simliarity
#' @author Tustison NJ

jaccardSimilarity <- function( boxes1, boxes2 )
{
  np <- reticulate::import( "numpy" )

  if( is.null( dim( boxes1 ) ) )
  {
    boxes1 <- np$expand_dims( boxes1, axis = 0L )
  }
  if( is.null( dim( boxes2 ) ) )
  {
    boxes2 <- np$expand_dims( boxes2, axis = 0L )
  }

  intersection <- np$maximum( 0, np$minimum( boxes1[, 2], boxes2[, 2] ) -
                                np$maximum( boxes1[, 1], boxes2[, 1] ) ) *
    np$maximum( 0, np$minimum( boxes1[, 4], boxes2[, 4] ) -
                  np$maximum( boxes1[, 3], boxes2[, 3] ) )
  if( ncol( boxes1 ) == 6 )
  {
    intersection <- intersection *
      np$maximum( 0, np$minimum( boxes1[, 6], boxes2[, 6] ) -
                    np$maximum( boxes1[, 5], boxes2[, 5] ) )
  }

  union1 <- ( boxes1[, 2] - boxes1[, 1] ) * ( boxes1[, 4] - boxes1[, 3] )
  if( ncol( boxes1 ) == 6 )
  {
    union1 <- union1 * ( boxes1[, 6] - boxes1[, 5] )
  }
  union2 <- ( boxes2[, 2] - boxes2[, 1] ) * ( boxes2[, 4] - boxes2[, 3] )
  if( ncol( boxes1 ) == 6 )
  {
    union2 <- union2 * ( boxes2[, 6] - boxes2[, 5] )
  }
  union <- union1 + union2 - intersection

  return( intersection / union )
}

###############################################################################
#
#            2-D SSD helper functions
#
###############################################################################

#' Plotting function for 2-D object detection visualization.
#'
#' Renders boxes on objects within rasterized images.
#'
#' @param image standard image using something like \pkg{jpeg::readJPEG}.
#' @param boxes a data frame or comprising where each row has the
#' format: xmin, xmax, ymin, ymax.
#' @param boxColors Optional scalar or vector of length = \code{numberOfBoxes}
#' used for determining the colors of the different boxes.
#' @param confidenceValues Optional vector of length = \code{numberOfBoxes} where
#' each element is in the range \verb{[0, 1]}.  Used for determining border width.
#' @param captions Optional vector of length = \code{numberOfBoxes} where
#' each element is the caption rendered with each box.
#'
#' @author Tustison NJ
#' @importFrom graphics rasterImage rect plot.new text
#' @export
drawRectangles <- function( image, boxes, boxColors = "red",
                            confidenceValues = NULL, captions = NULL )
{

  # Need to flip the y-axis due to the way rectangles are superimposed on
  # the rasterized image.  Also, we rescale the spatial domain to
  # \verb{[0, 1] x [0, 1]}, again, because of the pecularities of the plotting
  # functionality in R.

  if( is.null( dim( boxes ) ) )
  {
    scaledBoxes <- matrix( boxes, ncol = 4 )
  } else {
    scaledBoxes <- as.matrix( boxes, ncol = 4 )
  }

  scaledBoxes[, 1] <- ( scaledBoxes[, 1] - 1 ) / ( dim( image )[2] - 1 )
  scaledBoxes[, 2] <- ( scaledBoxes[, 2] - 1 ) / ( dim( image )[2] - 1 )
  scaledBoxes[, 3] <- 1 - ( scaledBoxes[, 3] - 1 ) / ( dim( image )[1] - 1 )
  scaledBoxes[, 4] <- 1 - ( scaledBoxes[, 4] - 1 ) / ( dim( image )[1] - 1 )

  numberOfBoxes <- nrow( scaledBoxes )

  if( length( boxColors ) != numberOfBoxes )
  {
    boxColors <- rep( boxColors[1], numberOfBoxes )
  }

  lineWidths <- rep( 1, numberOfBoxes )
  if( !is.null( confidenceValues ) )
  {
    if( length( confidenceValues ) != numberOfBoxes )
    {
      stop( "Number of confidenceValues doesn't match the number of boxes." )
    }
    lineWidths <- confidenceValues
  }

  plot.new()
  rasterImage( image, xleft = 0, xright = 1, ybottom = 0, ytop = 1 )
  rect( xleft = scaledBoxes[, 1], xright = scaledBoxes[, 2],
        ybottom = scaledBoxes[, 3], ytop = scaledBoxes[, 4],
        border = boxColors, lwd = lineWidths )

  if( !is.null( captions ) )
  {
    text( x = scaledBoxes[, 1], y = scaledBoxes[, 4], labels = captions,
          adj = c( -0.1, -0.3 ), col = boxColors, cex = 0.8 )
  }
}

#' Encoding function for 2-D Y_train
#'
#' Function for translating the min/max ground truth box coordinates to
#' something expected by the SSD network.  This is a SSD-specific analog
#' for keras::to_categorical().  For each image in the batch, we compare
#' the ground truth boxes for that image with all the anchor boxes.  If
#' the overlap measure exceeds a specific threshold, we write the ground
#' truth box coordinates and class to the specific position of the matched
#' anchor box.  Note that the background class will be assigned to all the
#' anchor boxes for which there was no match with any ground truth box.
#' However, an exception to this are the anchor boxes whose overlap measure
#' is higher that the specified negative threshold.
#'
#' This particular implementation was heavily influenced by the following
#' python and R implementations:
#'
#'         \url{https://github.com/pierluigiferrari/ssd_keras}
#'         \url{https://github.com/rykov8/ssd_keras}
#'         \url{https://github.com/gsimchoni/ssdkeras}
#'
#' @param groundTruthLabels A list of length `batchSize` that contains one
#' 2-D array per image.  Each 2-D array has k rows where each row corresponds
#' to a single box consisting of the format (classId,xmin,xmax,ymin,ymax).
#' Note that \verb{classId} must be greater than 0 since 0 is reserved for the
#' background label.
#' @param anchorBoxes a list of 2-D arrays where each element comprises the
#' anchor boxes for a specific aspect ratios layer.  The row of each 2-D array
#' comprises a single box specified in the form (xmin,xmax,ymin,ymax).
#' @param imageSize 2-D vector specifying the spatial domain of the input
#' images.
#' @param variances A list of 4 floats > 0 with scaling factors (actually it's
#' not factors but divisors to be precise) for the encoded predicted box
#' coordinates. A variance value of 1.0 would apply no scaling at all to the
#' predictions, while values in \verb{(0, 1)} upscale the encoded predictions and
#' values greater than 1.0 downscale the encoded predictions. These are the same
#' variances used to construct the model. Default = \code{c( 1.0, 1.0, 1.0, 1.0 )}
#' @param foregroundThreshold float between 0 and 1 determining the min threshold
#' for matching an anchor box with a ground truth box and, thus, labeling an anchor
#' box as a non-background class.  If an anchor box exceeds the ``backgroundThreshold``
#' but does not meet the foregroundThreshold for a ground truth box, then it is ignored
#' during training.  Default = 0.5.
#' @param backgroundThreshold float between 0 and 1 determining the max threshold
#' for labeling an anchor box as `background`.  If an anchor box exceeds the
#' ``backgroundThreshold`` but does not meet the foregroundThreshold for a ground
#' truth box, then it is ignored during training.  Default = 0.2.
#'
#' @return a 3-D array of shape (\code{batchSize}, \code{numberOfBoxes},
#' \code{numberOfClasses} + 4 + 4 + 4)
#'
#' where the additional 4's along the third dimension correspond to
#' the 4 predicted box coordinate offsets, the 4 coordinates for
#' the anchor boxes, and the 4 variance values.
#'
#' @author Tustison NJ
#' @export
encodeSsd2D <- function( groundTruthLabels, anchorBoxes, imageSize,
                         variances = rep( 1.0, 4 ), foregroundThreshold = 0.5,
                         backgroundThreshold = 0.2 )
{
  np <- reticulate::import( "numpy" )

  batchSize <- length( groundTruthLabels )
  classIds <- c()
  for( i in 1:batchSize )
  {
    classIds <- append( classIds, groundTruthLabels[[i]][, 1] )
  }
  classIds <- sort( unique( c( 0, classIds ) ) )
  numberOfClassificationLabels <- length( classIds )

  numberOfBoxes <- 0L
  for( i in 1:length( anchorBoxes ) )
  {
    numberOfBoxes <- numberOfBoxes + nrow( anchorBoxes[[i]] )
  }

  anchorBoxesList <- list()
  for( i in 1:length( anchorBoxes ) )
  {
    anchorBoxes[[i]] <-
      convertCoordinates( anchorBoxes[[i]], type = "minmax2centroids" )
    anchorBoxExpanded <- np$expand_dims( anchorBoxes[[i]], axis = 0L )
    anchorBoxExpanded <- np$tile( anchorBoxes[[i]], c( batchSize, 1L, 1L ) )
    anchorBoxesList[[i]] <- anchorBoxExpanded
  }
  boxesTensor <- np$concatenate( anchorBoxesList, axis = 1L )
  classesTensor <- np$zeros( reticulate::tuple(
    batchSize, numberOfBoxes, numberOfClassificationLabels ) )
  variancesTensor <- np$zeros_like( boxesTensor ) + variances

  # ``boxesTensor`` is concatenated the second time as a space filler
  yEncodedTemplate <- np$concatenate( reticulate::tuple(
    classesTensor, boxesTensor, boxesTensor, variancesTensor ), axis = 2L )
  yEncoded = np$copy( yEncodedTemplate )

  # We now fill in ``yEncoded``

  # identity matrix used for one-hot encoding
  classEye <- np$eye( numberOfClassificationLabels )

  boxIndices <- numberOfClassificationLabels + 1:4
  classIndices <- 1:( numberOfClassificationLabels + 4 )

  for( i in 1:batchSize )
  {
    availableBoxes <- np$ones( numberOfBoxes )
    backgroundBoxes <- np$ones( numberOfBoxes )

    for( j in 1:nrow( groundTruthLabels[[i]] ) )
    {
      groundTruthBox <- as.double( groundTruthLabels[[i]][j,] )

      groundTruthCoords <- as.numeric( groundTruthBox[-1] )
      groundTruthLabel <- as.integer( groundTruthBox[1] )

      similarities <- jaccardSimilarity( convertCoordinates(
        yEncodedTemplate[i,, boxIndices], type = "centroids2minmax" ),
        groundTruthCoords )

      if( abs( groundTruthCoords[2] - groundTruthCoords[1] ) < 0.001 ||
          abs( groundTruthCoords[4] - groundTruthCoords[3] ) < 0.001 )
      {
        next()
      }
      groundTruthCoords <-
        convertCoordinates( groundTruthCoords, type = 'minmax2centroids' )

      # check to see which boxes exceed the background threshold and are no
      # longer potential background boxes.  Also, clear out those background
      # boxes from the \code{imilarities} list.
      backgroundBoxes[similarities >= backgroundThreshold] <- 0
      similarities <- similarities * availableBoxes

      availableAndThreshold <- np$copy( similarities )
      availableAndThreshold[availableAndThreshold < foregroundThreshold] <- 0

      nonZeroIndices <- np$nonzero( availableAndThreshold )[[1]] + 1
      if( length( nonZeroIndices ) > 0 )
      {
        yEncoded[i, nonZeroIndices, classIndices] <- rep(
          np$concatenate( reticulate::tuple( classEye[groundTruthLabel + 1,],
                                             groundTruthCoords ), axis = 0L ), each = length( nonZeroIndices ) )
        availableBoxes[nonZeroIndices] <- 0
      } else {
        bestMatchIndex <- np$argmax( similarities ) + 1
        yEncoded[i, bestMatchIndex, classIndices] <-
          np$concatenate( reticulate::tuple( classEye[groundTruthLabel + 1,],
                                             groundTruthCoords ), axis = 0L )
        availableBoxes[bestMatchIndex] <- 0
        backgroundBoxes[bestMatchIndex] <- 0
      }
    }
    # Set the remaining background indices to the background class
    backgroundClassIndices <- np$nonzero( backgroundBoxes )[[1]] + 1
    yEncoded[i, backgroundClassIndices, 1] <- 1
  }

  # Convert absolute coordinates to offsets from anchor boxes and normalize

  indices1 <- numberOfClassificationLabels + 1:2
  indices2 <- numberOfClassificationLabels + 3:4
  indices3 <- numberOfClassificationLabels + 9:10
  indices4 <- numberOfClassificationLabels + 11:12

  yEncoded[,, indices1] <- yEncoded[,, indices1] - yEncodedTemplate[,, indices1]
  yEncoded[,, indices1] <- yEncoded[,, indices1] /
    ( yEncodedTemplate[,, indices2] * yEncodedTemplate[,, indices3] )
  yEncoded[,, indices2] <- yEncoded[,, indices2] / yEncodedTemplate[,, indices2]
  yEncoded[,, indices2] <- np$log( yEncoded[,, indices2] ) /
    yEncodedTemplate[,, indices4]

  return( yEncoded )
}

#' Decoding function for 2-D Y_train
#'
#' Function for translating the predictions from the SSD model output to
#' boxes, (centerx, centery, width, height), for subsequent usage.
#'
#' This particular implementation was heavily influenced by the following
#' python and R implementations:
#'
#'         \url{https://github.com/pierluigiferrari/ssd_keras}
#'         \url{https://github.com/rykov8/ssd_keras}
#'         \url{https://github.com/gsimchoni/ssdkeras}
#'
#' @param yPredicted The predicted output produced by the SSD model expected to
#' be an array of shape (\code{batchSize}, \code{numberOfBoxes},
#' \code{numberOfClasses} + 4 + 4 + 4)
#' where the additional 4's along the third dimension correspond to the box
#' coordinates (centerx, centery, width, height), dummy variables, and the variances.
#' \code{numberOfClasses} includes the background class.
#' @param imageSize 2-D vector specifying the spatial domain of the input
#' images.
#' @param confidenceThreshold  Float between 0 and 1.  The minimum
#' classification value required for a given box to be considered a "positive
#' prediction."  A lower value will result in better recall while a higher
#' value yields higher precision results.  Default = 0.5.
#' @param overlapThreshold  'NULL' or a float between 0 and 1.  If 'NULL' then
#' no non-maximum suppression will be performed.  Otherwise, a greedy non-
#' maximal suppression is performed following confidence thresholding.  In
#' other words all boxes with Jaccard similarities > \code{overlapThreshold} will
#' be removed from the set of predictions.   Default = 0.45.
#'
#' @return a list of length \code{batchSize} where each element comprises a 2-D
#' array where each row describes a single box using the following six elements
#' (classId, confidenceValue, xmin, xmax, ymin, ymax)
#'
#' @author Tustison NJ
#' @export
decodeSsd2D <- function( yPredicted, imageSize, confidenceThreshold = 0.5,
                         overlapThreshold = 0.45 )
{
  np <- reticulate::import( "numpy" )

  greedyNonMaximalSuppression <- function( predictions,
                                           overlapThreshold = 0.45 )
  {
    predictionsLeft <- np$copy( predictions )

    index <- 1
    maximumBoxList <- list()
    while( !is.null( dim( predictionsLeft ) )
           && dim( predictionsLeft )[1] > 0 )
    {
      maximumIndex <- np$argmax( predictionsLeft[, 2] ) + 1L
      maximumBox <- np$copy( predictionsLeft[maximumIndex,] )
      maximumBoxList[[index]] <- maximumBox
      index <- index + 1
      predictionsLeft <-
        np$delete( predictionsLeft, maximumIndex - 1L, axis = 0L )
      if( is.null( dim( predictionsLeft ) ) )
      {
        break
      }
      similarities <- jaccardSimilarity(
        predictionsLeft[, 3:6], array( maximumBox[3:6], c( 1, 4 ) ) )
      predictionsLeft <- predictionsLeft[similarities <= overlapThreshold, ]
    }
    return( do.call( rbind, maximumBoxList ) )
  }

  numberOfClassificationLabels <- dim( yPredicted )[3] - 12L
  batchSize <- dim( yPredicted )[1]

  # slice out the four normalized offset predictions plus two more for
  # later storage confidence values and class ids
  indices <- numberOfClassificationLabels + -1:4
  yPredictedConverted <- np$copy( yPredicted[,, indices, drop = FALSE] )

  # store class ID
  yPredictedConverted[,, 1] <-
    np$argmax( yPredicted[,, 1:numberOfClassificationLabels], axis = -1L )

  # store confidence values
  yPredictedConverted[,, 2] <-
    np$amax( yPredicted[,, 1:numberOfClassificationLabels], axis = -1L )

  # convert from predicted normalized anchor box offsets to absolute coordinates
  indices1 <- numberOfClassificationLabels + 11:12
  indices2 <- numberOfClassificationLabels + 7:8
  indices3 <- numberOfClassificationLabels + 9:10
  indices4 <- numberOfClassificationLabels + 5:6

  yPredictedConverted[,, 5:6] <-
    np$exp( yPredictedConverted[,, 5:6] * yPredicted[,, indices1] )
  yPredictedConverted[,, 5:6] <-
    yPredictedConverted[,, 5:6] * yPredicted[,, indices2]
  yPredictedConverted[,, 3:4] <- yPredictedConverted[,, 3:4] *
    ( yPredicted[,, indices3] * yPredicted[,, indices2] )
  yPredictedConverted[,, 3:4] <- yPredictedConverted[,, 3:4] +
    yPredicted[,, indices4]

  yPredictedConverted[,,3:6] <-
    convertCoordinates( yPredictedConverted[,,3:6], type = 'centroids2minmax' )

  yDecoded <- list()
  for( i in 1:batchSize )
  {
    ySingle <- yPredictedConverted[i,,]

    boxes <- ySingle[unlist( np$nonzero( ySingle[, 1] ) ) + 1,, drop = FALSE]
    boxes <- boxes[boxes[, 2] >= confidenceThreshold,, drop = FALSE]

    if( !is.null( overlapThreshold ) )
    {
      boxes <- greedyNonMaximalSuppression( boxes, overlapThreshold )
    }
    if( is.null( boxes ) )
    {
      yDecoded[[i]] <- matrix(, nrow = 0, ncol = 6 )
    } else {
      yDecoded[[i]] <- boxes
    }
  }
  return( yDecoded )
}

#' L2 2-D normalization layer for SSD300/512 architecture.
#'
#' L2 2-D normalization layer for SSD300/512 architecture described in
#'
#' Wei Liu, Andrew Rabinovich, and Alexander C. Berg.  ParseNet: Looking Wider
#'     to See Better.
#'
#' available here:
#'
#'         \code{https://arxiv.org/abs/1506.04579}
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{layer <- L2NormalizationLayer2D$new( scale )
#'
#' layer$call( x, mask = NULL )
#' layer$build( input_shape )
#' layer$compute_output_shape( input_shape )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{layer}{A \code{process} object.}
#'  \item{scale}{feature scale.  Default = 20}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$build}
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return output tensor with the same shape as the input.
#'
#' @name L2NormalizationLayer2D
NULL

#' @export
L2NormalizationLayer2D <- R6::R6Class( "L2NormalizationLayer2D",

inherit = KerasLayer,

public = list(

  scale = NULL,

  channelAxis = NULL,

  gamma = NULL,

  initialize = function( scale = 20 )
  {
    K <- keras::backend()

    if( K$image_data_format() == "channels_last" )
    {
      self$channelAxis <- 4
    } else {
      self$channelAxis <- 2
    }
    self$scale <- scale
  },

  build = function( input_shape )
  {
    self$gamma <- self$add_weight(
      name = paste0( 'gamma_', self$name ),
      shape = list( input_shape[[self$channelAxis]] ),
      initializer = initializer_constant( value = self$scale ),
      trainable = TRUE )
  },

  call = function( x, mask = NULL )
  {
    K <- keras::backend()
    output <- K$l2_normalize( x, self$channelAxis )
    output <- output * self$gamma
    return( output )
  },

  compute_output_shape = function( input_shape )
  {
    return( reticulate::tuple( input_shape ) )
  }
)
)

#' Normalization layer (2-D and 3-D)
#'
#' Wraps a custom layer for the SSD network
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param scale box scale
#' @param name The name of the layer
#' @param trainable Whether the layer weights will be updated during training.
#'
#' @return a keras layer tensor
#' @export
#' @rdname layer_l2_normalization_2d
layer_l2_normalization_2d <- function( object, scale = 20, name = NULL,
                                       trainable = TRUE ) {
  create_layer( L2NormalizationLayer2D, object,
                list( scale = scale, name = name, trainable = TRUE ) )
}

#' @export
#' @rdname layer_l2_normalization_2d
layer_l2_normalization_3d <- function( object, scale = 20, name = NULL,
                                       trainable = TRUE ) {
  create_layer( L2NormalizationLayer3D, object,
                list( scale = scale, name = name, trainable = TRUE ) )
}

#' Anchor box layer for SSD architecture (2-D).
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{anchorBoxGenerator <- AnchorBoxLayer2D$new( imageSize,
#'      scale, nextScale, aspectRatios = c( '1:1', '2:1', '1:2' ),
#'      variances = 1.0 )
#'
#' anchorBoxGenerator$call( x, mask = NULL )
#' anchorBoxGenerator$compute_output_shape( input_shape )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{anchorBoxGenerator}{A \code{process} object.}
#'  \item{imageSize}{size of the input image.}
#'  \item{scale}{scale of each box (in pixels).}
#'  \item{nextScale}{next scale of each box (in pixels).}
#'  \item{aspectRatios}{vector describing the geometries of the anchor boxes
#'    for this layer.}
#'  \item{variances}{a list of 4 floats > 0 with scaling factors for the encoded
#'    predicted box coordinates. A variance value of 1.0 would apply no scaling at
#'    all to the predictions, while values in (0,1) upscale the encoded
#'    predictions and values greater than 1.0 downscale the encoded predictions.
#'    Defaults to 1.0.}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
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
#' @return a 5-D tensor with shape
#' \eqn{ batchSize \times widthSize \times heightSize \times numberOfBoxes \times 8 }
#' In the last dimension, the first 4 values correspond to the
#' 2-D coordinates of the bounding boxes and the other 4 are the variances.
#'
#' @name AnchorBoxLayer2D
NULL

#' @export
AnchorBoxLayer2D <- R6::R6Class( "AnchorBoxLayer2D",

inherit = KerasLayer,

public = list(

  imageSize = NULL,

  scale = NULL,

  nextScale = NULL,

  aspectRatios = NULL,

  variances = NULL,

  imageSizeAxes = NULL,

  channelAxis = NULL,

  numberOfBoxes = NULL,

  anchorBoxesArray = NULL,

  initialize = function( imageSize, scale, nextScale,
                         aspectRatios = c( '1:1', '2:1', '1:2' ), variances = 1.0 )
  {
    K <- keras::backend()

    if( K$image_data_format() == "channels_last" )
    {
      self$imageSizeAxes[1] <- 2
      self$imageSizeAxes[2] <- 3
      self$channelAxis <- 4
    } else {
      self$imageSizeAxes[1] <- 3
      self$imageSizeAxes[2] <- 4
      self$channelAxis <- 2
    }
    self$scale <- scale
    self$nextScale <- nextScale

    self$imageSize <- imageSize

    if( is.null( aspectRatios ) )
    {
      self$aspectRatios <- c( '1:1' )
    } else {
      self$aspectRatios <- aspectRatios
    }

    if( length( variances ) == 1 )
    {
      self$variances <- rep( variances, 4 )
    } else if( length( variances ) == 4 ) {
      self$variances <- variances
    } else {
      stop( "Error: Length of variances must be 1 or 4." )
    }
  },

  call = function( x, mask = NULL )
  {
    K <- keras::backend()

    np <- reticulate::import( "numpy" )

    input_shape <- K$int_shape( x )
    layerSize <- c()
    layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
    layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]

    minImageSize <- min( self$imageSize )

    widths <- c()
    heights <- c()
    count <- 1L

    for( i in 1:length( self$aspectRatios ) )
    {
      aspectRatioValues <- as.numeric(
        unlist( strsplit( self$aspectRatios[i], ':' ) ) )
      if( length( aspectRatioValues ) == 1 )
      {
        aspectRatioValues <- rep( aspectRatioValues[1], 2 )
      } else if( length( aspectRatioValues ) != 2 ) {
        stop( "Incorrect aspect ratio specification." )
      }
      aspectRatio <- max( aspectRatioValues ) / min( aspectRatioValues )
      if( aspectRatio == 1 )
      {
        size <- self$scale * minImageSize
        widths[count] <- size
        heights[count] <- size
        count <- count + 1L

        size <- sqrt( self$scale * self$nextScale ) * minImageSize
        widths[count] <- size
        heights[count] <- size
        count <- count + 1L
      } else {
        scaleFactor <- self$scale * minImageSize * sqrt( aspectRatio ) /
          max( aspectRatioValues )
        widths[count] <- scaleFactor * aspectRatioValues[1]
        heights[count] <- scaleFactor * aspectRatioValues[2]
        count <- count + 1L
      }
    }
    self$numberOfBoxes <- count - 1L

    boxDimensions <- list()
    boxDimensions[[1]] <- widths
    boxDimensions[[2]] <- heights

    cellSize <- self$imageSize / layerSize
    centers <- list()
    for( i in 1:length( cellSize ) )
    {
      centers[[i]] <- seq( 0.5 * cellSize[i],
                           self$imageSize[i] - 0.5 * cellSize[i], length.out = layerSize[i] )
    }

    boxesTensor <- np$zeros( reticulate::tuple(
      layerSize[1], layerSize[2], self$numberOfBoxes, 4L ) )

    grid <- np$meshgrid( centers[[1]], centers[[2]] )
    for( i in 1:length( grid ) )
    {
      boxesTensor[,,, i] <- np$tile( np$expand_dims( grid[[i]], axis = -1L ),
                                     reticulate::tuple( 1L, 1L, self$numberOfBoxes ) )
      boxesTensor[,,, i + length( grid )] <- array( rep( boxDimensions[[i]],
                                                         each = layerSize[1] * layerSize[2] ),
                                                    dim = c( layerSize[1], layerSize[2], self$numberOfBoxes ) )
    }

    self$anchorBoxesArray <- reticulate::array_reshape( boxesTensor,
                                                        dim = c( layerSize[1] * layerSize[2] * self$numberOfBoxes, 4 ) )

    # Convert to (xmin, xmax, ymin, ymax)
    self$anchorBoxesArray <- convertCoordinates( self$anchorBoxesArray,
                                                 type = 'centroids2minmax' )

    variancesTensor <- np$zeros_like( boxesTensor )
    variancesTensor <- variancesTensor + self$variances

    anchorBoxesTensor = np$concatenate(
      reticulate::tuple( boxesTensor, variancesTensor ), axis = -1L )
    anchorBoxesTensor <- np$expand_dims( anchorBoxesTensor, axis = 0L )

    anchorBoxesTensor <- K$constant( anchorBoxesTensor, dtype = 'float32' )
    anchorBoxesTensor <- K$tile( anchorBoxesTensor,
                                 c( K$shape( x )[1], 1L, 1L, 1L, 1L ) )

    return( anchorBoxesTensor )
  },

  compute_output_shape = function( input_shape )
  {
    layerSize <- c()
    layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
    layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]

    return( reticulate::tuple( input_shape[[1]], layerSize[1],
                               layerSize[2], self$numberOfBoxes, 8L ) )
  }
)
)

#' Anchor box layer (2-D and 3-D)
#'
#' Wraps a custom layer for the SSD network
#'
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param imageSize size of the image, passed to \code{\link{create_layer}}
#' @param scale box scale, passed to \code{\link{create_layer}}
#' @param nextScale box scale, passed to \code{\link{create_layer}}
#' @param aspectRatios list of ratios used for the boxes,
#'  passed to \code{\link{create_layer}}
#' @param variances list of variances, passed to \code{\link{create_layer}}
#' @param name The name of the layer
#' @param trainable logical indicating if it is trainable or not
#'
#' @return a keras layer tensor
#' @rdname layer_anchor_box_2d
#' @export
layer_anchor_box_2d <- function(
  object, imageSize, scale, nextScale,
  aspectRatios, variances, name = NULL, trainable = TRUE ) {
  create_layer( AnchorBoxLayer2D, object,
                list( imageSize = imageSize, scale = scale, nextScale = nextScale,
                      aspectRatios = aspectRatios, variances = variances, name = name,
                      trainable = trainable )
  )
}

#' @rdname layer_anchor_box_2d
#' @export
layer_anchor_box_3d <- function( object, imageSize, scale, nextScale,
                                 aspectRatios, variances, name = NULL, trainable = TRUE ) {
  create_layer( AnchorBoxLayer3D, object,
                list( imageSize = imageSize, scale = scale, nextScale = nextScale,
                      aspectRatios = aspectRatios, variances = variances, name = name,
                      trainable = trainable )
  )
}


###############################################################################
#
#            3-D SSD helper functions
#
###############################################################################

#' Encoding function for 3-D Y_train
#'
#' Function for translating the min/max ground truth box coordinates to
#' something expected by the SSD network.  This is a SSD-specific analog
#' for \pkg{keras::to_categorical()}.  For each image in the batch, we compare
#' the ground truth boxes for that image with all the anchor boxes.  If
#' the overlap measure exceeds a specific threshold, we write the ground
#' truth box coordinates and class to the specific position of the matched
#' anchor box.  Note that the background class will be assigned to all the
#' anchor boxes for which there was no match with any ground truth box.
#' However, an exception to this are the anchor boxes whose overlap measure
#' is higher that the specified negative threshold.
#'
#' This particular implementation was heavily influenced by the following
#' python and R implementations:
#'
#'         \url{https://github.com/pierluigiferrari/ssd_keras}
#'         \url{https://github.com/rykov8/ssd_keras}
#'         \url{https://github.com/gsimchoni/ssdkeras}
#'
#' @param groundTruthLabels A list of length `batchSize` that contains one
#' 2-D array per image.  Each 2-D array has k rows where each row corresponds
#' to a single box consisting of the format (classId,xmin,xmax,ymin,ymax,zmin,zmax).
#' Note that `classId` must be greater than 0 since 0 is reserved for the
#' background label.
#' @param anchorBoxes a list of 2-D arrays where each element comprises the
#' anchor boxes for a specific aspect ratios layer.  The row of each 2-D array
#' comprises a single box specified in the for (xmin,xmax,ymin,ymax,zmin,zmax).
#' @param imageSize 3-D vector specifying the spatial domain of the input
#' images.
#' @param variances A list of 6 floats > 0 with scaling factors (actually it's
#' not factors but divisors to be precise) for the encoded predicted box
#' coordinates. A variance value of 1.0 would apply no scaling at all to the
#' predictions, while values in (0,1) upscale the encoded predictions and
#' values greater than 1.0 downscale the encoded predictions. These are the same
#' variances used to construct the model. Default =
#' \code{c( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )}
#' @param foregroundThreshold float between 0 and 1 determining the min threshold
#' for matching an anchor box with a ground truth box and, thus, labeling an anchor
#' box as a non-background class.  If an anchor box exceeds the ``backgroundThreshold``
#' but does not meet the foregroundThreshold for a ground truth box, then it is ignored
#' during training.  Default = 0.5.
#' @param backgroundThreshold float between 0 and 1 determining the max threshold
#' for labeling an anchor box as `background`.  If an anchor box exceeds the
#' \code{backgroundThreshold} but does not meet the foregroundThreshold for a ground
#' truth box, then it is ignored during training.  Default = 0.2.
#'
#' @return a 3-D array of shape (\code{batchSize}, \code{numberOfBoxes},
#' \code{numberOfClasses} + 6 + 6 + 6)
#' where the additional 6's along the third dimension correspond to
#' the 6 predicted box coordinate offsets, the 6 coordinates for
#' the anchor boxes, and the 6 variance values.
#'
#' @author Tustison NJ
#' @export
encodeSsd3D <- function( groundTruthLabels, anchorBoxes, imageSize,
                         variances = rep( 1.0, 6 ), foregroundThreshold = 0.5,
                         backgroundThreshold = 0.2 )
{
  np <- reticulate::import( "numpy" )

  batchSize <- length( groundTruthLabels )
  classIds <- c()
  for( i in 1:batchSize )
  {
    classIds <- append( classIds, groundTruthLabels[[i]][, 1] )
  }
  classIds <- sort( unique( c( 0, classIds ) ) )
  numberOfClassificationLabels <- length( classIds )

  numberOfBoxes <- 0L
  for( i in 1:length( anchorBoxes ) )
  {
    numberOfBoxes <- numberOfBoxes + nrow( anchorBoxes[[i]] )
  }

  anchorBoxesList <- list()
  for( i in 1:length( anchorBoxes ) )
  {
    anchorBoxes[[i]] <-
      convertCoordinates( anchorBoxes[[i]], type = "minmax2centroids" )
    anchorBoxExpanded <- np$expand_dims( anchorBoxes[[i]], axis = 0L )
    anchorBoxExpanded <- np$tile( anchorBoxes[[i]], c( batchSize, 1L, 1L ) )
    anchorBoxesList[[i]] <- anchorBoxExpanded
  }
  boxesTensor <- np$concatenate( anchorBoxesList, axis = 1L )
  classesTensor <- np$zeros( reticulate::tuple(
    batchSize, numberOfBoxes, numberOfClassificationLabels ) )
  variancesTensor <- np$zeros_like( boxesTensor ) + variances

  # ``boxesTensor`` is concatenated the second time as a space filler
  yEncodedTemplate <- np$concatenate( reticulate::tuple(
    classesTensor, boxesTensor, boxesTensor, variancesTensor ), axis = 2L )
  yEncoded = np$copy( yEncodedTemplate )

  # We now fill in ``yEncoded``

  # identity matrix used for one-hot encoding
  classEye <- np$eye( numberOfClassificationLabels )

  boxIndices <- numberOfClassificationLabels + 1:6
  classIndices <- 1:( numberOfClassificationLabels + 6 )

  for( i in 1:batchSize )
  {
    availableBoxes <- np$ones( numberOfBoxes )
    backgroundBoxes <- np$ones( numberOfBoxes )

    for( j in 1:nrow( groundTruthLabels[[i]] ) )
    {
      groundTruthBox <- as.double( groundTruthLabels[[i]][j,] )

      groundTruthCoords <- as.numeric( groundTruthBox[-1] )
      groundTruthLabel <- as.integer( groundTruthBox[1] )

      similarities <- jaccardSimilarity( convertCoordinates(
        yEncodedTemplate[i,, boxIndices], type = "centroids2minmax" ),
        groundTruthCoords )

      if( abs( groundTruthCoords[2] - groundTruthCoords[1] ) < 0.001 ||
          abs( groundTruthCoords[4] - groundTruthCoords[3] ) < 0.001 ||
          abs( groundTruthCoords[6] - groundTruthCoords[5] ) < 0.001 )
      {
        next()
      }
      groundTruthCoords <-
        convertCoordinates( groundTruthCoords, type = 'minmax2centroids' )

      # check to see which boxes exceed the background threshold and are no
      # longer potential background boxes.  Also, clear out those background
      # boxes from the similarities list.
      backgroundBoxes[similarities >= backgroundThreshold] <- 0
      similarities <- similarities * availableBoxes

      availableAndThreshold <- np$copy( similarities )
      availableAndThreshold[availableAndThreshold < foregroundThreshold] <- 0

      nonZeroIndices <- np$nonzero( availableAndThreshold )[[1]] + 1
      if( length( nonZeroIndices ) > 0 )
      {
        yEncoded[i, nonZeroIndices, classIndices] <- rep(
          np$concatenate( reticulate::tuple( classEye[groundTruthLabel + 1,],
                                             groundTruthCoords ), axis = 0L ), each = length( nonZeroIndices ) )
        availableBoxes[nonZeroIndices] <- 0
      } else {
        bestMatchIndex <- np$argmax( similarities ) + 1
        yEncoded[i, bestMatchIndex, classIndices] <-
          np$concatenate( reticulate::tuple( classEye[groundTruthLabel + 1,],
                                             groundTruthCoords ), axis = 0L )
        availableBoxes[bestMatchIndex] <- 0
        backgroundBoxes[bestMatchIndex] <- 0
      }
    }
    # Set the remaining background indices to the background class
    backgroundClassIndices <- np$nonzero( backgroundBoxes )[[1]] + 1
    yEncoded[i, backgroundClassIndices, 1] <- 1
  }

  # Convert absolute coordinates to offsets from anchor boxes and normalize

  indices1 <- numberOfClassificationLabels + 1:3
  indices2 <- numberOfClassificationLabels + 4:6
  indices3 <- numberOfClassificationLabels + 13:15
  indices4 <- numberOfClassificationLabels + 16:18

  yEncoded[,, indices1] <- yEncoded[,, indices1] - yEncodedTemplate[,, indices1]
  yEncoded[,, indices1] <- yEncoded[,, indices1] /
    ( yEncodedTemplate[,, indices2] * yEncodedTemplate[,, indices3] )
  yEncoded[,, indices2] <- yEncoded[,, indices2] / yEncodedTemplate[,, indices2]
  yEncoded[,, indices2] <- np$log( yEncoded[,, indices2] ) /
    yEncodedTemplate[,, indices4]

  return( yEncoded )
}

#' Decoding function for 3-D Y_train
#'
#' Function for translating the predictions from the SSD model output to
#' boxes, (centerx, centery, width, height), for subsequent usage.
#'
#' This particular implementation was heavily influenced by the following
#' python and R implementations:
#'
#'         \url{https://github.com/pierluigiferrari/ssd_keras}
#'         \url{https://github.com/rykov8/ssd_keras}
#'         \url{https://github.com/gsimchoni/ssdkeras}
#'
#' @param yPredicted The predicted output produced by the SSD model expected to
#' be an array of shape (\code{batchSize}, \code{numberOfBoxes},
#' \code{numberOfClasses} + 6 + 6 + 6)
#' where the additional 6's along the third dimension correspond to the box
#' coordinates (centerx, centery, width, height), dummy variables, and the variances.
#' \code{numberOfClasses} includes the background class.
#' @param imageSize 3-D vector specifying the spatial domain of the input
#' images.
#' @param confidenceThreshold  Float between 0 and 1.  The minimum
#' classification value required for a given box to be considered a "positive
#' prediction."  A lower value will result in better recall while a higher
#' value yields higher precision results.  Default = 0.5.
#' @param overlapThreshold  \code{NULL} or a float between 0 and 1.  If
#' \code{NULL} then no non-maximum suppression will be performed.  Otherwise, a
#' greedy non-maximal suppression is performed following confidence thresholding.
#' In other words all boxes with Jaccard similarities > \code{overlapThreshold}
#' will be removed from the set of predictions.   Default = 0.45.
#'
#' @return a list of length \code{batchSize} where each element comprises a 2-D
#' array where each row describes a single box using the following six elements
#' (classId, confidenceValue, xmin, xmax, ymin, ymax, zmin, zmax).
#'
#' @author Tustison NJ
#' @export
decodeSsd3D <- function( yPredicted, imageSize, confidenceThreshold = 0.5,
                         overlapThreshold = 0.45 )
{
  np <- reticulate::import( "numpy" )

  greedyNonMaximalSuppression <- function( predictions,
                                           overlapThreshold = 0.45 )
  {
    predictionsLeft <- np$copy( predictions )

    index <- 1
    maximumBoxList <- list()
    while( !is.null( dim( predictionsLeft ) )
           && dim( predictionsLeft )[1] > 0 )
    {
      maximumIndex <- np$argmax( predictionsLeft[, 2] ) + 1L
      maximumBox <- np$copy( predictionsLeft[maximumIndex,] )
      maximumBoxList[[index]] <- maximumBox
      index <- index + 1
      predictionsLeft <-
        np$delete( predictionsLeft, maximumIndex - 1L, axis = 0L )
      if( is.null( dim( predictionsLeft ) ) )
      {
        break
      }
      similarities <- jaccardSimilarity(
        predictionsLeft[, 3:8], array( maximumBox[3:8], c( 1, 6 ) ) )
      predictionsLeft <- predictionsLeft[similarities <= overlapThreshold, ]
    }
    return( do.call( rbind, maximumBoxList ) )
  }

  numberOfClassificationLabels <- dim( yPredicted )[3] - 18L
  batchSize <- dim( yPredicted )[1]

  # slice out the four normalized offset predictions plus two more for
  # later storage confidence values and class ids
  indices <- numberOfClassificationLabels + -1:4
  yPredictedConverted <- np$copy( yPredicted[,, indices, drop = FALSE] )

  # store class ID
  yPredictedConverted[,, 1] <-
    np$argmax( yPredicted[,, 1:numberOfClassificationLabels], axis = -1L )

  # store confidence values
  yPredictedConverted[,, 2] <-
    np$amax( yPredicted[,, 1:numberOfClassificationLabels], axis = -1L )

  # convert from predicted normalized anchor box offsets to absolute coordinates

  indices1 <- numberOfClassificationLabels + 16:18
  indices2 <- numberOfClassificationLabels + 7:8
  indices3 <- numberOfClassificationLabels + 13:15
  indices4 <- numberOfClassificationLabels + 6:8

  yPredictedConverted[,, 6:8] <-
    np$exp( yPredictedConverted[,, 6:8] * yPredicted[,, indices1] )
  yPredictedConverted[,, 6:8] <-
    yPredictedConverted[,, 6:8] * yPredicted[,, indices2]
  yPredictedConverted[,, 3:5] <- yPredictedConverted[,, 3:5] *
    ( yPredicted[,, indices3] * yPredicted[,, indices2] )
  yPredictedConverted[,, 3:5] <- yPredictedConverted[,, 3:5] +
    yPredicted[,, indices4]

  yPredictedConverted[,,3:8] <-
    convertCoordinates( yPredictedConverted[,,3:8], type = 'centroids2minmax' )

  yDecoded <- list()
  for( i in 1:batchSize )
  {
    ySingle <- yPredictedConverted[i,,]

    boxes <- ySingle[unlist( np$nonzero( ySingle[, 1] ) ) + 1,, drop = FALSE]
    boxes <- boxes[boxes[, 2] >= confidenceThreshold,, drop = FALSE]

    if( !is.null( overlapThreshold ) )
    {
      boxes <- greedyNonMaximalSuppression( boxes, overlapThreshold )
    }
    if( is.null( boxes ) )
    {
      yDecoded[[i]] <- matrix(, nrow = 0, ncol = 6 )
    } else {
      yDecoded[[i]] <- boxes
    }
  }
  return( yDecoded )
}

#' L2 3-D normalization layer for SSD300/512 architecture.
#'
#' L2 3-D normalization layer for SSD300/512 architecture described in
#'
#' Wei Liu, Andrew Rabinovich, and Alexander C. Berg.  ParseNet: Looking Wider
#'     to See Better.
#'
#' available here:
#'
#'         \code{https://arxiv.org/abs/1506.04579}
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{layer <- L2NormalizationLayer3D$new( scale )
#'
#' layer$call( x, mask = NULL )
#' layer$build( input_shape )
#' layer$compute_output_shape( input_shape )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{layer}{A \code{process} object.}
#'  \item{scale}{feature scale.  Default = 20}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
#' }
#'
#' @section Details:
#'   \code{$initialize} instantiates a new class.
#'
#'   \code{$build}
#'
#'   \code{$call} main body.
#'
#'   \code{$compute_output_shape} computes the output shape.
#'
#' @author Tustison NJ
#'
#' @return output tensor with the same shape as the input.
#'
#' @name L2NormalizationLayer3D
NULL

#' @export
L2NormalizationLayer3D <- R6::R6Class( "L2NormalizationLayer3D",

inherit = KerasLayer,

public = list(

  scale = NULL,

  channelAxis = NULL,

  gamma = NULL,

  initialize = function( scale = 20 )
  {
    K <- keras::backend()

    if( K$image_data_format() == "channels_last" )
    {
      self$channelAxis <- 5
    } else {
      self$channelAxis <- 2
    }
    self$scale <- scale
  },

  build = function( input_shape )
  {
    self$gamma <- self$add_weight(
      name = paste0( 'gamma_', self$name ),
      shape = list( input_shape[[self$channelAxis]] ),
      initializer = initializer_constant( value = self$scale ),
      trainable = TRUE )
  },

  call = function( x, mask = NULL )
  {
    K <- keras::backend()

    output <- K$l2_normalize( x, self$channelAxis )
    output <- output * self$gamma
    return( output )
  },

  compute_output_shape = function( input_shape )
  {
    return( reticulate::tuple( input_shape ) )
  }
)
)


#' Anchor box layer for SSD architecture (3-D).
#'
#' @docType class
#'
#' @section Usage:
#' \preformatted{anchorBoxGenerator <- AnchorBoxLayer3D$new( imageSize,
#'      scale, nextScale, aspectRatios = c( '1:1:1', '2:1:1', '1:2:1', '1:1:2' ),
#'      variances = 1.0 )
#'
#' anchorBoxGenerator$call( x, mask = NULL )
#' anchorBoxGenerator$compute_output_shape( input_shape )
#' }
#'
#' @section Arguments:
#' \describe{
#'  \item{anchorBoxGenerator}{A \code{process} object.}
#'  \item{imageSize}{size of the input image.}
#'  \item{scale}{scale of each box (in pixels).}
#'  \item{nextScale}{next scale of each box (in pixels).}
#'  \item{aspectRatios}{vector describing the geometries of the anchor boxes
#'    for this layer.}
#'  \item{variances}{a list of 6 floats > 0 with scaling factors for the encoded
#'    predicted box coordinates. A variance value of 1.0 would apply no scaling at
#'    all to the predictions, while values in (0,1) upscale the encoded
#'    predictions and values greater than 1.0 downscale the encoded predictions.
#'    Defaults to 1.0.}
#'  \item{x}{}
#'  \item{mask}{}
#'  \item{input_shape}{}
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
#' @return a 6-D tensor with shape
#' \eqn{ batchSize \times widthSize \times heightSize \times depthSize \times numberOfBoxes \times 12 }
#' In the last dimension, the first 6 values correspond to the
#' 3-D coordinates of the bounding boxes and the other 6 are the variances.
#'
#' @name AnchorBoxLayer3D
NULL

#' @export
AnchorBoxLayer3D <- R6::R6Class( "AnchorBoxLayer3D",

inherit = KerasLayer,

public = list(

  imageSize = NULL,

  scale = NULL,

  nextScale = NULL,

  aspectRatios = NULL,

  variances = NULL,

  imageSizeAxes = NULL,

  channelAxis = NULL,

  numberOfBoxes = NULL,

  anchorBoxesArray = NULL,

  initialize = function( imageSize, scale, nextScale,
                         aspectRatios = c( '1:1:1', '2:1:1', '1:2:1', '1:1:2' ), variances = 1.0 )
  {

    K <- keras::backend()

    if( K$image_data_format() == "channels_last" )
    {
      self$imageSizeAxes[1] <- 2
      self$imageSizeAxes[2] <- 3
      self$imageSizeAxes[3] <- 4
      self$channelAxis <- 5
    } else {
      self$imageSizeAxes[1] <- 3
      self$imageSizeAxes[2] <- 4
      self$imageSizeAxes[3] <- 5
      self$channelAxis <- 2
    }
    self$scale <- scale
    self$nextScale <- nextScale

    self$imageSize <- imageSize

    if( is.null( aspectRatios ) )
    {
      self$aspectRatios <- c( '1:1:1' )
    } else {
      self$aspectRatios <- aspectRatios
    }

    if( length( variances ) == 1 )
    {
      self$variances <- rep( variances, 6 )
    } else if( length( variances ) == 6 ) {
      self$variances <- variances
    } else {
      stop( "Error: Length of variances must be 1 or 6." )
    }
  },

  call = function( x, mask = NULL )
  {
    K <- keras::backend()

    np <- reticulate::import( "numpy" )

    input_shape <- K$int_shape( x )
    layerSize <- c()
    layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
    layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]
    layerSize[3] <- input_shape[[self$imageSizeAxes[3]]]

    minImageSize <- min( self$imageSize )

    widths <- c()
    heights <- c()
    depths <- c()
    count <- 1L

    for( i in 1:length( self$aspectRatios ) )
    {
      aspectRatioValues <- as.numeric(
        unlist( strsplit( self$aspectRatios[i], ':' ) ) )
      if( length( aspectRatioValues ) == 1 )
      {
        aspectRatioValues <- rep( aspectRatioValues[1], 3 )
      } else if( length( aspectRatioValues ) != 3 ) {
        stop( "Incorrect aspect ratio specification." )
      }
      aspectRatio <- max( aspectRatioValues ) / min( aspectRatioValues )
      if( aspectRatio == 1 )
      {
        size <- self$scale * minImageSize
        widths[count] <- size
        heights[count] <- size
        depths[count] <- size
        count <- count + 1L

        size <- sqrt( self$scale * self$nextScale ) * minImageSize
        widths[count] <- size
        heights[count] <- size
        depths[count] <- size
        count <- count + 1L
      } else {
        scaleFactor <- self$scale * minImageSize * sqrt( aspectRatio ) /
          max( aspectRatioValues )
        widths[count] <- scaleFactor * aspectRatioValues[1]
        heights[count] <- scaleFactor * aspectRatioValues[2]
        depths[count] <- scaleFactor * aspectRatioValues[3]
        count <- count + 1L
      }
    }
    self$numberOfBoxes <- count - 1L

    boxDimensions <- list()
    boxDimensions[[1]] <- widths
    boxDimensions[[2]] <- heights
    boxDimensions[[3]] <- depths

    cellSize <- self$imageSize / layerSize
    centers <- list()
    for( i in 1:length( cellSize ) )
    {
      centers[[i]] <- seq( 0.5 * cellSize[i],
                           self$imageSize[i] - 0.5 * cellSize[i], length.out = layerSize[i] )
    }

    boxesTensor <- np$zeros( reticulate::tuple(
      layerSize[1], layerSize[2], layerSize[3], self$numberOfBoxes, 6L ) )

    grid <- np$meshgrid( centers[[1]], centers[[2]], centers[[3]] )
    for( i in 1:length( grid ) )
    {
      boxesTensor[,,,, i] <- np$tile( np$expand_dims( grid[[i]], axis = -1L ),
                                      reticulate::tuple( 1L, 1L, 1L, 1L, self$numberOfBoxes ) )
      boxesTensor[,,,, i + length( grid )] <- array( rep( boxDimensions[[i]],
                                                          each = layerSize[1] * layerSize[2] * layerSize[3] ),
                                                     dim = c( layerSize[1], layerSize[2], layerSize[3],
                                                              self$numberOfBoxes ) )
    }

    self$anchorBoxesArray <- reticulate::array_reshape( boxesTensor,
                                                        dim = c( layerSize[1] * layerSize[2] * layerSize[3] *
                                                                   self$numberOfBoxes, 6 ) )

    # Convert to (xmin, xmax, ymin, ymax, zmin, zmax)
    self$anchorBoxesArray <- convertCoordinates( self$anchorBoxesArray,
                                                 type = 'centroids2minmax' )

    variancesTensor <- np$zeros_like( boxesTensor )
    variancesTensor <- variancesTensor + self$variances

    anchorBoxesTensor = np$concatenate(
      reticulate::tuple( boxesTensor, variancesTensor ), axis = -1L )
    anchorBoxesTensor <- np$expand_dims( anchorBoxesTensor, axis = 0L )

    anchorBoxesTensor <- K$constant( anchorBoxesTensor, dtype = 'float32' )
    anchorBoxesTensor <- K$tile( anchorBoxesTensor,
                                 c( K$shape( x )[1], 1L, 1L, 1L, 1L, 1L ) )

    return( anchorBoxesTensor )
  },

  compute_output_shape = function( input_shape )
  {
    layerSize <- c()
    layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
    layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]
    layerSize[3] <- input_shape[[self$imageSizeAxes[3]]]

    return( reticulate::tuple( input_shape[[1]], layerSize[1],
                               layerSize[2], layerSize[3], self$numberOfBoxes, 12L ) )
  }
)
)

