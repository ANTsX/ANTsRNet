#' 2-D implementation of the SSD deep learning architecture.
#'
#' Creates a keras model of the SSD deep learning architecture for 
#' object detection based on the paper
#' 
#' W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, A. Berg. 
#'     SSD: Single Shot MultiBox Detector.
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1512.02325
#'
#' This particular implementation was influenced by the following python 
#' and R implementations: 
#' 
#'         https://github.com/pierluigiferrari/ssd_keras     
#'         https://github.com/rykov8/ssd_keras
#'         https://github.com/gsimchoni/ssdkeras
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of classification labels. 
#' Needs to include the background as one of the labels. 
#' @param l2Regularization The L2-regularization rate.  Default = 0.0005.
#' @param minScale The smallest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images.
#' @param maxScale The largest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images. All scaling 
#' factors between the smallest and the largest are linearly interpolated. 
#' @param aspectRatiosPerLayer A list containing one aspect ratio list for
#' each predictor layer.  The default lists follows the original 
#' implementation except each aspect ratio is defined as a character string
#' (e.g. \verb{'1:2'}).
#' @param variances A list of 4 floats > 0 with scaling factors for the encoded 
#' predicted box coordinates. A variance value of 1.0 would apply no scaling at 
#' all to the predictions, while values in (0,1) upscale the encoded 
#' predictions and values greater than 1.0 downscale the encoded predictions. 
#' Defaults to 1.0.
#'
#' @return an SSD keras model
#' @author Tustison NJ

createSsdModel2D <- function( inputImageSize, 
                              numberOfClassificationLabels,
                              l2Regularization = 0.0005,
                              minScale = 0.1,
                              maxScale = 0.9,
                              aspectRatiosPerLayer = 
                                list( c( '1:1', '2:1', '1:2' ),
                                      c( '1:1', '2:1', '1:2', '3:1', '1:3' ),
                                      c( '1:1', '2:1', '1:2', '3:1', '1:3' ),
                                      c( '1:1', '2:1', '1:2', '3:1', '1:3' ),
                                      c( '1:1', '2:1', '1:2' ),
                                      c( '1:1', '2:1', '1:2' )
                                    ),
                              variances = rep( 1.0, 4 ),
                              style = 300
                            )
{

  if( style != 300 && style != 512 )
    {
    stop( "Incorrect style.  Must be either '300' or '512'." )
    }

  K <- keras::backend()  

  filterSizes <- c( 64, 128, 256, 512, 1024 ) 

  numberOfPredictorLayers <- 6
  if( style == 512 )
    {
    numberOfPredictorLayers <- 7  
    }
  
  if( length( aspectRatiosPerLayer ) != numberOfPredictorLayers )
    {
    stop( paste( "The number of sets of aspect ratios should be equal to",
       numberOfPredictorLayers, "\n") )
    }

  numberOfBoxesPerLayer <- rep( 0, numberOfPredictorLayers )
  for( i in 1:numberOfPredictorLayers )
    {
    numberOfBoxesPerLayer[i] <- length( aspectRatiosPerLayer[[i]] )  
    if( 1 %in% aspectRatiosPerLayer[[i]] )
      {
      numberOfBoxesPerLayer[i] <- numberOfBoxesPerLayer[i] + 1   
      }
    }

  scales <- seq( from = minScale, to = maxScale, 
    length.out = numberOfPredictorLayers + 1 )

  # For each of the \code{numberOfClassificationLabels}, we predict confidence 
  # values for each box.  This translates into each confidence predictor 
  # having a depth of  \code{numberOfBoxesPerLayer * numberOfClassificationLabels}.
  boxClasses <- list()

  # For each box we need to predict the 2 * imageDimension coordinates.  The 
  # output shape of these localization layers is:
  # \code{( batchSize, imageHeight, imageWidth, 
  #      numberOfBoxesPerLayer * 2 * imageDimension )}
  boxLocations <- list()

  imageDimension <- 2
  numberOfCoordinates <- 2 * imageDimension

  # Initial convolutions 1-4

  inputs <- layer_input( shape = inputImageSize, name = "input_4" )

  outputs <- inputs

  numberOfLayers <- 4
  for( i in 1:numberOfLayers )
    {
    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
      kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), 
      activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = paste0( "conv", i, "_1" ) ) 

    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
      kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = paste0( "conv", i, "_2" ) ) 

    if( i > 2 ) 
      {
      outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
        kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), 
        activation = 'relu', padding = 'same', 
        kernel_initializer = initializer_he_normal(), 
        kernel_regularizer = regularizer_l2( l2Regularization ),
        name = paste0( "conv", i, "_3" ) )  

      if( i == numberOfLayers )
        {
        l2NormalizedOutputs <- outputs %>% 
          layer_l2_normalization_2d( scale = 20, name = "conv4_3_norm" )

        boxClasses[[1]] <- l2NormalizedOutputs %>% layer_conv_2d( 
          filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels, 
          kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
          padding = 'same', kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( l2Regularization ), 
          name = "conv4_3_norm_mbox_conf" )

        boxLocations[[1]] <- l2NormalizedOutputs %>% layer_conv_2d( 
          filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels,
          kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
          padding = 'same', kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( l2Regularization ), 
          name = "conv4_3_norm_mbox_loc" )
        }
      }

    outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ), 
      strides = c( 2, 2 ), padding = 'same',
      name = paste0( "pool", i ) )
    }

  # Conv5

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4], 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv5_1" )

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4], 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv5_2" )

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4], 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv5_3" )

  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same', name = "pool5" )

  # fc6

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[5],
    kernel_size = c( 3, 3 ), dilation_rate = c( 6L, 6L ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), name = "fc6" ) 

  # fc7

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[5],
    kernel_size = c( 1, 1 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), name = "fc7" ) 

  boxClasses[[2]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[2] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "fc7_mbox_conf" )

  boxLocations[[2]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[2] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "fc7_mbox_loc" )

  # Conv6

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 1, 1 ),  dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv6_1" )

  outputs <- outputs %>% layer_zero_padding_2d( 
    padding = list( c( 1, 1 ), c( 1, 1 ) ), name = "conv6_padding" )  

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4],
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), strides = c( 2, 2 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ),
    name = "conv6_2" ) 

  boxClasses[[3]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[3] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv6_2_mbox_conf" )

  boxLocations[[3]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[3] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv6_2_mbox_loc" )

  # Conv7

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
    kernel_size = c( 1, 1 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ),
    name = "conv7_1" ) 

  outputs <- outputs %>% layer_zero_padding_2d( 
    padding = list( c( 1, 1 ), c( 1, 1 ) ),
    name = "conv7_padding" ) 

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), strides = c( 2, 2 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ),
    name = "conv7_2" ) 

  boxClasses[[4]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[4] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv7_2_mbox_conf" )

  boxLocations[[4]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[4] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv7_2_mbox_loc" )

  # Conv8

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
    kernel_size = c( 1, 1 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_1" ) 

  if( style == 512 )
    {
    outputs <- outputs %>% layer_zero_padding_2d( 
      padding = list( c( 1, 1 ), c( 1, 1 ) ), name = "conv8_padding" )  
    }

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), strides = c( 1, 1 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_2" ) 

  boxClasses[[5]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[5] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_2_mbox_conf" )

  boxLocations[[5]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[5] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_2_mbox_loc" )

  # Conv9

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
    kernel_size = c( 1, 1 ), dilation_rate = c( 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_1" ) 

  if( style == 512 )
    {
    outputs <- outputs %>% layer_zero_padding_2d( 
      padding = list( c( 1, 1 ), c( 1, 1 ) ), name = "conv9_padding" )  
    } 

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), strides = c( 1, 1 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_2" ) 

  boxClasses[[6]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[6] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), padding = 'same', 
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_2_mbox_conf" )

  boxLocations[[6]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[6] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), padding = 'same', 
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_2_mbox_loc" )

  if( style == 512 )
    {
    # Conv10

    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
      kernel_size = c( 1, 1 ), dilation_rate = c( 1L, 1L ),
      activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_1" ) 

    outputs <- outputs %>% layer_zero_padding_2d( 
      padding = list( c( 1, 1 ), c( 1, 1 ) ), name = "conv10_padding" )  

    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
      kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), strides = c( 1, 1 ), 
      activation = 'relu', padding = 'valid', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_2" ) 

    boxClasses[[7]] <- outputs %>% layer_conv_2d( 
      filters = numberOfBoxesPerLayer[7] * numberOfClassificationLabels, 
      kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), padding = 'same', 
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_2_mbox_conf" )

    boxLocations[[7]] <- outputs %>% layer_conv_2d( 
      filters = numberOfBoxesPerLayer[7] * numberOfCoordinates, 
      kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L ), padding = 'same', 
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_2_mbox_loc" )
    } 

  # Generate the anchor boxes.  Output shape of anchor boxes =
  #   \code{( batch, height, width, numberOfBoxes, 8L )}
  anchorBoxes <- list()
  anchorBoxLayers <- list()
  predictorSizes <- list()

  imageSize <- inputImageSize[1:imageDimension]
  
  layerNames <- paste0( c( "conv4_3_norm", "fc7", "conv6_2", "conv7_2", 
    "conv8_2", "conv9_2", "conv10_2" ), "_mbox" )
  if( style == 300 )
    {
    layerNames <- head( layerNames, -1 ) 
    }

  for( i in 1:length( boxLocations ) )
    {
    anchorBoxLayer <- layer_anchor_box_2d( imageSize = imageSize, 
      scale = scales[i], nextScale = scales[i + 1],
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances, 
      name = paste0( 'anchors', i + 3 ) )
    anchorBoxLayers[[i]] <- boxLocations[[i]] %>% anchorBoxLayer

    # We calculate the anchor box values again to return as output for 
    # encoding Y_train.  I'm guessing there's a better way to do this 
    # but it's the cleanest I've found.
    anchorBoxGenerator <- AnchorBoxLayer2D$new( imageSize = imageSize,
      scales[i], scales[i + 1],
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances )
    anchorBoxGenerator$call( boxLocations[[i]] )  
    anchorBoxes[[i]] <- anchorBoxGenerator$anchorBoxesArray
    }

  # Reshape the box confidence values, box locations, and 
  boxClassesReshaped <- list()
  boxLocationsReshaped <- list()
  anchorBoxLayersReshaped <- list()
  for( i in 1:length( boxClasses ) )
    {
    # reshape \code{( batch, height, width, numberOfBoxes * numberOfClasses )}
    #   to \code{(batch, height * width * numberOfBoxes, numberOfClasses )}
    inputShape <- K$int_shape( boxClasses[[i]] )
    numberOfBoxes <- 
      as.integer( inputShape[[4]] / numberOfClassificationLabels )

    boxClassesReshaped[[i]] <- boxClasses[[i]] %>% layer_reshape( 
      target_shape = c( -1, numberOfClassificationLabels ), 
      name = paste0( layerNames[i], "_conf_reshape" ) )

    # reshape \code{( batch, height, width, numberOfBoxes * 4 )}
    #   to \code{( batch, height * width * numberOfBoxes, 4 )}
    boxLocationsReshaped[[i]] <- boxLocations[[i]] %>% layer_reshape( 
      target_shape = c( -1, 4 ), 
      name = paste0( layerNames[i], "_loc_reshape" ) )

    # reshape \code{( batch, height, width, numberOfBoxes * 8 )}
    #   to \code{( batch, height * width * numberOfBoxes, 8 )}
    anchorBoxLayersReshaped[[i]] <- anchorBoxLayers[[i]] %>% layer_reshape( 
      target_shape = c( -1, 8 ), 
      name = paste0( layerNames[i], "_priorbox_reshape" ) )
    }  
  
  # Concatenate the predictions from the different layers

  outputClasses <- layer_concatenate( boxClassesReshaped, 
    axis = 1, trainable = TRUE, name = "mbox_conf" )
  outputLocations <- layer_concatenate( boxLocationsReshaped, 
    axis = 1, trainable = TRUE, name = "mbox_loc" )
  outputAnchorBoxes <- layer_concatenate( anchorBoxLayersReshaped, 
    axis = 1, trainable = TRUE, name = "mbox_priorbox" )

  confidenceActivation <- outputClasses %>% 
    layer_activation( activation = "softmax", name = "mbox_conf_softmax" )

  predictions <- layer_concatenate( list( confidenceActivation, 
    outputLocations, outputAnchorBoxes ), axis = 2, trainable = TRUE, 
    name = "predictions" )

  ssdModel <- keras_model( inputs = inputs, outputs = predictions )

  return( list( ssdModel = ssdModel, anchorBoxes = anchorBoxes ) )
}

#' 3-D implementation of the SSD deep learning architecture.
#'
#' Creates a keras model of the SSD deep learning architecture for 
#' object detection based on the paper
#' 
#' W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, A. Berg. 
#'     SSD: Single Shot MultiBox Detector.
#' 
#' available here:
#' 
#'         \url{https://arxiv.org/abs/1512.02325}
#'
#' This particular implementation was influenced by the following python 
#' and R implementations: 
#' 
#'         https://github.com/pierluigiferrari/ssd_keras     
#'         https://github.com/rykov8/ssd_keras
#'         https://github.com/gsimchoni/ssdkeras
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of classification labels. 
#' Needs to include the background as one of the labels. 
#' @param l2Regularization The L2-regularization rate.  Default = 0.0005.
#' @param minScale The smallest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images.
#' @param maxScale The largest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images. All scaling 
#' factors between the smallest and the largest are linearly interpolated. 
#' @param aspectRatiosPerLayer A list containing one aspect ratio list for
#' each predictor layer.  The default lists follows the original 
#' implementation except each aspect ratio is defined as a character string
#' (e.g. \verb{'1:1:2'}).
#' @param variances A list of 6 floats > 0 with scaling factors for the encoded 
#' predicted box coordinates. A variance value of 1.0 would apply no scaling at 
#' all to the predictions, while values in (0,1) upscale the encoded 
#' predictions and values greater than 1.0 downscale the encoded predictions. 
#' Defaults to 1.0.
#'
#' @return an SSD keras model
#' @author Tustison NJ

createSsdModel3D <- function( inputImageSize, 
                              numberOfClassificationLabels,
                              l2Regularization = 0.0005,
                              minScale = 0.1,
                              maxScale = 0.9,
                              aspectRatiosPerLayer = 
                                list( c( '1:1:1', '2:1:1', '1:2:1', '1:1:2' ),
                                      c( '1:1:1', '2:1:1', '1:2:1', '1:1:2', '3:1:1', '1:3:1', '1:1:3' ),
                                      c( '1:1:1', '2:1:1', '1:2:1', '1:1:2', '3:1:1', '1:3:1', '1:1:3' ),
                                      c( '1:1:1', '2:1:1', '1:2:1', '1:1:2', '3:1:1', '1:3:1', '1:1:3' ),
                                      c( '1:1:1', '2:1:1', '1:2:1', '1:1:2' ),
                                      c( '1:1:1', '2:1:1', '1:2:1', '1:1:2' )
                                    ),
                              variances = rep( 1.0, 6 ),
                              style = 300
                            )
{

  if( style != 300 && style != 512 )
    {
    stop( "Incorrect style.  Must be either '300' or '512'." )
    }

  K <- keras::backend()  

  filterSizes <- c( 64, 128, 256, 512, 1024 ) 

  numberOfPredictorLayers <- 6
  if( style == 512 )
    {
    numberOfPredictorLayers <- 7  
    }
  
  if( length( aspectRatiosPerLayer ) != numberOfPredictorLayers )
    {
    stop( paste( "The number of sets of aspect ratios should be equal to",
       numberOfPredictorLayers, "\n") )
    }

  numberOfBoxesPerLayer <- rep( 0, numberOfPredictorLayers )
  for( i in 1:numberOfPredictorLayers )
    {
    numberOfBoxesPerLayer[i] <- length( aspectRatiosPerLayer[[i]] )  
    if( 1 %in% aspectRatiosPerLayer[[i]] )
      {
      numberOfBoxesPerLayer[i] <- numberOfBoxesPerLayer[i] + 1   
      }
    }

  scales <- seq( from = minScale, to = maxScale, 
    length.out = numberOfPredictorLayers + 1 )

  # For each of the \code{numberOfClassificationLabels}, we predict confidence 
  # values for each box.  This translates into each confidence predictor 
  # having a depth of  \code{numberOfBoxesPerLayer * numberOfClassificationLabels}.
  boxClasses <- list()

  # For each box we need to predict the 2 * imageDimension coordinates.  The 
  # output shape of these localization layers is:
  # \code{( batchSize, imageHeight, imageWidth, 
  #      numberOfBoxesPerLayer * 2 * imageDimension )}
  boxLocations <- list()

  imageDimension <- 3
  numberOfCoordinates <- 2 * imageDimension

  # Initial convolutions 1-4

  inputs <- layer_input( shape = inputImageSize, name = "input_4" )

  outputs <- inputs

  numberOfLayers <- 4
  for( i in 1:numberOfLayers )
    {
    outputs <- outputs %>% layer_conv_3d( filters = filterSizes[i], 
      kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
      activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = paste0( "conv", i, "_1" ) ) 

    outputs <- outputs %>% layer_conv_3d( filters = filterSizes[i], 
      kernel_size = c( 3, 3, 3 ), activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = paste0( "conv", i, "_2" ) ) 

    if( i > 2 ) 
      {
      outputs <- outputs %>% layer_conv_3d( filters = filterSizes[i], 
        kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
        activation = 'relu', padding = 'same', 
        kernel_initializer = initializer_he_normal(), 
        kernel_regularizer = regularizer_l2( l2Regularization ),
        name = paste0( "conv", i, "_3" ) )  

      if( i == numberOfLayers )
        {
        l2NormalizedOutputs <- outputs %>% 
          layer_l2_normalization_3d( scale = 20, name = "conv4_3_norm" )

        boxClasses[[1]] <- l2NormalizedOutputs %>% layer_conv_3d( 
          filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels, 
          kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
          padding = 'same', kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( l2Regularization ), 
          name = "conv4_3_norm_mbox_conf" )

        boxLocations[[1]] <- l2NormalizedOutputs %>% layer_conv_3d( 
          filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels,
          kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
          padding = 'same', kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( l2Regularization ), 
          name = "conv4_3_norm_mbox_loc" )
        }
      }

    outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ), 
      strides = c( 2, 2, 2 ), padding = 'same',
      name = paste0( "pool", i ) )
    }

  # Conv5

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[4], 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv5_1" )

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[4], 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv5_2" )

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[4], 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv5_3" )

  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ), 
    strides = c( 1, 1, 1 ), padding = 'same', name = "pool5" )

  # fc6

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[5],
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 6L, 6L, 6L ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), name = "fc6" ) 

  # fc7

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[5],
    kernel_size = c( 1, 1, 1 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), name = "fc7" ) 

  boxClasses[[2]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[2] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "fc7_mbox_conf" )

  boxLocations[[2]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[2] * numberOfCoordinates, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "fc7_mbox_loc" )

  # Conv6

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[3],
    kernel_size = c( 1, 1, 1 ),  dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv6_1" )

  outputs <- outputs %>% layer_zero_padding_3d( 
    padding = list( c( 1, 1 ), c( 1, 1 ), c( 1, 1 ) ), name = "conv6_padding" )  

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[4],
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
    strides = c( 2, 2, 2 ), activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ),
    name = "conv6_2" ) 

  boxClasses[[3]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[3] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv6_2_mbox_conf" )

  boxLocations[[3]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[3] * numberOfCoordinates, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv6_2_mbox_loc" )

  # Conv7

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[2],
    kernel_size = c( 1, 1, 1 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ),
    name = "conv7_1" ) 

  outputs <- outputs %>% layer_zero_padding_3d( 
    padding = list( c( 1, 1 ), c( 1, 1 ), c( 1, 1 ) ),
    name = "conv7_padding" ) 

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[3],
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
    strides = c( 2, 2, 2 ), activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ),
    name = "conv7_2" ) 

  boxClasses[[4]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[4] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv7_2_mbox_conf" )

  boxLocations[[4]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[4] * numberOfCoordinates, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv7_2_mbox_loc" )

  # Conv8

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[2],
    kernel_size = c( 1, 1, 1 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_1" ) 

  if( style == 512 )
    {
    outputs <- outputs %>% layer_zero_padding_3d( 
      padding = list( c( 1, 1 ), c( 1, 1 ), c( 1, 1 ) ), 
      name = "conv8_padding" )  
    }

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
    strides = c( 1, 1, 1 ), activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_2" ) 

  boxClasses[[5]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[5] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_2_mbox_conf" )

  boxLocations[[5]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[5] * numberOfCoordinates, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv8_2_mbox_loc" )

  # Conv9

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[2],
    kernel_size = c( 1, 1, 1 ), dilation_rate = c( 1L, 1L, 1L ),
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_1" ) 

  if( style == 512 )
    {
    outputs <- outputs %>% layer_zero_padding_3d( 
      padding = list( c( 1, 1 ), c( 1, 1 ), c( 1, 1 ) ), 
      name = "conv9_padding" )  
    } 

  outputs <- outputs %>% layer_conv_3d( filters = filterSizes[3],
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
    strides = c( 1, 1, 1 ), activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_2" ) 

  boxClasses[[6]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[6] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_2_mbox_conf" )

  boxLocations[[6]] <- outputs %>% layer_conv_3d( 
    filters = numberOfBoxesPerLayer[6] * numberOfCoordinates, 
    kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ), 
    name = "conv9_2_mbox_loc" )

  if( style == 512 )
    {
    # Conv10

    outputs <- outputs %>% layer_conv_3d( filters = filterSizes[2],
      kernel_size = c( 1, 1, 1 ), dilation_rate = c( 1L, 1L, 1L ),
      activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_1" ) 

    outputs <- outputs %>% layer_zero_padding_3d( 
      padding = list( c( 1, 1 ), c( 1, 1 ), c( 1, 1 ) ), 
      name = "conv10_padding" )  

    outputs <- outputs %>% layer_conv_3d( filters = filterSizes[3],
      kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ),
      strides = c( 1, 1, 1 ), activation = 'relu', padding = 'valid', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_2" ) 

    boxClasses[[7]] <- outputs %>% layer_conv_3d( 
      filters = numberOfBoxesPerLayer[7] * numberOfClassificationLabels, 
      kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
      padding = 'same', kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_2_mbox_conf" )

    boxLocations[[7]] <- outputs %>% layer_conv_3d( 
      filters = numberOfBoxesPerLayer[7] * numberOfCoordinates, 
      kernel_size = c( 3, 3, 3 ), dilation_rate = c( 1L, 1L, 1L ), 
      padding = 'same', kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( l2Regularization ), 
      name = "conv10_2_mbox_loc" )
    } 

  # Generate the anchor boxes.  Output shape of anchor boxes =
  #   \code{( batch, height, width, depth, numberOfBoxes, 12L )}
  anchorBoxes <- list()
  anchorBoxLayers <- list()
  predictorSizes <- list()

  imageSize <- inputImageSize[1:imageDimension]
  
  layerNames <- paste0( c( "conv4_3_norm", "fc7", "conv6_2", "conv7_2", 
    "conv8_2", "conv9_2", "conv10_2" ), "_mbox" )
  if( style == 300 )
    {
    layerNames <- head( layerNames, -1 ) 
    }

  for( i in 1:length( boxLocations ) )
    {
    anchorBoxLayer <- layer_anchor_box_3d( imageSize = imageSize, 
      scale = scales[i], nextScale = scales[i + 1],
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances, 
      name = paste0( 'anchors', i + 3 ) )
    anchorBoxLayers[[i]] <- boxLocations[[i]] %>% anchorBoxLayer

    # We calculate the anchor box values again to return as output for 
    # encoding Y_train.  I'm guessing there's a better way to do this 
    # but it's the cleanest I've found.
    anchorBoxGenerator <- AnchorBoxLayer3d$new( imageSize = imageSize,
      scales[i], scales[i + 1],
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances )
    anchorBoxGenerator$call( boxLocations[[i]] )  
    anchorBoxes[[i]] <- anchorBoxGenerator$anchorBoxesArray
    }

  # Reshape the box confidence values, box locations 
  boxClassesReshaped <- list()
  boxLocationsReshaped <- list()
  anchorBoxLayersReshaped <- list()
  for( i in 1:length( boxClasses ) )
    {
    # reshape \code{( batch, height, width, depth, numberOfBoxes * numberOfClasses )}
    #   to \code{(batch, height * width * depth * numberOfBoxes, numberOfClasses )}
    inputShape <- K$int_shape( boxClasses[[i]] )
    numberOfBoxes <- 
      as.integer( inputShape[[4]] / numberOfClassificationLabels )

    boxClassesReshaped[[i]] <- boxClasses[[i]] %>% layer_reshape( 
      target_shape = c( -1, numberOfClassificationLabels ), 
      name = paste0( layerNames[i], "_conf_reshape" ) )

    # reshape \code{( batch, height, width, depth, numberOfBoxes * 6L )}
    #   to \code{( batch, height * width * depth * numberOfBoxes, 6L )}
    boxLocationsReshaped[[i]] <- boxLocations[[i]] %>% layer_reshape( 
      target_shape = c( -1, 6L ), 
      name = paste0( layerNames[i], "_loc_reshape" ) )

    # reshape \code{( batch, height, width, depth, numberOfBoxes * 12 )}
    #   to \code{( batch, height * width * depth * numberOfBoxes, 12 )}
    anchorBoxLayersReshaped[[i]] <- anchorBoxLayers[[i]] %>% layer_reshape( 
      target_shape = c( -1, 12L ), 
      name = paste0( layerNames[i], "_priorbox_reshape" ) )
    }  
  
  # Concatenate the predictions from the different layers

  outputClasses <- layer_concatenate( boxClassesReshaped, 
    axis = 1, trainable = TRUE, name = "mbox_conf" )
  outputLocations <- layer_concatenate( boxLocationsReshaped, 
    axis = 1, trainable = TRUE, name = "mbox_loc" )
  outputAnchorBoxes <- layer_concatenate( anchorBoxLayersReshaped, 
    axis = 1, trainable = TRUE, name = "mbox_priorbox" )

  confidenceActivation <- outputClasses %>% 
    layer_activation( activation = "softmax", name = "mbox_conf_softmax" )

  predictions <- layer_concatenate( list( confidenceActivation, 
    outputLocations, outputAnchorBoxes ), axis = 2, trainable = TRUE, 
    name = "predictions" )

  ssdModel <- keras_model( inputs = inputs, outputs = predictions )

  return( list( ssdModel = ssdModel, anchorBoxes = anchorBoxes ) )
}

