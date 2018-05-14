#' 2-D implementation of the U-net deep learning architecture.
#'
#' Creates a keras model of the U-net deep learning architecture for image 
#' segmentation.  More information is provided at the authors' website:
#' 
#'         \url{https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/}
#' 
#' with the paper available here:
#' 
#'         \url{https://arxiv.org/abs/1505.04597}
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         \url{https://github.com/joelthelion/ultrasound-nerve-segmentation}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning and end of 
#' the \verb{'U'}.
#' @param convolutionKernelSize 2-d vector defining the kernel size 
#' during the encoding path
#' @param deconvolutionKernelSize 2-d vector defining the kernel size 
#' during the decoding 
#' @param poolSize 2-d vector defining the region for each pooling layer.
#' @param strides 2-d vector describing the stride length in each direction.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#'
#' @return a u-net keras model to be used with subsequent fitting
#' @author Tustison NJ
#' @examples
#' # Simple examples, must run successfully and quickly. These will be tested.
#' \dontrun{ 
#' 
#'  library( ANTsR )
#'
#'  imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
#'
#'  # Perform simple 3-tissue segmentation.  For convenience we are going 
#   # to use kmeans segmentation to define the "ground-truth" segmentations.
#'  
#'  segmentationLabels <- c( 1, 2, 3 )
#'  
#'  images <- list()
#'  kmeansSegs <- list()
#'
#'  trainingImageArrays <- list()
#'  trainingMaskArrays <- list()
#'
#'  for( i in 1:length( imageIDs ) )
#'    {
#'    cat( "Processing image", imageIDs[i], "\n" )
#'    images[[i]] <- antsImageRead( getANTsRData( imageIDs[i] ) )
#'    mask <- getMask( images[[i]] )
#'    kmeansSegs[[i]] <- kmeansSegmentation( images[[i]], 
#'      length( segmentationLabels ), mask, mrf = 0.0 )$segmentation
#' 
#'    trainingImageArrays[[i]] <- as.array( images[[i]] )
#'    trainingMaskArrays[[i]] <- as.array( mask )
#'    }
#'  
#'  # reshape the training data to the format expected by keras
#'  
#'  trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
#'  trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )
#'
#'  trainingData <- abind( trainingImageArrays, along = 3 )   
#'  trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
#'  
#'  # Perform an easy normalization which is important for U-net. 
#'  # Other normalization methods might further improve results.
#'  
#'  trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )
#'
#'  X_train <- array( trainingData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  Y_train <- array( trainingLabelData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  
#'  # Create the model
#'  
#'  unetModel <- createUnetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- unetModel %>% fit( X_train, Y_train, 
#'                 epochs = 100, batch_size = 32, verbose = 1, shuffle = TRUE,
#'                 callbacks = list( 
#'                   callback_model_checkpoint( paste0( baseDirectory, "weights.h5" ), 
#'                      monitor = 'val_loss', save_best_only = TRUE ),
#'                 #  callback_early_stopping( patience = 2, monitor = 'loss' ),
#'                   callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#'                 ), 
#'                 validation_split = 0.2 )
#'
#'  # Save the model and/or save the model weights
#'
#'  save_model_hdf5( unetModel, filepath = 'unetModel.h5' )
#'  save_model_weights_hdf5( unetModel, filepath = 'unetModelWeights.h5' ) )
#' }
#' @import keras
#' @export
createUnetModel2D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1,
                               layers = 1:4, 
                               lowestResolution = 32, 
                               convolutionKernelSize = c( 3, 3 ), 
                               deconvolutionKernelSize = c( 2, 2 ), 
                               poolSize = c( 2, 2 ), 
                               strides = c( 2, 2 ),
                               dropoutRate = 0.0
                             )
{

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path  

  encodingConvolutionLayers <- list()
  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_2d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', 
        padding = 'same' )
      } else {
      conv <- pool %>% layer_conv_2d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', 
        padding = 'same' )
      }
    if( dropoutRate > 0.0 )
      {
      conv <- conv %>% layer_dropout( rate = dropoutRate )
      }

    encodingConvolutionLayers[[i]] <- conv %>% layer_conv_2d( 
      filters = numberOfFilters, kernel_size = convolutionKernelSize, 
      activation = 'relu', padding = 'same' )
    
    if( i < length( layers ) )
      {
      pool <- encodingConvolutionLayers[[i]] %>% 
        layer_max_pooling_2d( pool_size = poolSize, strides = strides,
        padding = 'same' )
      }
    }

  # Decoding path 

  outputs <- encodingConvolutionLayers[[length( layers )]]
  for( i in 2:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( length( layers ) - layers[i] )
    deconvolution <- outputs %>%
      layer_conv_2d_transpose( filters = numberOfFilters,
        kernel_size = deconvolutionKernelSize,
        padding = 'same' )
    deconvolution <- deconvolution %>% layer_upsampling_2d( size = poolSize )
    outputs <- layer_concatenate( list( deconvolution,
      encodingConvolutionLayers[[length( layers ) - i + 1]] ),
      axis = 3
      )

    if( dropoutRate > 0.0 )
      {
      outputs <- outputs %>% 
        layer_conv_2d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, 
          activation = 'relu', padding = 'same' ) %>%
        layer_dropout( rate = dropoutRate ) %>% 
        layer_conv_2d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, 
          activation = 'relu', padding = 'same'  )  
      } else {
      outputs <- outputs %>% 
        layer_conv_2d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, 
          activation = 'relu', padding = 'same' ) %>%
        layer_conv_2d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, 
          activation = 'relu', padding = 'same' )  
      }
    }
  if( numberOfClassificationLabels == 2 )  
    {
    outputs <- outputs %>% 
      layer_conv_2d( filters = numberOfClassificationLabels, 
        kernel_size = c( 1, 1 ), activation = 'sigmoid' )
    } else {
    outputs <- outputs %>% 
      layer_conv_2d( filters = numberOfClassificationLabels, 
        kernel_size = c( 1, 1 ), activation = 'softmax' )
    }
    
  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

  
#' 3-D image segmentation implementation of the U-net deep learning architecture.
#'
#' Creates a keras model of the U-net deep learning architecture for image 
#' segmentation.  More information is provided at the authors' website:
#' 
#'         \url{https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/}
#' 
#' with the paper available here:
#' 
#'         \url{https://arxiv.org/abs/1505.04597}
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         \url{https://github.com/joelthelion/ultrasound-nerve-segmentation}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning and end of 
#' the \verb{'U'}.
#' @param convolutionKernelSize 3-d vector defining the kernel size 
#' during the encoding path
#' @param deconvolutionKernelSize 3-d vector defining the kernel size 
#' during the decoding 
#' @param poolSize 3-d vector defining the region for each pooling layer.
#' @param strides 3-d vector describing the stride length in each direction.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#'
#' @return a u-net keras model to be used with subsequent fitting
#' @author Tustison NJ
#' @examples
#' # Simple examples, must run successfully and quickly. These will be tested.
#' \dontrun{ 
#' 
#'  library( ANTsR )
#'
#'  imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
#'
#'  # Perform simple 3-tissue segmentation.  For convenience we are going 
#   # to use kmeans segmentation to define the "ground-truth" segmentations.
#'  
#'  segmentationLabels <- c( 1, 2, 3 )
#'  
#'  images <- list()
#'  kmeansSegs <- list()
#'
#'  trainingImageArrays <- list()
#'  trainingMaskArrays <- list()
#'
#'  for( i in 1:length( imageIDs ) )
#'    {
#'    cat( "Processing image", imageIDs[i], "\n" )
#'    images[[i]] <- antsImageRead( getANTsRData( imageIDs[i] ) )
#'    mask <- getMask( images[[i]] )
#'    kmeansSegs[[i]] <- kmeansSegmentation( images[[i]], 
#'      length( segmentationLabels ), mask, mrf = 0.0 )$segmentation
#' 
#'    trainingImageArrays[[i]] <- as.array( images[[i]] )
#'    trainingMaskArrays[[i]] <- as.array( mask )
#'    }
#'  
#'  # reshape the training data to the format expected by keras
#'  
#'  trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
#'  trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )
#'
#'  trainingData <- abind( trainingImageArrays, along = 3 )   
#'  trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
#'  
#'  # Perform an easy normalization which is important for U-net. 
#'  # Other normalization methods might further improve results.
#'  
#'  trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )
#'
#'  X_train <- array( trainingData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  Y_train <- array( trainingLabelData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  
#'  # Create the model (3-D function is a straightforward analog)
#'  
#'  unetModel <- createUnetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- unetModel %>% fit( X_train, Y_train, 
#'                 epochs = 100, batch_size = 32, verbose = 1, shuffle = TRUE,
#'                 callbacks = list( 
#'                   callback_model_checkpoint( paste0( baseDirectory, "weights.h5" ), 
#'                      monitor = 'val_loss', save_best_only = TRUE ),
#'                 #  callback_early_stopping( patience = 2, monitor = 'loss' ),
#'                   callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#'                 ), 
#'                 validation_split = 0.2 )
#'
#'  # Save the model and/or save the model weights
#'
#'  save_model_hdf5( unetModel, filepath = 'unetModel.h5' )
#'  save_model_weights_hdf5( unetModel, filepath = 'unetModelWeights.h5' ) )
#' }
#' @import keras
#' @export
createUnetModel3D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1,
                               layers = 1:4, 
                               lowestResolution = 32, 
                               convolutionKernelSize = c( 3, 3, 3 ), 
                               deconvolutionKernelSize = c( 2, 2, 2 ), 
                               poolSize = c( 2, 2, 2 ), 
                               strides = c( 2, 2, 2 ),
                               dropoutRate = 0.0
                             )
{

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path  

  encodingConvolutionLayers <- list()
  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_3d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', 
        padding = 'same' )
      } else {
      conv <- pool %>% layer_conv_3d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', 
        padding = 'same' )
      }
    if( dropoutRate > 0.0 )
      {
      conv <- conv %>% layer_dropout( rate = dropoutRate )
      }

    encodingConvolutionLayers[[i]] <- conv %>% layer_conv_3d( 
      filters = numberOfFilters, kernel_size = convolutionKernelSize, 
      activation = 'relu', padding = 'same' )
    
    if( i < length( layers ) )
      {
      pool <- encodingConvolutionLayers[[i]] %>% 
        layer_max_pooling_3d( pool_size = poolSize, strides = strides,
        padding = 'same' )
      }
    }

  # Decoding path 

  outputs <- encodingConvolutionLayers[[length( layers )]]
  for( i in 2:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( length( layers ) - layers[i] )
    deconvolution <- outputs %>%
      layer_conv_2d_transpose( filters = numberOfFilters,
        kernel_size = deconvolutionKernelSize,
        padding = 'same' )
    deconvolution <- deconvolution %>% layer_upsampling_2d( size = poolSize )
    outputs <- layer_concatenate( list( deconvolution,
      encodingConvolutionLayers[[length( layers ) - i + 1]] ),
      axis = 4
      )
      
    if( dropoutRate > 0.0 )
      {
      outputs <- outputs %>% 
        layer_conv_3d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, activation = 'relu', 
          padding = 'same' )  %>%
        layer_dropout( rate = dropoutRate ) %>% 
        layer_conv_3d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, activation = 'relu', 
          padding = 'same' )  
      } else {
      outputs <- outputs %>% 
        layer_conv_3d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, 
          activation = 'relu', padding = 'same'  )  %>%
        layer_conv_3d( filters = numberOfFilters, 
          kernel_size = convolutionKernelSize, 
          activation = 'relu', padding = 'same'  )  
      }
    }  
  if( numberOfClassificationLabels == 2 )  
    {
    outputs <- outputs %>% 
      layer_conv_3d( filters = numberOfClassificationLabels, 
        kernel_size = c( 1, 1, 1 ), activation = 'sigmoid' )
    } else {
    outputs <- outputs %>% 
      layer_conv_3d( filters = numberOfClassificationLabels, 
        kernel_size = c( 1, 1, 1 ), activation = 'softmax' )
    }

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
  