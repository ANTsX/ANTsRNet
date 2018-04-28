#' 2-D implementation of the Vgg deep learning architecture.
#'
#' Creates a keras model of the Vgg deep learning architecture for image 
#' recognition based on the paper
#' 
#' K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for 
#'   Large-Scale Image Recognition
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1409.1556
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d       
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning.
#' @param convolutionKernelSize 2-d vector definining the kernel size 
#' during the encoding path
#' @param poolSize 2-d vector defining the region for each pooling layer.
#' @param strides 2-d vector describing the stride length in each direction.
#' @param denseUnits integer for the number of units in the last layers.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param style '16' or '19' for VGG16 or VGG19, respectively.
#'
#' @return a VGG keras model to be used with subsequent fitting
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
#'  vggModel <- createVggModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- vggModel %>% fit( X_train, Y_train, 
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

createVggModel2D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1000,
                               layers = c( 1, 2, 3, 4, 4 ), 
                               lowestResolution = 64, 
                               convolutionKernelSize = c( 3, 3 ), 
                               poolSize = c( 2, 2 ), 
                               strides = c( 2, 2 ),
                               denseUnits = 4096,
                               dropoutRate = 0.0,
                               style = 19
                             )
{

  if( style != 19 && style != 16 )
    {
    stop( "Incorrect style.  Must be either '16' or '19'." )
    }
  
  vggModel <- keras_model_sequential()

  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )    

    if( i == 1 )
      {
      vggModel %>% 
        layer_conv_2d( input_shape = inputImageSize, filters = numberOfFilters, 
                      kernel_size = convolutionKernelSize, activation = 'relu', 
                      padding = 'same' ) %>% 
        layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                      activation = 'relu', padding = 'same' )
      } else if( i == 2 ) {
      vggModel %>% 
        layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                      activation = 'relu', padding = 'same' ) %>% 
        layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                      activation = 'relu', padding = 'same' )
      }  else {
      if( style == 16 )  
        {
        vggModel %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' )
        } else {  # style == 19
        vggModel %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' )
        }  
      }
      
    vggModel %>% layer_max_pooling_2d( pool_size = poolSize, strides = strides )
    }

  vggModel %>% layer_flatten()
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = numberOfClassificationLabels, activation = 'softmax' )

  return( vggModel )
}

#' 3-D implementation of the Vgg deep learning architecture.
#'
#' Creates a keras model of the Vgg deep learning architecture for image 
#' recognition based on the paper
#' 
#' K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for 
#'   Large-Scale Image Recognition
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1409.1556
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d       
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning.
#' @param convolutionKernelSize 2-d vector definining the kernel size 
#' during the encoding path
#' @param poolSize 2-d vector defining the region for each pooling layer.
#' @param strides 2-d vector describing the stride length in each direction.
#' @param denseUnits integer for the number of units in the last layers.
#' @param dropoutRate float between 0 and 1 to use between dense layers.
#' @param style '16' or '19' for VGG16 or VGG19, respectively.
#'
#' @return a VGG keras model to be used with subsequent fitting
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
#'  vggModel <- createVggModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- vggModel %>% fit( X_train, Y_train, 
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

createVggModel3D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1000,
                               layers = c( 1, 2, 3, 4, 4 ), 
                               lowestResolution = 64, 
                               convolutionKernelSize = c( 3, 3, 3 ), 
                               poolSize = c( 2, 2, 2 ), 
                               strides = c( 2, 2, 2 ),
                               denseUnits = 4096,
                               dropoutRate = 0.0,
                               style = 19
                             )
{

  if( style != 19 && style != 16 )
    {
    stop( "Incorrect style.  Must be either '16' or '19'." )
    }
  
  vggModel <- keras_model_sequential()

  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )    

    if( i == 1 )
      {
      vggModel %>% 
        layer_conv_3d( input_shape = inputImageSize, filters = numberOfFilters, 
                      kernel_size = convolutionKernelSize, activation = 'relu', 
                      padding = 'same' ) %>% 
        layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                      activation = 'relu', padding = 'same' )
      } else if( i == 2 ) {
      vggModel %>% 
        layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                      activation = 'relu', padding = 'same' ) %>% 
        layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                      activation = 'relu', padding = 'same' )
      }  else {
      if( style == 16 )  
        {
        vggModel %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' )
        } else {  # style == 19
        vggModel %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>% 
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' ) %>%
          layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
                        activation = 'relu', padding = 'same' )
        }  
      }
      
    vggModel %>% layer_max_pooling_3d( pool_size = poolSize, strides = strides )
    }

  vggModel %>% layer_flatten()
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = numberOfClassificationLabels, 
                            activation = 'softmax' )
    
  return( vggModel )
}

