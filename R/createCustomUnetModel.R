#' Implementation of the "NoBrainer" U-net architecture
#'
#' Creates a keras model implementation of the u-net architecture
#' avaialable here:
#'
#'         \url{https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/unet.py}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param numberOfOutputs Number of segmentation labels.
#' @param numberOfFiltersAtBaseLayer number of filters at the beginning and end
#' of the \verb{'U'}.  Doubles at each descending/ascending layer. Default = 16.
#' @param convolutionKernelSize 3-d vector defining the kernel size
#' during the encoding path.
#' @param deconvolutionKernelSize 3-d vector defining the kernel size
#' during the decoding.
#'
#' @return a u-net keras model
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{
#'
#' library( ANTsRNet )
#' library( keras )
#'
#' model <- createNoBrainerUnetModel3D( list( NULL, NULL, NULL, 1 ) )
#' url <- "https://github.com/neuronets/nobrainer-models/releases/download/0.1/brain-extraction-unet-128iso-weights.h5"
#' weightsFile <- "nobrainerWeights.h5"
#' download.file( url, weightsFile )
#' model$load_weights( weightsFile )
#'
#' url <- "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Example/1097782_defaced_MPRAGE.nii.gz?raw=true"
#' imageFile <- "head.nii.gz"
#' download.file( url, imageFile )
#'
#' image = ( antsImageRead( imageFile ) %>% iMath( "Normalize" ) ) * 255.0
#' imageResampled <- resampleImage( image, rep( 256, 3 ), useVoxels = TRUE )
#' imageArray <- array( as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )
#'
#' brainMaskArray <- predict( model, imageArray )
#' brainMaskResampled <- as.antsImage( brainMaskArray[1,,,,1] ) %>%
#'   antsCopyImageInfo2( imageResampled )
#' brainMaskImage = resampleImage( brainMaskResampled, dim( image ),
#'   useVoxels = TRUE, interpType = "nearestneighbor" )
#' minimumBrainVolume <- round( 649933.7 / prod( antsGetSpacing( image ) ) )
#' brainMaskLabeled = labelClusters( brainMaskImage, minimumBrainVolume )
#' antsImageWrite( brainMaskLabeled, 'brainMask.nii.gz' )
#' }
#'
#' @import keras
#' @export
createNoBrainerUnetModel3D <- function( inputImageSize,
                                        numberOfOutputs = 1,
                                        numberOfFiltersAtBaseLayer = 16,
                                        convolutionKernelSize = c( 3, 3, 3 ),
                                        deconvolutionKernelSize = c( 2, 2, 2 )
                                      )
{

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  outputs <- inputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 16,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  # Decoding path

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 16,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip3, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip2, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip1, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  convActivation <- ''
  if( numberOfOutputs == 1 )
    {
    convActivation <- 'sigmoid'
    } else {
    convActivation <- 'softmax'
    }

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = 1, activation = convActivation )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
