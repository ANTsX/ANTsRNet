#' 2-D implementation of the deep back-projection network.
#'
#' Creates a keras model of the deep back-project network for image super
#' resolution.  More information is provided at the authors' website:
#'
#'         \url{https://www.toyota-ti.ac.jp/Lab/Denshi/iim/members/muhammad.haris/projects/DBPN.html}
#'
#' with the paper available here:
#'
#'         \url{https://arxiv.org/abs/1803.02735}
#'
#' This particular implementation was influenced by the following keras (python)
#' implementation:
#'
#'         \url{https://github.com/rajatkb/DBPN-Keras}
#'
#' with help from the original author's Caffe and Pytorch implementations:
#'
#'         \url{https://github.com/alterzero/DBPN-caffe}
#'         \url{https://github.com/alterzero/DBPN-Pytorch}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param numberOfOutputs number of outputs (e.g., 3 for RGB images).
#' @param numberOfFeatureFilters number of feature filters.
#' @param numberOfBaseFilters number of base filters.
#' @param numberOfBackProjectionStages number of up-/down-projection stages.  This
#' number includes the final up block.
#' @param convolutionKernelSize kernel size for certain convolutional layers.  This
#' and \code{strides} are dependent on the scale factor discussed in
#' original paper.  Factors used in the original implementation are as follows:
#' 2x --> \code{convolutionKernelSize = c( 2, 2 )},
#' 4x --> \code{convolutionKernelSize = c( 8, 8 )},
#' 8x --> \code{convolutionKernelSize = c( 12, 12 )}.  We default to 8x parameters.
#' @param strides strides for certain convolutional layers.  This and the
#' \code{convolutionKernelSize} are dependent on the scale factor discussed in
#' original paper.  Factors used in the original implementation are as follows:
#' 2x --> \code{strides = c( 2, 2 )}, 4x --> \code{strides = c( 4, 4 )}, 8x -->
#' \code{strides = c( 8, 8 )}. We default to 8x parameters.
#'
#' @return a keras model defining the deep back-projection network.
#' @author Tustison NJ
#' @examples
#' #
#' \dontrun{
#'
#' }
#' @import keras
#' @export
createDeepBackProjectionNetworkModel2D <-
                    function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfBaseFilters = 64,
                               numberOfFeatureFilters = 256,
                               numberOfBackProjectionStages = 7,
                               convolutionKernelSize = c( 12, 12 ),
                               strides = c( 8, 8 )
                             )
{

  upBlock2D <- function( L, numberOfFilters = 64, kernelSize = c( 12, 12 ),
    strides = c( 8, 8 ), includeDenseConvolutionLayer = FALSE, numberOfStages = 1 )
    {
    if( includeDenseConvolutionLayer )
      {
      L <- L %>% layer_conv_2d( filters = numberOfFilters, # * numberOfStages,
        kernel_size = c( 1, 1 ), stride = c( 1, 1 ), padding = 'same', use_bias = TRUE )
      L <- L %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )
      }

    # Scale up
    H0 <- L %>% layer_conv_2d_transpose( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    H0 <- H0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

    # Scale down
    L0 <- H0 %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    L0 <- L0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

    # Residual
    E <- layer_subtract( list( L0, L ) )

    # Scale residual up
    H1 <- E %>% layer_conv_2d_transpose( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    H1 <- H1 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

    # Output feature map
    upBlock <- layer_add( list( H0, H1 ) )

    return( upBlock )
    }

  downBlock2D <- function( H, numberOfFilters = 64, kernelSize = c( 12, 12 ),
    strides = c( 8, 8 ), includeDenseConvolutionLayer = FALSE, numberOfStages = 1 )
    {
    if( includeDenseConvolutionLayer )
      {
      H <- H %>% layer_conv_2d( filters = numberOfFilters, # * numberOfStages,
        kernel_size = c( 1, 1 ), stride = c( 1, 1 ), padding = 'same', use_bias = TRUE )
      H <- H %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )
      }

    # Scale down
    L0 <- H %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    L0 <- L0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

    # Scale up
    H0 <- L0 %>% layer_conv_2d_transpose( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    H0 <- H0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

    # Residual
    E <- layer_subtract( list( H0, H ) )

    # Scale residual down
    L1 <- E %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    L1 <- L1 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2 ) )

    # Output feature map
    downBlock <- layer_add( list( L0, L1 ) )

    return( downBlock )
    }

  inputs <- layer_input( shape = inputImageSize )

  # Initial feature extraction
  model <- inputs %>% layer_conv_2d( filters = numberOfFeatureFilters,
    kernel_size = c( 3, 3 ), strides = c( 1, 1 ), padding = 'same',
    kernel_initializer = "glorot_uniform" )
  model <- model %>% layer_activation_parametric_relu( alpha_initializer = 'zero',
    shared_axes = c( 1, 2 ) )

  # Feature smashing
  model <- model %>% layer_conv_2d( filters = numberOfBaseFilters,
    kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same',
    kernel_initializer = "glorot_uniform" )
  model <- model %>% layer_activation_parametric_relu( alpha_initializer = 'zero',
    shared_axes = c( 1, 2 ) )

  # Back projection
  upProjectionBlocks <- list()
  downProjectionBlocks <- list()

  model <- upBlock2D( model, numberOfFilters = numberOfBaseFilters,
    kernelSize = convolutionKernelSize, strides = strides )
  upProjectionBlocks[[1]] <- model

  for( i in seq_len( numberOfBackProjectionStages ) )
    {
    if( i == 1 )
      {
      model <- downBlock2D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides )
      downProjectionBlocks[[i]] <- model

      model <- upBlock2D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides )
      upProjectionBlocks[[i+1]] <- model
      model <- layer_concatenate( upProjectionBlocks )
      } else {
      model <- downBlock2D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides,
        includeDenseConvolutionLayer = TRUE, numberOfStages = i )
      downProjectionBlocks[[i]] <- model
      model <- layer_concatenate( downProjectionBlocks )

      model <- upBlock2D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides,
        includeDenseConvolutionLayer = TRUE, numberOfStages = i )
      upProjectionBlocks[[i+1]] <- model
      model <- layer_concatenate( upProjectionBlocks )
      }
    }
  # Final convolution layer
  outputs <- model %>% layer_conv_2d( filters = numberOfOutputs,
    kernel_size = c( 3, 3 ), strides = c( 1, 1 ), padding = 'same',
    kernel_initializer = "glorot_uniform" )

  deepBackProjectionNetworkModel <- keras_model( inputs = inputs, outputs = outputs )

  return( deepBackProjectionNetworkModel )
}

#' 3-D implementation of the deep back-projection network.
#'
#' Creates a keras model of the deep back-project network for image super
#' resolution.  More information is provided at the authors' website:
#'
#'         \url{https://www.toyota-ti.ac.jp/Lab/Denshi/iim/members/muhammad.haris/projects/DBPN.html}
#'
#' with the paper available here:
#'
#'         \url{https://arxiv.org/abs/1803.02735}
#'
#' This particular implementation was influenced by the following keras (python)
#' implementation:
#'
#'         \url{https://github.com/rajatkb/DBPN-Keras}
#'
#' with help from the original author's Caffe and Pytorch implementations:
#'
#'         \url{https://github.com/alterzero/DBPN-caffe}
#'         \url{https://github.com/alterzero/DBPN-Pytorch}
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).
#' @param numberOfOutputs number of outputs (e.g., 3 for RGB images).
#' @param numberOfFeatureFilters number of feature filters.
#' @param numberOfBaseFilters number of base filters.
#' @param numberOfBackProjectionStages number of up-/down-projection stages.  This
#' number includes the final up block.
#' @param convolutionKernelSize kernel size for certain convolutional layers.  This
#' and \code{strides} are dependent on the scale factor discussed in
#' original paper.  Factors used in the original implementation are as follows:
#' 2x --> \code{convolutionKernelSize = c( 2, 2, 2 )},
#' 4x --> \code{convolutionKernelSize = c( 8, 8, 8 )},
#' 8x --> \code{convolutionKernelSize = c( 12, 12, 12 )}.  We default to 8x parameters.
#' @param strides strides for certain convolutional layers.  This and the
#' \code{convolutionKernelSize} are dependent on the scale factor discussed in
#' original paper.  Factors used in the original implementation are as follows:
#' 2x --> \code{strides = c( 2, 2, 2 )}, 4x --> \code{strides = c( 4, 4, 4 )},
#' 8x --> \code{strides = c( 8, 8, 8 )}. We default to 8x parameters.
#'
#' @return a keras model defining the deep back-projection network.
#' @author Tustison NJ
#' @examples
#' #
#' \dontrun{
#'
#' }
#' @import keras
#' @export
createDeepBackProjectionNetworkModel3D <-
                    function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfBaseFilters = 64,
                               numberOfFeatureFilters = 256,
                               numberOfBackProjectionStages = 7,
                               convolutionKernelSize = c( 12, 12, 12 ),
                               strides = c( 8, 8, 8 )
                             )
{

  upBlock3D <- function( L, numberOfFilters = 64, kernelSize = c( 12, 12, 12 ),
    strides = c( 8, 8, 8 ), includeDenseConvolutionLayer = FALSE, numberOfStages = 1 )
    {
    if( includeDenseConvolutionLayer )
      {
      L <- L %>% layer_conv_3d( filters = numberOfFilters, # * numberOfStages,
        kernel_size = c( 1, 1, 1 ), stride = c( 1, 1, 1 ), padding = 'same',
        use_bias = TRUE )
      L <- L %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )
      }

    # Scale up
    H0 <- L %>% layer_conv_3d_transpose( filters = numberOfFilters,
      kernel_size = kernel_size, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    H0 <- H0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

    # Scale down
    L0 <- H0 %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    L0 <- L0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

    # Residual
    E <- layer_subtract( list( L0, L ) )

    # Scale residual up
    H1 <- E %>% layer_conv_3d_transpose( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    H1 <- H1 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

    # Output feature map
    upBlock <- layer_add( list( H0, H1 ) )

    return( upBlock )
    }

  downBlock3D <- function( H, numberOfFilters = 64, kernelSize = c( 12, 12, 12 ),
    strides = c( 8, 8, 8 ), includeDenseConvolutionLayer = FALSE, numberOfStages = 1 )
    {
    if( includeDenseConvolutionLayer )
      {
      H <- H %>% layer_conv_3d( filters = numberOfFilters, # * numberOfStages,
        kernel_size = c( 1, 1, 1 ), stride = c( 1, 1, 1 ), padding = 'same',
        use_bias = TRUE )
      H <- H %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )
      }

    # Scale down
    L0 <- H %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    L0 <- L0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

    # Scale up
    H0 <- L0 %>% layer_conv_3d_transpose( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    H0 <- H0 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

    # Residual
    E <- layer_subtract( list( H0, H ) )

    # Scale residual down
    L1 <- E %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides,
      kernel_initializer = 'glorot_uniform', padding = 'same' )
    L1 <- L1 %>% layer_activation_parametric_relu(
      alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

    # Output feature map
    downBlock <- layer_add( list( L0, L1 ) )

    return( downBlock )
    }

  inputs <- layer_input( shape = inputImageSize )

  # Initial feature extraction
  model <- inputs %>% layer_conv_3d( filters = numberOfFeatureFilters,
    kernel_size = c( 3, 3, 3 ), strides = c( 1, 1, 1 ), padding = 'same',
    kernel_initializer = "glorot_uniform" )
  model <- model %>% layer_activation_parametric_relu( alpha_initializer = 'zero',
    shared_axes = c( 1, 2 ) )

  # Feature smashing
  model <- model %>% layer_conv_3d( filters = numberOfBaseFilters,
    kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same',
    kernel_initializer = "glorot_uniform" )
  model <- model %>% layer_activation_parametric_relu( alpha_initializer = 'zero',
    shared_axes = c( 1, 2, 3 ) )

  # Back projection
  upProjectionBlocks <- list()
  downProjectionBlocks <- list()

  model <- upBlock3D( model, numberOfFilters = numberOfBaseFilters,
    kernelSize = convolutionKernelSize, strides = strides )
  upProjectionBlocks[[1]] <- model
  model <- layer_concatenate( upProjectionBlocks )

  for( i in seq_len( numberOfBackProjectionStages ) )
    {
    if( i == 1 )
      {
      model <- downBlock3D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides )
      downProjectionBlocks[[i]] <- model
      model <- layer_concatenate( downProjectionBlocks )

      model <- upBlock3D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides )
      upProjectionBlocks[[i+1]] <- model
      model <- layer_concatenate( upProjectionBlocks )
      } else {
      model <- downBlock3D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides,
        includeDenseConvolutionLayer = TRUE, numberOfStages = i )
      downProjectionBlocks[[i]] <- model
      model <- layer_concatenate( downProjectionBlocks )

      model <- upBlock3D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides,
        includeDenseConvolutionLayer = TRUE, numberOfStages = i )
      upProjectionBlocks[[i+1]] <- model
      model <- layer_concatenate( upProjectionBlocks )
      }
    }
  # Final convolution layer
  outputs <- model %>% layer_conv_3d( filters = numberOfOutputs,
    kernel_size = c( 3, 3, 3 ), strides = c( 1, 1, 1 ), padding = 'same',
    kernel_initializer = "glorot_uniform" )

  deepBackProjectionNetworkModel <- keras_model( inputs = inputs, outputs = outputs )

  return( deepBackProjectionNetworkModel )
}

