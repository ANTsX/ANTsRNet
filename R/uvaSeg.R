#' Unsupervised variational autoencoder training
#'
#' Trains a variational autoencoding with a convolutional network. This is
#' followed by k-means clustering to produce a segmentation and probabilities.
#'
#' @param patches input patch list, see \code{extractImagePatches}
#' @param k number of embedding layers
#' @param convControl optional named list with control parameters ( see code )
#' @return model is output
#' @author Avants BB
#' @examples
#' library(ANTsR)
#' patch <- ri( 1 ) %>% resampleImage( 4 ) %>% iMath( "Normalize" ) %>%
#'   extractImagePatches( c(3,3), maxNumberOfPatches = 50 )
#' mdl = uvaSegTrain( patch, 3 )
#'
#' @export uvaSegTrain
#### @useDynLib ANTsRNet
uvaSegTrain <- function( patches, k, convControl ) {

  ##############################################################################
  # unsupervised segmentation - variational aec
  ##############################################################################
  p <- as.integer( dim( patches[[1]] )[1] ) # patchSize
  #### Parameterization ####
  act1 = 'relu'
  hiddenAct = 'relu'
  img_chns <- 1L # FIXME should generalize
  # number of convolutional filters to use
  filters <- 32L
  # convolution kernel size
  conv_kern_sz <- 1L
  front_kernel_size = c(2L, 2L)
  latent_dim <- as.integer( k )
  intermediate_dim <- 64L
  epsilon_std <- 1.0
  epochs = 50
  # training parameters
  batch_size <- min( c( 100L, round( length( patches )/2 ) ) )
  output_shape <- c( batch_size, p, p, filters )

  #### Model Construction ####
  original_img_size <- c( p, p, img_chns)
  x <- layer_input(shape = c(original_img_size))
  conv_1 <- layer_conv_2d(
    x,
    filters = img_chns,
    kernel_size = front_kernel_size,
    strides = c(1L, 1L),
    padding = "same",
    activation = act1
  )

  conv_2 <- layer_conv_2d(
    conv_1,
    filters = filters,
    kernel_size = front_kernel_size,
    strides = c(2L, 2L),
    padding = "same",
    activation = act1
  )

  conv_3 <- layer_conv_2d(
    conv_2,
    filters = filters,
    kernel_size = c(conv_kern_sz, conv_kern_sz),
    strides = c(1L, 1L),
    padding = "same",
    activation = act1
  )

  conv_4 <- layer_conv_2d(
    conv_3,
    filters = filters,
    kernel_size = c(conv_kern_sz, conv_kern_sz),
    strides = c(1L, 1L),
    padding = "same",
    activation = act1
  )

  flat <- layer_flatten(conv_4)
  hidden <- layer_dense(flat, units = intermediate_dim, activation = hiddenAct )

  z_mean <- layer_dense(hidden, units = latent_dim)
  z_log_var <- layer_dense(hidden, units = latent_dim)

  sampling <- function(args) {
    z_mean <- args[, 1:(latent_dim)]
    z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]

    epsilon <- k_random_normal(
      shape = c(k_shape(z_mean)[[1]]),
      mean = 0.,
      stddev = epsilon_std
    )
    z_mean + k_exp(z_log_var) * epsilon
  }

  z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

  decoder_hidden <- layer_dense(units = intermediate_dim, activation = act1)
  decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = act1)

  decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
  decoder_deconv_1 <- layer_conv_2d_transpose(
    filters = filters,
    kernel_size = c(conv_kern_sz, conv_kern_sz),
    strides = c(1L, 1L),
    padding = "same",
    activation = act1
  )

  decoder_deconv_2 <- layer_conv_2d_transpose(
    filters = filters,
    kernel_size = c(conv_kern_sz, conv_kern_sz),
    strides = c(1L, 1L),
    padding = "same",
    activation = act1
  )


  decoder_deconv_3_upsample <- layer_conv_2d_transpose(
    filters = filters,
    kernel_size = c(3L, 3L),
    strides = c(1L, 1L),
    padding = "same",
    activation = act1
  )

  decoder_mean_squash <- layer_conv_2d(
    filters = 1,
    kernel_size = c(1L, 1L),
    strides = c(1L, 1L),
    padding = "same",
    activation = "sigmoid"
  )

  hidden_decoded <- decoder_hidden(z)
  up_decoded <- decoder_upsample(hidden_decoded)
  reshape_decoded <- decoder_reshape(up_decoded)
  deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
  deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
  x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
  x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)

  # custom loss function
  vae_loss <- function(x, x_decoded_mean_squash) {
    x <- k_flatten(x)
    x_decoded_mean_squash <- k_flatten( x_decoded_mean_squash )
    xent_loss <- 1.0 * p * p *
      loss_binary_crossentropy(x, x_decoded_mean_squash)
    kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                             k_exp(z_log_var), axis = -1L)
    k_mean(xent_loss + kl_loss)
  }

  ## variational autoencoder
  vae <- keras_model(x, x_decoded_mean_squash)
  vae %>% compile( optimizer = optimizer_adam( lr = 0.0001 ), loss = vae_loss )

  ## build a generator that can sample from the learned distribution
  ## this reuses the layers trained above
  gen_decoder_input <- layer_input(shape = latent_dim)
  gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
  gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
  gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
  gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
  gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
  gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
  gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
  generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)


  #### Data Preparation ####
  mytrn = length( patches )
  x_train = array( dim =  c( mytrn, p, p, img_chns   ) )
  for ( i in 1:mytrn ) {
    x_train[ i, , , 1 ] = patches[[i]]
    }

  #### Model Fitting ####
  vae %>% fit(
    x_train, x_train,
    shuffle = TRUE,
    epochs = epochs,
    batch_size = batch_size
  )

  ## encoder: model to project inputs on the latent space
  encoder <- keras_model( x, z_mean )

  ## build a generator that can sample from the learned distribution
  gen_decoder_input <- layer_input(shape = latent_dim)
  gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
  gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
  gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
  gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
  gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
  gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
  gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
  generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)

  return( list( encoder = encoder, generator = generator ) )

}



#' Unsupervised variational autoencoder segmentation
#'
#' Trains a variational autoencoding with a convolutional network. This is
#' followed by k-means clustering to produce a segmentation and probabilities.
#'
#' @param image input image
#' @param model the model output from \code{uvaSegTrain}
#' @param k number of clusters or cluster centers
#' @param mask defining output segmentation space
#' @return segmentation and probability images are output
#' @author Avants BB
#' @examples
#'
#' library(ANTsR)
#' patch <- ri( 1 ) %>% resampleImage( 4 ) %>% iMath( "Normalize" ) %>%
#'   extractImagePatches( c(3,3), maxNumberOfPatches = 50 )
#' uvaSegModel = uvaSegTrain( patch, 6 )
#' tarImg = ri( 3 ) %>% resampleImage( 4 )
#' uvaSegmentation = uvaSeg(tarImg, uvaSegModel, k = 3, getMask( tarImg ) )
#'
#' @export uvaSeg
#' @importFrom Rcpp cppFunction
uvaSeg <- function( image, model, k,  mask ) {
  normimg <-function( img ) {
    iMath( img, "Normalize" )
    }

  Rcpp::cppFunction("
  #include <Rcpp.h>
#include <math.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix fuzzyClustering(NumericMatrix data, NumericMatrix centers, int m) {
/* data is a matrix with observations(rows) and variables,
   centers is a matrix with cluster centers coordinates,
   m is a parameter of equation, c is a number of clusters
*/

  int c=centers.rows();
  int rows = data.rows();
  int cols = data.cols(); /*number of columns equals number of variables, the same as is in centers matrix*/
  double tempDist=0;        /*dist and tempDist are variables storing temporary euclidean distances */
  double dist=0;
  double denominator=0;    //denominator of main equation

  NumericMatrix result(rows,c);    //declaration of matrix of results

  for(int i=0;i<rows;i++){
    for(int j=0;j<c;j++){
      for(int k=0;k<c;k++){
        for(int p=0;p<cols;p++){
          tempDist = tempDist+pow(centers(j,p)-data(i,p),2);
         //in innermost loop an euclidean distance is calculated.
          dist = dist + pow(centers(k,p)-data(i,p),2);
/*tempDist is nominator inside the sum operator in the equation, dist is the denominator inside the sum operator in the equation*/
        }
        tempDist = sqrt(tempDist);
        dist = sqrt(dist);
        denominator = denominator+pow((tempDist/dist),(2/(m-1)));
        tempDist = 0;
        dist = 0;
      }
      result(i,j) = 1/denominator;
// nominator/denominator in the  main equation
      denominator = 0;
    }
  }
  return result;
} ")
  p = as.integer( model$encoder$input_shape[[ 2 ]] )
  patches = extractImagePatches( normimg( image ),
    c( p, p ), maxNumberOfPatches = 'all',
    strideLength = 1 )
  img_chns = 1L
  x_test = array( dim =  c( length( patches ), p, p, img_chns   ) )
  for ( i in 1:length( patches ) ) {
    x_test[ i, , , 1 ] = patches[[i]]
    }
  ###################################################################
  x_test_encoded <- predict( model$encoder, x_test, batch_size = 1000 )
  kmAlg = stats::kmeans( x_test_encoded, k )
  mykm = kmAlg$cluster
  myprobs = fuzzyClustering( x_test_encoded, kmAlg$centers, m=2 )
  probs = list()
  mid = round( p / 2 )
  for ( myk in 1:ncol( myprobs ) ) {
    for ( i in 1:length( patches ) ) {
      temp = array( 0, dim=c(p,p) )
      wm = myprobs[i,myk]
      temp[mid,mid] = wm
      patches[[i]][,] = temp
      }
    if ( ! missing( mask ) )
      imageReconstructed <- reconstructImageFromPatches(
        patches, mask, strideLength=1 )
      else imageReconstructed <- reconstructImageFromPatches(
          patches, image, strideLength=1 )
    probs[[ myk ]] = imageReconstructed
    }

# segmentation image
  for ( i in 1:length( patches ) ) {
    temp = array( 0, dim=c(p,p) )
    wm = mykm[i]
    temp[mid,mid] = wm
    patches[[i]][,] = temp
    }
  if ( ! missing( mask ) )
    imageReconstructed <- reconstructImageFromPatches(
      patches, mask, strideLength=1 )
    else imageReconstructed <- reconstructImageFromPatches(
        patches, image, strideLength=1 )

  return(
    list(
      segmentation = imageReconstructed,
      probabilities = probs  )
      )
}
