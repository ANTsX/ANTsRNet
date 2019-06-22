#' Unsupervised variational autoencoder training
#'
#' Trains a variational autoencoding with a convolutional network. This is
#' followed by k-means clustering to produce a segmentation and probabilities.
#'
#' @param patches input patch matrix, see \code{getNeighborhoodInMask}
#' @param k number of embedding layers
#' @param convControl optional named list with control parameters ( see code )
#' \itemize{
#' \item{hiddenAct}{ activation function for hidden layers eg relu}
#' \item{img_chns}{ eg 1 number of channels}
#' \item{filters}{ eg 32L}
#' \item{conv_kern_sz}{ eg 1L}
#' \item{front_kernel_size}{ eg 2L}
#' \item{intermediate_dim}{ eg 32L}
#' \item{epochs}{ eg 50}
#' \item{batch_size}{ eg 32}
#' \item{squashAct}{ activation function for squash layers eg sigmoid}
#' \item{tensorboardLogDirectory}{ tensorboard logs stored here }
#' }
#' @param standardize boolean controlling whether patches are standardized
#' @param patches2 input target patch matrix, see \code{getNeighborhoodInMask},
#' may be useful for super-resolution
#' @return model is output
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#'
#' library(ANTsR)
#' img <- ri( 1 ) %>% resampleImage( 4 ) %>% iMath( "Normalize" )
#' mask = randomMask( getMask( img ), 50 )
#' r = c( 3, 3 )
#' patch = getNeighborhoodInMask( img, mask, r, boundary.condition = "NA" )
#' uvaSegModel = uvaSegTrain( patch, 6 )
#' }
#'
#' @export uvaSegTrain
uvaSegTrain <- function( patches,
  k,
  convControl,
  standardize = TRUE,
  patches2 ) {

  ##############################################################################
  # unsupervised segmentation - variational aec
  ##############################################################################
  p <- as.integer( sqrt( nrow( patches ) ) ) # patchSize
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
  intermediate_dim <- 512L
  epochs = 2
  epsilon_std <- 1.0
  batch_size = 32
  squashAct = "sigmoid"
  tensorboardLogDirectory = NA
  if ( ! missing( "convControl" ) ) {
    hiddenAct = convControl$hiddenAct
    img_chns <- as.integer( convControl$img_chns ) # FIXME should generalize
    filters <- as.integer( convControl$filters )
    conv_kern_sz <- as.integer( convControl$conv_kern_sz )
    front_kernel_size = convControl$front_kernel_size
    intermediate_dim <- as.integer( convControl$intermediate_dim )
    epochs = convControl$epochs
    batch_size = convControl$batch_size
    squashAct = convControl$squashAct
    tensorboardLogDirectory = convControl$tensorboardLogDirectory
  }


  #### Data Preparation ####
  nona <-function( x ) {
    x[ is.na( x ) ] = 0
    return( x )
  }
  x_train = array( dim =  c( ncol( patches ), p, p, img_chns   ) )
  for ( i in 1:ncol( patches ) ) {
    locp = nona( patches[ , i ] )
    if ( standardize ) {
      mymu = mean( locp, na.rm=T )
      mysd = sd( locp, na.rm=T )
      if ( mysd == 0 ) mysd = 1
      x_train[ i, , , 1] = ( locp - mymu ) / mysd
    } else x_train[ i, , , 1] = locp
    }

  if ( ! missing( patches2 ) ) {
    x_train2 = array( dim =  c( ncol( patches2 ), p, p, img_chns   ) )
    for ( i in 1:ncol( patches2 ) ) {
      locp = nona( patches2[ , i ] )
      if ( standardize ) {
        mymu = mean( locp, na.rm=T )
        mysd = sd( locp, na.rm=T )
        if ( mysd == 0 ) mysd = 1
        x_train2[ i, , , 1] = ( locp - mymu ) / mysd
      } else x_train2[ i, , , 1] = locp
      }
  } else x_train2 = x_train

  # training parameters
  batch_size <- min( c( batch_size, round( length( patches )/2 ) ) )
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
    activation = squashAct
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
#  vae %>% compile( optimizer = optimizer_adam( lr = 0.0001 ), loss = vae_loss )
  vae %>% compile( optimizer = "rmsprop", loss = vae_loss )

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


  #### Model Fitting ####
  if ( ! is.na(  tensorboardLogDirectory  ) ) {
    keras::tensorboard( tensorboardLogDirectory, launch_browser = FALSE )
    vae %>% fit(
      x_train, x_train2,
      shuffle = TRUE,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = callback_tensorboard( tensorboardLogDirectory )
    )
  }

  if ( is.na(  tensorboardLogDirectory  ) )
    vae %>% fit(
      x_train, x_train2, shuffle = TRUE, epochs = epochs, batch_size = batch_size
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

  return( list( encoder = encoder, generator = generator, vae = vae ) )

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
#' @param returnProbabilities boolean
#' @param batchSize for the prediction
#' @param standardize boolean controlling whether patches are standardized
#' @param verbose boolean
#' @return segmentation and probability images are output
#' @author Avants BB
#' @examples
#' \dontrun{
#' library(ANTsR)
#' img <- ri( 1 ) %>% resampleImage( 4 ) %>% iMath( "Normalize" )
#' mask = randomMask( getMask( img ), 50 )
#' patch = getNeighborhoodInMask( img, mask, c(3,3), boundary.condition = "NA" )
#' uvaSegModel = uvaSegTrain( patch, 6 )
#' tarImg = ri( 3 ) %>% resampleImage( 4 )
#' uvaSegmentation = uvaSeg(tarImg, uvaSegModel, k = 3, getMask( tarImg ) )
#' }
#' @export uvaSeg
#' @importFrom Rcpp cppFunction
#' @importFrom utils download.file
#' @importFrom ANTsRCore antsGetDirection antsGetOrigin resampleImage labelStats
uvaSeg <- function(
  image,
  model,
  k,
  mask,
  returnProbabilities = FALSE,
  batchSize = 1028,
  standardize = TRUE,
  verbose = FALSE )
{
  normimg <-function( img ) {
    iMath( img, "Normalize" )
    }

  Rcpp::cppFunction("
#include <Rcpp.h>
#include <math.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix fuzzyClustering(NumericMatrix data, NumericMatrix centers, int m) {
  int c=centers.rows();
  int rows = data.rows();
  int cols = data.cols();
  double tempDist=0;
  double dist=0;
  double denominator=0;

  NumericMatrix result(rows,c);

  for(int i=0;i<rows;i++){
    for(int j=0;j<c;j++){
      for(int k=0;k<c;k++){
        for(int p=0;p<cols;p++){
          tempDist = tempDist+pow(centers(j,p)-data(i,p),2);
          dist = dist + pow(centers(k,p)-data(i,p),2);
        }
        tempDist = sqrt(tempDist);
        dist = sqrt(dist);
        denominator = denominator+pow((tempDist/dist),(2/(m-1)));
        tempDist = 0;
        dist = 0;
      }
      result(i,j) = 1/denominator;
      denominator = 0;
    }
  }
  return result;
} ")
  img_chns = 1L
  p = as.integer( model$encoder$input_shape[[ 2 ]] )
  patches = ANTsRCore::getNeighborhoodInMask(
    image = image,
    mask = mask,
    radius = rep( floor( p / 2 ), image@dimension ),
    boundary.condition = "NA" )
  img_chns = 1L
  x_test = array(
    dim =  c( ncol( patches ), p, p, img_chns   ) )
  nona <-function( x ) {
      x[ is.na( x ) ] = 0
      return( x )
    }
  for ( i in 1:ncol( patches ) ) {
    locp = nona( patches[ , i ] )
    if ( standardize ) {
      mymu = mean( locp, na.rm=T )
      mysd = sd( locp, na.rm=T )
      if ( mysd == 0 ) mysd = 1
      x_test[ i, , , 1] = ( locp - mymu ) / mysd
    } else x_test[ i, , , 1] = locp
    }
  rm( patches )
  invisible( gc() )
  ###################################################################
  if ( verbose ) print("begin predict")
  x_test_encoded <- predict( model$encoder, x_test, batch_size = batchSize )
  if ( verbose ) print("begin patch prediction")
  barker = NA
#  barker <- predict( model$vae, x_test, batch_size = batchSize )
  wkmalg = "Forgy"
  if ( verbose ) print("begin km")
  kmAlg = stats::kmeans( x_test_encoded, k, iter.max=30 )
  mykm = kmAlg$cluster
  segmentation = makeImage( mask, mykm )
  centerimgmeans = labelStats( image, segmentation )$Mean[-1]
  kmAlg = stats::kmeans( x_test_encoded,
    kmAlg$centers[ order( centerimgmeans ),  ], iter.max=30 )
  mykm = kmAlg$cluster
  segmentation = makeImage( mask, mykm )
  probs = list()
  mid = round( p / 2 )
  intensityvec = rep( NA, sum( mask == 1 ) )
#  for ( i in 1:dim( barker )[1] ) intensityvec[ i ] = barker[i,mid,mid,1]
  if ( returnProbabilities ) {
    if ( verbose ) print("begin probs")
    myprobs = fuzzyClustering( x_test_encoded, kmAlg$centers, m=2 )
    for ( k in 1:ncol( myprobs ) )
      probs[[ k ]] = makeImage( mask, myprobs[ , k ] )
    if ( verbose ) print("end probs")
  }
  return(
    list(
      segmentation  = segmentation,
      probabilities = probs,
      embedding     = x_test_encoded,
      x_test        = x_test,
      predictedPatches = barker,
      centers      = kmAlg$centers,
      intensityvec = intensityvec  )
      )
}
