#' linMatchIntensity
#'
#' regression between two image intensity spaces
#'
#' @param fromImg image whose intensity function we will match to the \code{toImg}
#' @param toImg defines the reference intensity function.
#' @param polyOrder of polynomial fit.  default is none or just linear fit.
#' @param truncate boolean which turns on/off the clipping of intensity.
#' @param mask mask the matching region
#' @return the \code{fromImg} matched to the \code{toImg}
#' @author Avants BB
#' @examples
#' library(ANTsRCore)
#' sourceImage <- antsImageRead( getANTsRData( "r16" ) )
#' referenceImage <- antsImageRead( getANTsRData( "r64" ) )
#' matchedImage <- linMatchIntensity( sourceImage, referenceImage )
#' @export
linMatchIntensity <- function( fromImg, toImg, polyOrder = 1, truncate = TRUE, mask ) {
  if ( missing( mask ) ) {
    tovec = as.numeric( toImg )
    fromvec = as.numeric( fromImg )
    mdl = lm( tovec  ~  stats::poly( fromvec, polyOrder ) )
    pp = predict( mdl )
  } else {
    tovec = as.numeric( toImg[ mask >= 0.5 ] )
    fromvec = as.numeric( fromImg[ mask >= 0.5 ] )
    mydf = data.frame( tovec = tovec, stats::poly( fromvec, polyOrder ) )
    mdl = lm( tovec  ~  . , data=mydf)
    fromvec = as.numeric( fromImg )
    mydfnew = data.frame( stats::poly( fromvec, polyOrder ) )
    pp = predict( mdl, newdata=mydfnew )
  }
  if ( truncate ) {
    pp[ pp < min( toImg ) ] = min( toImg )
    pp[ pp > max( toImg ) ] = max( toImg )
  }
  newImg = makeImage( dim( fromImg ), pp )
  temp = antsCopyImageInfo( fromImg,  newImg )
  return( newImg )
}

#' applySuperResolutionModel
#'
#' Apply pretrained super-resolution network
#'
#' Apply a patch-wise trained network to perform super-resolution. Can be applied
#' to variable sized inputs. Warning: This function may be better used on CPU
#' unless the GPU can accommodate the full image size. Warning 2: The global
#' intensity range (min to max) of the output will match the input where the
#' range is taken over all channels.
#'
#' @param image input image
#' @param model model object or filename see \code{getPretrainedNetwork}
#' @param targetRange a vector defining min max of each the input image,
#' eg -127.5, 127.5.  Output images will be scaled back to original intensity.
#' This range should match the mapping used in the training of the network.
#' @param batch_size for prediction call
#' @param linmatchOrder if not missing, then apply \code{linMatchIntensity} with given fit parameter
#' @param mask restrict intensity rescaling parameters within the mask
#' @param verbose If \code{TRUE}, show status messages
#' @return image upscaled to resolution provided by network
#' @author Avants BB
#' @examples
#' \donttest{
#' library(ANTsRCore)
#' library(keras)
#' orig_img = antsImageRead( getANTsRData( "r16" ) )
#' # input needs to be 48x48
#' img = resampleImage(orig_img, resampleParams = rep(256/48, 2))
#' model = getPretrainedNetwork( "dbpn4x" )
#' simg <- applySuperResolutionModel(img,  model = model)
#' plot(orig_img)
#' plot(img)
#' plot(simg)
#' }
#' @importFrom ANTsRCore antsCopyImageInfo antsSetSpacing
#' @importFrom ANTsRCore splitChannels antsAverageImages
#' @importFrom ANTsRCore antsTransformIndexToPhysicalPoint
#' @importFrom ANTsRCore antsTransformIndexToPhysicalPoint
#' @export applySuperResolutionModel
applySuperResolutionModel <- function(
  image,
  model,
  targetRange,
  batch_size = 32,
  linmatchOrder,
  mask,
  verbose = FALSE )
{
  if ( ! missing( targetRange ) )
    if ( targetRange[1] > targetRange[2] )
      targetRange = rev( targetRange )

    if ( verbose ) print( "1. load model" )
    tl1 = Sys.time()
    if ( ! is.object( model )  )
      if ( is.character( model ) ) {
        if ( file.exists( model ) ) {
          model = load_model_hdf5( model )
        } else stop("Model not found")
      }
    if ( verbose ) print( paste("load model in : ", Sys.time() - tl1 ) )
    shapeLength = length( model$input_shape )
    if ( shapeLength == 5 ) { # image dimension is 3
      if ( image@dimension != 3 ) stop("Expecting 3D input for this model")
      channelSize = model$input_shape[[5]]
      channelSizeOut = model$output_shape[[5]]
    }
    if ( shapeLength == 4 ) { # image dimension is 2
      if ( image@dimension != 2 ) stop("Expecting 2D input for this model")
      channelSize = model$input_shape[[4]]
      channelSizeOut = model$output_shape[[4]]
    }
    # check image components matches nchannels, otherwise replicate the input
    ncomponents = image@components
    if ( channelSize != ncomponents ) {
      stop( paste(
        "ChannelSize of model", channelSize,
        'does not match ncomponents', ncomponents, 'of image') )
    }
    ###############
    X_test <- extractImagePatches(
        image,
        dim( image ), maxNumberOfPatches = 1, returnAsArray = TRUE )
    X_test = array( X_test, dim = c( 1,  dim( image ), image@components ) )
    if ( ! missing( targetRange ) & missing( mask )  ) {
      X_test = X_test - min( X_test )
      X_test = X_test / max( X_test ) * ( targetRange[2] - targetRange[1] ) + targetRange[1]
      }
    if ( ! missing( targetRange ) & ! missing( mask )  ) {
      selector = mask >= 0.5
      minval = min( image[ selector ] )
      X_test = X_test - minval
      maxval = max( image[ selector ] - minval )
      X_test = X_test / maxval * ( targetRange[2] - targetRange[1] ) + targetRange[1]
      }
    #################################################
    if ( verbose ) print( "##### prediction" )
    t1 = Sys.time()
    pred = predict( model, X_test, batch_size = batch_size )
    expansionFactor = ( dim( pred ) / dim( X_test ) )[-1][1:image@dimension]
    if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
    # below is a fast simple linear regression model to map patch intensities
    if ( verbose ) print( "4. reconstruct intensity" )
    if ( ! missing( targetRange ) & missing( mask )  ) {
      temp = range( image )
      pred = pred - min( pred )
      pred = pred / max( pred ) * ( temp[2] - temp[1] ) + temp[1]
    }

    sliceArray <- function(  myArr, j ) {
      if ( shapeLength == 3 ) {
        return( myArr[j,,] )
      }
      if ( shapeLength == 4 ) {
        return( myArr[j,,,] )
      }
      if ( shapeLength == 5 ) {
        return( myArr[j,,,,] )
      }
    }

    sliceArrayChannel <- function(  myArr, j ) {
      if ( shapeLength == 3 ) {
        return( myArr[,,j] )
      }
      if ( shapeLength == 4 ) {
        return( myArr[,,,j] )
      }
      if ( shapeLength == 5 ) {
        return( myArr[,,,,j] )
      }
    }


    if ( verbose )
      print( paste( "expansionFactor: ", paste( expansionFactor, collapse= 'x' ) ) )

    if ( tail(dim(pred),1) == 1 ) {
      ivec = sliceArrayChannel( pred, 1 )
      predImg = makeImage( dim( image ) * expansionFactor, ivec )
      if ( ! missing( targetRange ) & ! missing( mask )  ) {
        selector = mask >= 0.5
        selectorBig = resampleImageToTarget( mask, predImg,  interpType = "nearestNeighbor" ) >= 0.5
        temp = range( image[ selector ] )
        minval = min( predImg[ selectorBig ] )
        predImg = predImg - minval
        maxval = max( predImg[ selectorBig ] )
        predImg = predImg / maxval * ( temp[2] - temp[1] ) + temp[1]
        }
      if ( ! missing( linmatchOrder ) ) {
        bilin = resampleImageToTarget( image, predImg )
        if ( missing( mask ) )
          predImg = linMatchIntensity( predImg, bilin, polyOrder = linmatchOrder  )
        if ( ! missing( mask ) ) {
          bigMask = resampleImageToTarget( mask, predImg,  interpType = "nearestNeighbor" )
          predImg = linMatchIntensity( predImg, bilin, polyOrder = linmatchOrder, mask = bigMask  )
          }
        }
    }
    if ( tail(dim(pred),1) > 1 ) {
      mcList = list()
      for ( k in 1:tail(dim(pred),1) ) {
        ivec = sliceArrayChannel( pred, k )
        mcList[[k]] = makeImage( dim( image ) * expansionFactor, ivec )
      }
      predImg = mergeChannels( mcList )
    }
    predImg = antsCopyImageInfo( image, predImg )
    antsSetSpacing( predImg, antsGetSpacing( image ) / expansionFactor )
    return( predImg )
}





#' applySuperResolutionModelPatch
#'
#' Apply pretrained super-resolution network by stitching together patches.
#'
#' Apply a patch-wise trained network to perform super-resolution. Can be applied
#' to variable sized inputs. Warning: This function may be better used on CPU
#' unless the GPU can accommodate the full patch size. Warning 2: The global
#' intensity range (min to max) of the output will match the input where the
#' range is taken over all channels.
#'
#' @param image input image
#' @param model model object or filename see \code{getPretrainedNetwork}
#' @param targetRange a vector defining min max of each the input image,
#' eg -127.5, 127.5.  Output images will be scaled back to original intensity.
#' This range should match the mapping used in the training of the network.
#' @param lowResolutionPatchSize size of patches to upsample
#' @param strideLength voxel/pixel steps between patches
#' @param batch_size for prediction call
#' @param mask restrict intensity rescaling parameters within the mask
#' @param verbose If \code{TRUE}, show status messages
#' @return image upscaled to resolution provided by network
#' @author Avants BB
#' @examples
#' \dontrun{
#' library(ANTsRCore)
#' library( keras )
#' orig_img = antsImageRead( getANTsRData( "r16" ) )
#' # input needs to be 48x48
#' model = createDeepBackProjectionNetworkModel2D( list(NULL,NULL, 1) )
#' img = resampleImage(orig_img, resampleParams = rep(256/48, 2))
#' simg <- applySuperResolutionModelPatch( img,
#'  model = model, lowResolutionPatchSize = 8, strideLength = 2)
#' simgm <- applySuperResolutionModelPatch( img, mask = getMask( img ),
#'  model = model, lowResolutionPatchSize = 8, strideLength = 2)
#' plot( orig_img )
#' plot( img )
#' plot( simg )
#' plot( simgm )
#' }
#' @importFrom stats cov var lm runif sd
#' @export applySuperResolutionModelPatch
applySuperResolutionModelPatch <- function(
  image,
  model,
  targetRange,
  lowResolutionPatchSize = 128,
  strideLength = 16,
  batch_size = 32,
  mask,
  verbose = FALSE )
{
  if ( ! missing( targetRange ) )
    if ( targetRange[1] > targetRange[2] )
      targetRange = rev( targetRange )

    if ( verbose ) print( "1. load model" )
    tl1 = Sys.time()
    if ( ! is.object( model )  )
      if ( is.character( model ) ) {
        if ( file.exists( model ) ) {
          model = load_model_hdf5( model )
        } else stop("Model not found")
      }
    if ( verbose ) print( paste("load model in : ", Sys.time() - tl1 ) )
    shapeLength = length( model$input_shape )
    if ( shapeLength == 5 ) { # image dimension is 3
      if ( image@dimension != 3 ) stop("Expecting 3D input for this model")
      channelSize = model$input_shape[[5]]
      channelSizeOut = model$output_shape[[5]]
    }
    if ( shapeLength == 4 ) { # image dimension is 2
      if ( image@dimension != 2 ) stop("Expecting 2D input for this model")
      channelSize = model$input_shape[[4]]
      channelSizeOut = model$output_shape[[4]]
    }
    # check image components matches nchannels, otherwise replicate the input
    ncomponents = image@components
    if ( channelSize != ncomponents ) {
      stop( paste(
        "ChannelSize of model", channelSize,
        'does not match ncomponents', ncomponents, 'of image') )
    }

    sliceArray <- function(  myArr, j ) {
      if ( shapeLength == 3 ) {
        return( myArr[j,,] )
      }
      if ( shapeLength == 4 ) {
        return( myArr[j,,,] )
      }
      if ( shapeLength == 5 ) {
        return( myArr[j,,,,] )
      }
    }
    ###############
    if ( length( lowResolutionPatchSize ) == 1 )
      lowResolutionPatchSize = rep( lowResolutionPatchSize, image@dimension )
    if ( length( strideLength ) == 1 )
      strideLength = rep( strideLength, image@dimension )
    X_test <- extractImagePatches(
      image,
      lowResolutionPatchSize,
      maxNumberOfPatches = 'all',
      strideLength = strideLength, returnAsArray = TRUE )
    numberOfPatches = nrow( X_test )
    X_test = array( X_test,
                    dim = c( numberOfPatches,  lowResolutionPatchSize, channelSize ) )
    if ( ! missing( targetRange ) & missing( mask ) ) {
      xvec = as.numeric( sliceArray( X_test, 1 ) )
      tempMat = matrix( nrow = numberOfPatches, ncol = length( xvec ) )
      for( j in 1:numberOfPatches )
      {
        temp = as.numeric( sliceArray( X_test, j ) )
        temp = temp - min( temp )
        temp = temp / max( temp ) * ( targetRange[2] - targetRange[1] ) + targetRange[1]
        tempMat[j,] = temp
      }
      X_test <- array( data = tempMat,
                       dim = c( numberOfPatches, lowResolutionPatchSize, channelSize ) )
      rm( tempMat )
      gc()
    }
    if ( ! missing( targetRange ) & ! missing( mask ) ) {
      selector = mask >= 0.5
      minval = min( image[ selector ] )
      X_test = X_test - minval
      maxval = max( image[ selector ] - minval )
      X_test = X_test / maxval * ( targetRange[2] - targetRange[1] ) + targetRange[1]
      }
    #################################################
    if ( verbose ) print( "##### prediction" )
    t1 = Sys.time()
    pred = predict( model, X_test, batch_size = batch_size )
    if ( verbose ) print( paste( "     - Predict in:", Sys.time()-t1 ) )
    # below is a fast simple linear regression model to map patch intensities
    if ( verbose ) print( "4. reconstruct intensity" )

    expansionFactor = ( dim( pred ) / dim( X_test ) )[-1][1:image@dimension]
    if ( verbose )
      print( paste( "expansionFactor: ", paste( expansionFactor, collapse= 'x' ) ) )

    bigStrides = strideLength * expansionFactor
    bigImg = resampleImage( image,
                            dim( image ) * expansionFactor, useVoxels = T )
    if ( channelSizeOut != bigImg@components ) {
      if ( bigImg@components > 1 )
        bigImgSplit = splitChannels( bigImg ) else bigImgSplit=list( bigImg )
        bigavg = antsAverageImages( bigImgSplit )
        blist = list()
        for ( k in 1:channelSizeOut ) {
          blist[[k]] = bigavg
        }
        bigImg = mergeChannels( blist )
    }

    highResolutionPatchSize = lowResolutionPatchSize * expansionFactor
    Y_test <- extractImagePatches(
      bigImg,
      highResolutionPatchSize, maxNumberOfPatches = 'all',
      strideLength = bigStrides, returnAsArray = FALSE )

    # below is a fast simple linear regression model to map patch intensities
    # back to the original space defined by the upsampled image
    if ( verbose ) print( "4. reconstruct intensity" )
    plist = list()
    for( j in seq_len( numberOfPatches ) ) {
      temp = sliceArray( pred, j )
      ivec1 = as.numeric( temp )
      ivec2 = as.numeric( Y_test[[j]])
      mybeta = cov(ivec1,ivec2)/var(ivec1)
      ivec1t = mybeta * ivec1
      ivec1t = ivec1t + mean( ivec2 ) - mean( ivec1t )
      plist[[j]] = array( ivec1t, dim = dim( temp ) )
    }
    if ( verbose ) print( "5. reconstruct full image" )
    predImg = reconstructImageFromPatches( plist,
                                           bigImg,   strideLength = bigStrides )
    antsSetSpacing( predImg, antsGetSpacing( image ) / expansionFactor )

    if ( ! missing( targetRange ) & ! missing( mask )  ) {
      selector = mask >= 0.5
      if ( predImg@components == 1 )
        selectorBig = resampleImageToTarget( mask, predImg,  interpType = "nearestNeighbor" ) >= 0.5
      if ( predImg@components > 1 )
          selectorBig = resampleImageToTarget( mask, splitChannels(predImg)[[1]],  interpType = "nearestNeighbor" ) >= 0.5
      temp = range( image[ selector ] )
      minval = min( predImg[ selectorBig ] )
      predImg = predImg - minval
      maxval = max( predImg[ selectorBig ] )
      predImg = predImg / maxval * ( temp[2] - temp[1] ) + temp[1]
      }

    return( predImg )
}
