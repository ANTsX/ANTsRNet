#' Spatial transformer layer (2-D)
#'
#' @docType class
#'
#'
#' @section Arguments:
#' \describe{
#'  \item{inputs}{list of size 2 where the first element are the images. The second
#'                element are the weights.}
#'  \item{resampledSize}{size of the resampled output images.}
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
#' @return resampled batch images.
#'
#' @name SpatialTransformerLayer2D
NULL

#' @export
SpatialTransformerLayer2D <- R6::R6Class("SpatialTransformerLayer2D",
  inherit = KerasLayer,

  public = list(
    resampledSize = NULL,

    transformType = "affine",

    interpolatorType = "linear",

    initialize = function(resampledSize, transformType = "affine", interpolatorType = "linear") {
      K <- keras::backend()
      if (K$backend() != "tensorflow") {
        stop("Error:  tensorflow is required for this STN implementations.")
      }
      if (length(resampledSize) != 2) {
        stop("Error:  resampled size must be a vector of length 2 (for 2-D).")
      }

      self$resampledSize <- resampledSize
      self$transformType <- tolower(transformType)
      self$interpolatorType <- tolower(interpolatorType)
    },

    call = function(inputs, mask = NULL) {
      images <- inputs[[1]]
      transformParameters <- inputs[[2]]

      if (self$transformType == "affine") {
        output <-
          self$affineTransformImages(images, transformParameters, self$resampledSize)
      } else {
        stop("Error:  unsupported transform type.")
      }

      return(output)
    },

    compute_output_shape = function(input_shape) {
      numberOfChannels <- as.integer(tail(unlist(input_shape[[1]]), 1))

      return(list(
        NULL, as.integer(self$resampledSize[1]),
        as.integer(self$resampledSize[2]), numberOfChannels
      ))
    },

    affineTransformImages = function(images, affineTransformParameters, resampledSize) {
      K <- keras::backend()

      batchSize <- K$int_shape(images)[[1]]
      numberOfChannels <- K$int_shape(images)[[4]]
      if(is.null(batchSize)) {
      transformParameters <- K$reshape(affineTransformParameters,
        shape = reticulate::tuple(-1L, 2L, 3L))
      } else {
      transformParameters <- K$reshape(affineTransformParameters,
        shape = reticulate::tuple(batchSize, 2L, 3L))
      }

      regularGrid <- self$makeRegularGrid(resampledSize)
      sampledGrids <- K$dot(transformParameters, regularGrid)

      if (self$interpolatorType == "linear") {
        interpolatedImage <- self$linearInterpolate(images, sampledGrids, resampledSize)
      } else {
        stop("Error:  unsupported interpolator type.")
      }
      if(is.null(batchSize)) {
        newOutputShape <- reticulate::tuple(
          -1L, as.integer(resampledSize[1]),
          as.integer(resampledSize[2]), numberOfChannels)
        } else {
        newOutputShape <- reticulate::tuple(
          batchSize, as.integer(resampledSize[1]),
          as.integer(resampledSize[2]), numberOfChannels)
        }
      interpolatedImage <- K$reshape(interpolatedImage, shape = newOutputShape)

      return(interpolatedImage)
    },

    makeRegularGrid = function(resampledSize) {
      K <- keras::backend()

      xLinearSpace <- tensorflow::tf$linspace(-1.0, 1.0, as.integer(resampledSize[2]))
      yLinearSpace <- tensorflow::tf$linspace(-1.0, 1.0, as.integer(resampledSize[1]))

      coords <- tensorflow::tf$meshgrid(xLinearSpace, yLinearSpace)
      coords[[1]] <- K$flatten(coords[[1]])
      coords[[2]] <- K$flatten(coords[[2]])

      ones <- K$ones_like(coords[[1]])
      regularGrid <- K$concatenate(list(coords[[1]], coords[[2]], ones), axis = 0L)
      regularGrid <- K$flatten(regularGrid)
      regularGrid <- K$reshape(regularGrid, reticulate::tuple(3L, as.integer(prod(resampledSize))))

      return(regularGrid)
    },

    linearInterpolate = function(images, sampledGrids, resampledSize) {
      K <- keras::backend()

      batchSize <- K$shape(images)[1]
      height <- K$shape(images)[2]
      width <- K$shape(images)[3]
      numberOfChannels <- K$shape(images)[4]

      x <- K$cast(K$flatten(sampledGrids[, 1, ]), dtype = "float32")
      y <- K$cast(K$flatten(sampledGrids[, 2, ]), dtype = "float32")
      x <- 0.5 * (x + 1.0) * K$cast(width, dtype = "float32")
      y <- 0.5 * (y + 1.0) * K$cast(height, dtype = "float32")

      x0 <- K$cast(x, dtype = "int32")
      x1 <- x0 + 1L
      y0 <- K$cast(y, dtype = "int32")
      y1 <- y0 + 1L

      xMax <- as.integer(unlist(K$int_shape(images))[2] - 1L)
      yMax <- as.integer(unlist(K$int_shape(images))[1] - 1L)

      x0 <- K$clip(x0, 0L, xMax)
      x1 <- K$clip(x1, 0L, xMax)
      y0 <- K$clip(y0, 0L, yMax)
      y1 <- K$clip(y1, 0L, yMax)

      batchPixels <- K$arange(0L, batchSize) * (height * width)
      batchPixels <- K$expand_dims(batchPixels, axis = -1L)
      base <- K$repeat_elements(
        batchPixels,
        rep = as.integer(prod(resampledSize)), axis = 1L
      )
      base <- K$flatten(base)

      indices00 <- base + y0 * width + x0
      indices01 <- base + y1 * width + x0
      indices10 <- base + y0 * width + x1
      indices11 <- base + y1 * width + x1

      flatImages <- K$reshape(images, shape = c(-1L, numberOfChannels))
      flatImages <- K$cast(flatImages, dtype = "float32")

      pixelValues00 <- K$gather(flatImages, indices00)
      pixelValues01 <- K$gather(flatImages, indices01)
      pixelValues10 <- K$gather(flatImages, indices10)
      pixelValues11 <- K$gather(flatImages, indices11)

      x0 <- K$cast(x0, dtype = "float32")
      x1 <- K$cast(x1, dtype = "float32")
      y0 <- K$cast(y0, dtype = "float32")
      y1 <- K$cast(y1, dtype = "float32")

      weight00 <- K$expand_dims(((x1 - x) * (y1 - y)), axis = 1L)
      weight01 <- K$expand_dims(((x1 - x) * (y - y0)), axis = 1L)
      weight10 <- K$expand_dims(((x - x0) * (y1 - y)), axis = 1L)
      weight11 <- K$expand_dims(((x - x0) * (y - y0)), axis = 1L)

      interpolatedValues00 <- weight00 * pixelValues00
      interpolatedValues01 <- weight01 * pixelValues01
      interpolatedValues10 <- weight10 * pixelValues10
      interpolatedValues11 <- weight11 * pixelValues11

      interpolatedValues <- interpolatedValues00 + interpolatedValues01 +
        interpolatedValues10 + interpolatedValues11

      return(interpolatedValues)
    }
  )
)

#' spatial transformer layer (2-D and 3-D)
#'
#' Wraps a custom spatial transformer layer
#' @param object Object to compose layer with. This is either a
#' [keras::keras_model_sequential] to add the layer to,
#' or another Layer which this layer will call.
#' @param resampledSize size of the output in voxels
#' @param transformType the spatial transform
#' @param interpolatorType interpolation used for the sampling
#' @param name The name of the layer#'
#' @return a keras layer tensor
#' @export
#' @rdname layer_spatial_transformer_2d
layer_spatial_transformer_2d <- function(
                                         object,
                                         resampledSize, transformType = "affine",
                                         interpolatorType = "linear", name = NULL) {
  create_layer(
    SpatialTransformerLayer2D, object,
    list(
      resampledSize = resampledSize, transformType = transformType,
      interpolatorType = interpolatorType, name = name
    )
  )
}

#' @export
#' @rdname layer_spatial_transformer_2d
layer_spatial_transformer_3d <- function(
                                         object,
                                         resampledSize, transformType = "affine",
                                         interpolatorType = "linear", name = NULL) {
  create_layer(
    SpatialTransformerLayer3D, object,
    list(
      resampledSize = resampledSize, transformType = transformType,
      interpolatorType = interpolatorType, name = name
    )
  )
}


#' Spatial transfomer layer (3-D)
#'
#' @docType class
#'
#'
#' @section Arguments:
#' \describe{
#'  \item{inputs}{list of size 2 where the first element are the images. The second
#'                element are the weights.}
#'  \item{resampledSize}{size of the resampled output images.}
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
#' @return resampled batch images.
#'
#' @name SpatialTransformerLayer3D
NULL

#' @export
SpatialTransformerLayer3D <- R6::R6Class("SpatialTransformerLayer3D",
  inherit = KerasLayer,

  public = list(
    resampledSize = NULL,

    transformType = "affine",

    interpolatorType = "linear",

    initialize = function(resampledSize, transformType = "affine", interpolatorType = "linear") {
      K <- keras::backend()
      if (K$backend() != "tensorflow") {
        stop("Error:  tensorflow is required for this STN implementations.")
      }
      if (length(resampledSize) != 3) {
        stop("Error:  resampled size must be a vector of length 3 (for 3-D).")
      }

      self$resampledSize <- resampledSize
      self$transformType <- tolower(transformType)
      self$interpolatorType <- tolower(interpolatorType)
    },

    call = function(inputs, mask = NULL) {
      images <- inputs[[1]]
      transformParameters <- inputs[[2]]

      if (self$transformType == "affine") {
        output <-
          self$affineTransformImages(images, transformParameters, self$resampledSize)
      } else {
        stop("Error:  unsupported transform type.")
      }

      return(output)
    },

    compute_output_shape = function(input_shape) {
      numberOfChannels <- as.integer(tail(unlist(input_shape[[1]]), 1))

      return(list(
        NULL, as.integer(self$resampledSize[1]),
        as.integer(self$resampledSize[2]), as.integer(self$resampledSize[3]),
        numberOfChannels
      ))
    },

    affineTransformImages = function(images, affineTransformParameters, resampledSize) {
      K <- keras::backend()

      batchSize <- K$int_shape(images)[[1]]
      numberOfChannels <- K$int_shape(images)[[5]]
      if(is.null(batchSize)) {
      transformParameters <- K$reshape(affineTransformParameters,
        shape = reticulate::tuple(-1L, 3L, 4L))
      } else {
      transformParameters <- K$reshape(affineTransformParameters,
        shape = reticulate::tuple(batchSize, 3L, 4L))
      }

      regularGrid <- self$makeRegularGrid(resampledSize)
      sampledGrids <- K$dot(transformParameters, regularGrid)

      if (self$interpolatorType == "linear") {
        interpolatedImage <- self$linearInterpolate(images, sampledGrids, resampledSize)
      } else {
        stop("Error:  unsupported interpolator type.")
      }

      if(is.null(batchSize)) {
        newOutputShape <- reticulate::tuple(
          -1L, as.integer(resampledSize[1]),
          as.integer(resampledSize[2]), as.integer(resampledSize[3]), numberOfChannels)
        } else {
        newOutputShape <- reticulate::tuple(
          batchSize, as.integer(resampledSize[1]),
          as.integer(resampledSize[2]), as.integer(resampledSize[3]), numberOfChannels)
        }

      interpolatedImage <- K$reshape(interpolatedImage, shape = newOutputShape)

      return(interpolatedImage)
    },

    makeRegularGrid = function(resampledSize) {
      K <- keras::backend()

      xLinearSpace <- tensorflow::tf$linspace(-1.0, 1.0, as.integer(resampledSize[2]))
      yLinearSpace <- tensorflow::tf$linspace(-1.0, 1.0, as.integer(resampledSize[1]))
      zLinearSpace <- tensorflow::tf$linspace(-1.0, 1.0, as.integer(resampledSize[3]))

      coords <- tensorflow::tf$meshgrid(xLinearSpace, yLinearSpace, zLinearSpace)
      coords[[1]] <- K$flatten(coords[[1]])
      coords[[2]] <- K$flatten(coords[[2]])
      coords[[3]] <- K$flatten(coords[[3]])

      ones <- K$ones_like(coords[[1]])
      regularGrid <- K$concatenate(list(coords[[1]], coords[[2]], coords[[3]], ones), 0L)
      regularGrid <- K$flatten(regularGrid)
      regularGrid <- K$reshape(regularGrid,
        reticulate::tuple(batchSize, 4L, as.integer(prod(resampledSize))))

      return(regularGrid)
    },

    linearInterpolate = function(images, sampledGrids, resampledSize) {
      K <- keras::backend()

      batchSize <- K$shape(images)[1]
      height <- K$shape(images)[2]
      width <- K$shape(images)[3]
      depth <- K$shape(images)[4]
      numberOfChannels <- K$shape(images)[5]

      x <- K$cast(K$flatten(sampledGrids[, 1, ]), dtype = "float32")
      y <- K$cast(K$flatten(sampledGrids[, 2, ]), dtype = "float32")
      z <- K$cast(K$flatten(sampledGrids[, 3, ]), dtype = "float32")

      x <- 0.5 * (x + 1.0) * K$cast(width, dtype = "float32")
      y <- 0.5 * (y + 1.0) * K$cast(height, dtype = "float32")
      z <- 0.5 * (z + 1.0) * K$cast(depth, dtype = "float32")

      x0 <- K$cast(x, dtype = "int32")
      x1 <- x0 + 1L
      y0 <- K$cast(y, dtype = "int32")
      y1 <- y0 + 1L
      z0 <- K$cast(z, dtype = "int32")
      z1 <- z0 + 1L

      xMax <- as.integer(unlist(K$int_shape(images))[2]) - 1L
      yMax <- as.integer(unlist(K$int_shape(images))[1]) - 1L
      zMax <- as.integer(unlist(K$int_shape(images))[3]) - 1L

      x0 <- K$clip(x0, 0L, xMax)
      x1 <- K$clip(x1, 0L, xMax)
      y0 <- K$clip(y0, 0L, yMax)
      y1 <- K$clip(y1, 0L, yMax)
      z0 <- K$clip(z0, 0L, zMax)
      z1 <- K$clip(z1, 0L, zMax)

      batchPixels <- K$arange(0L, batchSize) * (height * width * depth)
      batchPixels <- K$expand_dims(batchPixels, axis = -1L)
      base <- K$repeat_elements(
        batchPixels,
        rep = as.integer(prod(resampledSize)), axis = 1L
      )
      base <- K$flatten(base)

      indices000 <- base + z0 * (width * height) + y0 * width + x0
      indices001 <- base + z1 * (width * height) + y0 * width + x0
      indices010 <- base + z0 * (width * height) + y1 * width + x0
      indices011 <- base + z1 * (width * height) + y1 * width + x0
      indices100 <- base + z0 * (width * height) + y0 * width + x1
      indices101 <- base + z1 * (width * height) + y0 * width + x1
      indices110 <- base + z0 * (width * height) + y1 * width + x1
      indices111 <- base + z1 * (width * height) + y1 * width + x1

      flatImages <- K$reshape(images, shape = c(-1L, numberOfChannels))
      flatImages <- K$cast(flatImages, dtype = "float32")

      pixelValues000 <- K$gather(flatImages, indices000)
      pixelValues001 <- K$gather(flatImages, indices001)
      pixelValues010 <- K$gather(flatImages, indices010)
      pixelValues011 <- K$gather(flatImages, indices011)
      pixelValues100 <- K$gather(flatImages, indices100)
      pixelValues101 <- K$gather(flatImages, indices101)
      pixelValues110 <- K$gather(flatImages, indices110)
      pixelValues111 <- K$gather(flatImages, indices111)

      x0 <- K$cast(x0, dtype = "float32")
      x1 <- K$cast(x1, dtype = "float32")
      y0 <- K$cast(y0, dtype = "float32")
      y1 <- K$cast(y1, dtype = "float32")
      z0 <- K$cast(z0, dtype = "float32")
      z1 <- K$cast(z1, dtype = "float32")

      weight000 <- K$expand_dims(((x1 - x) * (y1 - y) * (z1 - z)), axis = 1L)
      weight001 <- K$expand_dims(((x1 - x) * (y1 - y) * (z - z0)), axis = 1L)
      weight010 <- K$expand_dims(((x1 - x) * (y - y0) * (z1 - z)), axis = 1L)
      weight011 <- K$expand_dims(((x1 - x) * (y - y0) * (z - z0)), axis = 1L)
      weight100 <- K$expand_dims(((x - x0) * (y1 - y) * (z1 - z)), axis = 1L)
      weight101 <- K$expand_dims(((x - x0) * (y1 - y) * (z - z0)), axis = 1L)
      weight110 <- K$expand_dims(((x - x0) * (y - y0) * (z1 - z)), axis = 1L)
      weight111 <- K$expand_dims(((x - x0) * (y - y0) * (z - z0)), axis = 1L)

      interpolatedValues000 <- weight000 * pixelValues000
      interpolatedValues001 <- weight001 * pixelValues001
      interpolatedValues010 <- weight010 * pixelValues010
      interpolatedValues011 <- weight011 * pixelValues011
      interpolatedValues100 <- weight100 * pixelValues100
      interpolatedValues101 <- weight101 * pixelValues101
      interpolatedValues110 <- weight110 * pixelValues110
      interpolatedValues111 <- weight111 * pixelValues111

      interpolatedValues <-
        interpolatedValues000 +
        interpolatedValues001 +
        interpolatedValues010 +
        interpolatedValues011 +
        interpolatedValues100 +
        interpolatedValues101 +
        interpolatedValues110 +
        interpolatedValues111

      return(interpolatedValues)
    }
  )
)
