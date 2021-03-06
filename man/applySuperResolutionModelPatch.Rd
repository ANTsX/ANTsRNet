% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/applyDBPN4x.R
\name{applySuperResolutionModelPatch}
\alias{applySuperResolutionModelPatch}
\title{applySuperResolutionModelPatch}
\usage{
applySuperResolutionModelPatch(
  image,
  model,
  targetRange,
  lowResolutionPatchSize = 128,
  strideLength = 16,
  batch_size = 32,
  mask,
  verbose = FALSE
)
}
\arguments{
\item{image}{input image}

\item{model}{model object or filename see \code{getPretrainedNetwork}}

\item{targetRange}{a vector defining min max of each the input image,
eg -127.5, 127.5.  Output images will be scaled back to original intensity.
This range should match the mapping used in the training of the network.}

\item{lowResolutionPatchSize}{size of patches to upsample}

\item{strideLength}{voxel/pixel steps between patches}

\item{batch_size}{for prediction call}

\item{mask}{restrict intensity rescaling parameters within the mask}

\item{verbose}{If \code{TRUE}, show status messages}
}
\value{
image upscaled to resolution provided by network
}
\description{
Apply pretrained super-resolution network by stitching together patches.
}
\details{
Apply a patch-wise trained network to perform super-resolution. Can be applied
to variable sized inputs. Warning: This function may be better used on CPU
unless the GPU can accommodate the full patch size. Warning 2: The global
intensity range (min to max) of the output will match the input where the
range is taken over all channels.
}
\examples{
\dontrun{
library(ANTsRCore)
library( keras )
orig_img = antsImageRead( getANTsRData( "r16" ) )
# input needs to be 48x48
model = createDeepBackProjectionNetworkModel2D( list(NULL,NULL, 1) )
img = resampleImage(orig_img, resampleParams = rep(256/48, 2))
simg <- applySuperResolutionModelPatch( img,
 model = model, lowResolutionPatchSize = 8, strideLength = 2)
simgm <- applySuperResolutionModelPatch( img, mask = getMask( img ),
 model = model, lowResolutionPatchSize = 8, strideLength = 2)
plot( orig_img )
plot( img )
plot( simg )
plot( simgm )
}
}
\author{
Avants BB
}
