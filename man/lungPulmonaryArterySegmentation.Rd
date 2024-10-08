% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lungSegmentation.R
\name{lungPulmonaryArterySegmentation}
\alias{lungPulmonaryArterySegmentation}
\title{Pulmonary artery segmentation.}
\usage{
lungPulmonaryArterySegmentation(
  ct,
  lungMask = NULL,
  predictionBatchSize = 16,
  patchStrideLength = 32,
  verbose = FALSE
)
}
\arguments{
\item{ct}{input 3-D ct image.}

\item{lungMask}{input binary lung mask which defines the patch extraction.
If not supplied, one is estimated.}

\item{predictionBatchSize}{Control memory usage for prediction.  More consequential
for GPU-usage.}

\item{patchStrideLength}{3-D vector or int.   Dictates the stride length for
accumulating predicting patches.}

\item{verbose}{print progress.}
}
\value{
Probability image.
}
\description{
Perform pulmonary artery segmentation.  Training data taken from the
PARSE2022 challenge (Luo, Gongning, et al. "Efficient automatic segmentation
for multi-level pulmonary arteries: The PARSE challenge."
https://arxiv.org/abs/2304.03708).
}
\examples{
\dontrun{
library( ANTsRNet )
library( keras )

}
}
\author{
Tustison NJ
}
