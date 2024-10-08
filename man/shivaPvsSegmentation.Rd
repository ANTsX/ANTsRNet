% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/whiteMatterHyperintensitySegmentation.R
\name{shivaPvsSegmentation}
\alias{shivaPvsSegmentation}
\title{SHIVA PVS/VRS segmentation.}
\usage{
shivaPvsSegmentation(
  t1,
  flair = NULL,
  whichModel = "all",
  doPreprocessing = TRUE,
  verbose = FALSE
)
}
\arguments{
\item{t1}{input 3-D T1-weighted brain image.}

\item{flair}{(Optional) input 3-D FLAIR brain image (aligned to T1 image).}

\item{whichModel}{integer or string. Several models were trained for the
case of T1-only or T1/FLAIR image pairs.  One can use a specific single
trained model or the average of the entire ensemble.  I.e., options are:
* For T1-only:  0, 1, 2, 3, 4, 5.
* For T1/FLAIR: 0, 1, 2, 3, 4.
* Or "all" for using the entire ensemble.}

\item{doPreprocessing}{perform n4 bias correction, intensity truncation, brain
extraction.}

\item{verbose}{print progress.}
}
\value{
probabilistic image.
}
\description{
Perform segmentation of perivascular (PVS) or Vircho-Robin spaces (VRS).
\url{https://pubmed.ncbi.nlm.nih.gov/34262443/}
with the original implementation available here:
https://github.com/pboutinaud/SHIVA_PVS
}
\examples{
\dontrun{
library( ANTsRNet )
library( keras )

t1 <- antsImageRead( "t1.nii.gz" )
flair <- antsImageRead( "flair.nii.gz" )
results <- wmhSegmentation( t1, flair )
}
}
\author{
Tustison NJ
}
