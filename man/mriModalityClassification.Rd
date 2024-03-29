% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mriModalityClassification.R
\name{mriModalityClassification}
\alias{mriModalityClassification}
\title{MRI modality classification}
\usage{
mriModalityClassification(
  image,
  antsxnetCacheDirectory = NULL,
  verbose = FALSE
)
}
\arguments{
\item{image}{raw 3-D MRI whole head image.}

\item{antsxnetCacheDirectory}{destination directory for storing the downloaded
template and model weights.  Since these can be resused, if
\code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
subdirectory ~/.keras/ANTsXNet/.}

\item{verbose}{print progress.}
}
\value{
classification data frame
}
\description{
Predict MRI modality type (whole-head only).
Modalities:
\itemize{
\item{T1}
\item{T2}
\item{FLAIR}
\item{T2Star}
\item{Mean DWI}
\item{Mean Bold}
\item{ASL perfusion}
}
}
\examples{
\dontrun{
library( ANTsRNet )

image <- antsImageRead( getANTsXNetData( "mprageHippmap3r" ) )
classification <- mriModalityClassification( image )
}
}
\author{
Tustison NJ
}
