% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mouse.R
\name{mouseBrainParcellation}
\alias{mouseBrainParcellation}
\title{Mouse brain parcellation}
\usage{
mouseBrainParcellation(
  image,
  mask = NULL,
  returnIsotropicOutput = FALSE,
  whichParcellation = c("nick", "jay", "tct"),
  verbose = FALSE
)
}
\arguments{
\item{image}{input 3-D brain image (or list of images for multi-modal scenarios).}

\item{mask}{Brain mask.  If not specified, one is estimated using ANTsXNet mouse
brain extraction.}

\item{returnIsotropicOutput}{The network actually learns an interpolating
function specific to the mouse brain.  Setting this to true, the output
images are returned isotropically resampled.}

\item{whichParcellation}{Brain parcellation type:
\itemize{
\item{"nick":
\itemize{
\item{Label 0:}{background}
\item{Label 1:}{cerebral cortex}
\item{Label 2:}{cerebral nuclei}
\item{Label 3:}{brain stem}
\item{Label 4:}{cerebellum}
\item{Label 5:}{main olfactory bulb}
\item{Label 6:}{hippocampal formation}
}}
}
\itemize{
\item{"tct":
\itemize{
\item{Label 0:}{}
\item{Label 1:}{background}
\item{Label 2:}{Infralimbic area}
\item{Label 3:}{Prelimbic area}
\item{Label 4:}{Medial group of the dorsal thalamus}
\item{Label 5:}{Reticular nucleus of the thalamus}
\item{Label 6:}{Hippocampal formation}
\item{Label 7:}{Cerebellum}
}}
}
\itemize{
\item{"jay":
\itemize{
\item{Label 0:}{background}
\item{Label 1:}{}
\item{Label 2:}{}
\item{Label 3:}{}
\item{Label 4:}{}
}}
}}

\item{verbose}{print progress.}
}
\value{
brain probability mask
}
\description{
Perform brain extraction of mouse T2 MRI
}
\examples{
\dontrun{
library( ANTsRNet )
library( keras )

image <- antsImageRead( "brain.nii.gz" )
parcellation <- mouseBrainParcellation( image, modality = "t2" )
}
}
\author{
Tustison NJ
}
