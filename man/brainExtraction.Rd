% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/brainExtraction.R
\name{brainExtraction}
\alias{brainExtraction}
\title{Brain extraction}
\usage{
brainExtraction(
  image,
  modality = c("t1", "t1.v0", "t1.v1", "t1nobrainer", "t1combined", "t1threetissue",
    "t1hemi", "t1lobes", "t2", "t2.v0", "t2star", "flair", "flair.v0", "bold", "bold.v0",
    "fa", "fa.v0", "mra", "t1t2infant", "t1infant", "t2infant"),
  verbose = FALSE
)
}
\arguments{
\item{image}{input 3-D brain image (or list of images for multi-modal scenarios).}

\item{modality}{image type.  Options include:
\itemize{
\item{"t1": }{T1-weighted MRI---ANTs-trained.  Previous versions are specified as "t1.v0", "t1.v1".}
\item{"t1nobrainer": }{T1-weighted MRI---FreeSurfer-trained: h/t Satra Ghosh and Jakub Kaczmarzyk.}
\item{"t1combined": }{Brian's combination of "t1" and "t1nobrainer".  One can also specify
"t1combined\link{X}" where X is the morphological radius.  X = 12 by default.}
\item{"t1threetissue": }{T1-weighted MRI---originally developed from BrainWeb20 (and later expanded).
Label 1: brain + subdural CSF, label 2: sinuses + skull,
label 3: other head, face, neck tissue.}
\item{"t1hemi": }{Label 1 of "t1threetissue" subdivided into left and right hemispheres.}
\item{"flair": }{FLAIR MRI.}
\item{"t2": }{T2-w MRI.}
\item{"bold": }{3-D mean BOLD MRI.}
\item{"fa": }{Fractional anisotropy.}
\item{"t1t2infant": }{Combined T1-w/T2-w infant MRI h/t Martin Styner.}
\item{"t1infant": }{T1-w infant MRI h/t Martin Styner.}
\item{"t2infant": }{T2-w infant MRI h/t Martin Styner.}
}}

\item{verbose}{print progress.}
}
\value{
brain probability mask (ANTsR image)
}
\description{
Perform T1, FA, or bold brain extraction using a U-net architecture
training data.  "NoBrainer" is also possible where
brain extraction uses U-net and FreeSurfer
training data ported from the
}
\details{
\url{https://github.com/neuronets/nobrainer-models}
}
\examples{
\dontrun{
library( ANTsRNet )
library( keras )

image <- antsImageRead( "t1w_image.nii.gz" )
probabilityMask <- brainExtraction( image, modality = "t1" )
}
}
\author{
Tustison NJ
}
