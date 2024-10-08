% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/createCustomUnetModel.R
\name{createShivaUnetModel3D}
\alias{createShivaUnetModel3D}
\title{Implementation of the "shiva" u-net architecture for PVS and WMH
segmentation}
\usage{
createShivaUnetModel3D(numberOfModalities = 1)
}
\arguments{
\item{numberOfModalities}{Specifies number of channels in the
architecture.}
}
\value{
a u-net keras model
}
\description{
Publications:

\if{html}{\out{<div class="sourceCode">}}\preformatted{* PVS:  https://pubmed.ncbi.nlm.nih.gov/34262443/
* WMH:  https://pubmed.ncbi.nlm.nih.gov/38050769/
}\if{html}{\out{</div>}}
}
\details{
with respective GitHub repositories:

\if{html}{\out{<div class="sourceCode">}}\preformatted{* PVS:  https://github.com/pboutinaud/SHIVA_PVS
* WMH:  https://github.com/pboutinaud/SHIVA_WMH
}\if{html}{\out{</div>}}
}
\examples{
\dontrun{

model <- createShivaUnetModel3D()

}
}
\author{
Tustison NJ
}
