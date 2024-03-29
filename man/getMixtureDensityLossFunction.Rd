% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mixtureDensityUtilities.R
\name{getMixtureDensityLossFunction}
\alias{getMixtureDensityLossFunction}
\title{Returns a loss function for the mixture density.}
\usage{
getMixtureDensityLossFunction(outputDimension, numberOfMixtures)
}
\arguments{
\item{outputDimension}{output dimension}

\item{numberOfMixtures}{number of mixture components}
}
\value{
a function providing the mean square error accuracy
}
\description{
Ported from:
}
\details{
\if{html}{\out{<div class="sourceCode">}}\preformatted{    https://github.com/cpmpercussion/keras-mdn-layer/
}\if{html}{\out{</div>}}
}
\examples{

library( keras )


}
\author{
Tustison NJ
}
