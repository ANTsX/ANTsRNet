% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/customMetrics.R
\name{binary_surface_loss}
\alias{binary_surface_loss}
\title{Function for surface loss}
\usage{
binary_surface_loss(y_true, y_pred, dimensionality = 3L)
}
\arguments{
\item{y_true}{True labels (Tensor)}

\item{y_pred}{Predictions (Tensor of the same shape as \code{y_true})}
}
\value{
function value
}
\description{
\url{https://pubmed.ncbi.nlm.nih.gov/33080507/}
}
\details{
ported from this implementation:

\url{https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py}

Note:  Assumption is that y_true is a one-hot representation
of the segmentation batch.  The background (label 0) should
be included but is not used in the calculation.
}
\examples{

library( ANTsRNet )
library( keras )

model <- createUnetModel2D( c( 64, 64, 1 ), numberOfOutputs = 2 )

model \%>\% compile( loss = binary_surface_loss,
 optimizer = optimizer_adam( lr = 0.0001 ),
   metrics = "accuracy" )

########################################
#
# Run in isolation
#

library( ANTsR )

r16 <- antsImageRead( getANTsRData( "r16" ) )
r16seg <- kmeansSegmentation( r16, 3 )$segmentation
r16array <- array( data = as.array( r16seg ), dim = c( 1, dim( r16seg ) ) )
r16tensor <- tensorflow::tf$convert_to_tensor( encodeUnet( r16array, c( 0, 1, 2, 3 ) ) )

r64 <- antsImageRead( getANTsRData( "r64" ) )
r64seg <- kmeansSegmentation( r64, 3 )$segmentation
r64array <- array( data = as.array( r64seg ), dim = c( 1, dim( r64seg ) ) )
r64tensor <- tensorflow::tf$convert_to_tensor( encodeUnet( r64array, c( 0, 1, 2, 3 ) ) )

surface_loss <- binary_surface_loss( r16tensor, r64tensor, dimensionality = 2L )
loss_value <- surface_loss( r16tensor, r64tensor )$numpy()

}
\author{
Tustison NJ
}
