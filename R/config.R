
# Set default options when package is loaded
.onLoad <- function( libname, pkgname ) 
{
  op <- options()
  op.antsxnet <- list(
    antsxnet.deterministic = FALSE,
    antsxnet.randomSeed = NULL
  )
  toSet <- !( names( op.antsxnet ) %in% names( op ) )
  if( any( toSet ) ) {
    options( op.antsxnet[toSet] )
  } 
  invisible()
}

#' Set ANTXNet deterministic behavior
#'
#' @param on Set deterministic behavior.
#' @param seedValue Assign random seed value.
#'
#' @export setANTsXNetDeterministic
setANTsXNetDeterministic <- function( on = TRUE, seedValue = 123 ) 
{
  options( antsxnet.deterministic = on )
  if( on ) {
    if( "ANTs" %in% rownames( installed.packages() ) ) {
          try({
            ANTsR::setANTsDeterministic( on, seedValue = seedValue )
          }, silent = TRUE )
    }    
    if( !is.null( seedValue ) ) {
      set.seed( seedValue )
      options( antsxnet.randomSeed = seedValue )
    }
  }
}
