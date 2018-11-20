#' getPretrainedNetwork
#'
#' Downloads antsrnet pretrained network. \url{10.6084/m9.figshare.7246985}
#'
#' @param fileid one of the permitted file ids or pass "show" to list all
#'   valid possibilities. Note that most require internet access to download.
#' @param targetFileName optional target filename
#' @param verbose If \code{TRUE}, suppress status messages
#' (if any), and the progress bar.
#' @return filename string
#' @author Avants BB
#' @examples
#'
#' net <- getPretrainedNetwork( "dbpn4x" )
#'
#' @export getPretrainedNetwork
getPretrainedNetwork <- function(fileid,
                         targetFileName = FALSE,
                         verbose=FALSE ) {
  myusage <- "usage: getPretrainedNetwork(fileid = whatever , usefixedlocation = TRUE )"
  if (missing(fileid)) {
    print(myusage)
    return(NULL)
  }
  validlist = c( "dbpn4x", "brainExtraction", "protonLungMri", "show" )
  if (  sum( validlist == fileid ) == 0 ) {
    message("Try:")
    print( validlist )
    stop("no data with the id you passed - try show to get list of valid ids")
    }
  if ( fileid == "show" )
    return( validlist )
  myurl <- switch( fileid,
                   dbpn4x = "https://ndownloader.figshare.com/files/13347617",
                   brainExtraction = "https://ndownloader.figshare.com/files/13606802",
                   protonLungMri = "https://ndownloader.figshare.com/files/13606799"
                   )

  if ( missing( targetFileName ) ) {
    myext <- ".h5"
    tdir <- tempdir()  # for temporary storage
    targetFileName <- tempfile( tmpdir = tdir, fileext = myext )  # for temporary storage
    }
  if ( ! file.exists( targetFileName ) )
    {
    download.file( myurl, targetFileName  )
    }
  return( targetFileName )
}
