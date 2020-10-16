#' getANTsXNetData
#'
#' Download data such as prefabricated templates and spatial priors.
#'
#' @param fileId one of the permitted file ids or pass "show" to list all
#'   valid possibilities. Note that most require internet access to download.
#' @param targetFileName optional target filename
#' @param antsxnetCacheDirectory destination directory for storing the downloaded
#' template and model weights.  Since these can be resused, if
#' \code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
#' subdirectory ~/.keras/ANTsXNet/.
#' @return filename string
#' @author Avants BB
#' @note See \url{https://figshare.com/authors/Nick_Tustison/441144}
#' or \url{https://figshare.com/authors/Brian_Avants/418551}
#' for some more descriptions
#' @examples
#' \dontrun{
#' net <- getANTsXNetData("biobank")
#' }
#' @export getANTsXNetData
getANTsXNetData <- function(
  fileId = c( "show",
              "biobank",
              "croppedMni152",
              "mprage_hippmapp3r",
              "protonLungTemplate",
              "ctLungTemplate",
              "priorDktLabels",
              "S_template3" ),
  targetFileName, antsxnetCacheDirectory = NULL )
{


  if( fileId[1] == "show" )
    {
    return( fileId )
   }
  fileId = match.arg( fileId )

  url <- switch(
    fileId,
    biobank = "https://ndownloader.figshare.com/files/22429242",
    croppedMni152 = "https://ndownloader.figshare.com/files/22933754",
    mprage_hippmapp3r = "https://ndownloader.figshare.com/files/24139802",
    protonLungTemplate = "https://ndownloader.figshare.com/files/22707338",
    ctLungTemplate = "https://ndownloader.figshare.com/files/22707335",
    priorDktLabels = "https://ndownloader.figshare.com/files/24139802",
    S_template3 = "https://ndownloader.figshare.com/files/22597175"
  )

  if( missing( targetFileName ) )
    {
    targetFileName <- paste0( fileId, ".nii.gz" )
    }
  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }
  targetFileNamePath <- tensorflow::tf$keras$utils$get_file(
    targetFileName, url, cache_subdir = antsxnetCacheDirectory )
  return( targetFileNamePath )
}
