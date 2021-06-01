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
              "croppedMni152Priors",
              "mprage_hippmapp3r",
              "protonLungTemplate",
              "ctLungTemplate",
              "luna16LungPriors",
              "priorDktLabels",
              "priorDeepFlashLeftLabels",
              "priorDeepFlashRightLabels",
              "S_template3",
              "adni",
              "ixi",
              "kirby",
              "mni152",
              "nki",
              "nki10",
              "oasis"
            ),
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
    croppedMni152Priors = "https://ndownloader.figshare.com/files/27688437",
    mprage_hippmapp3r = "https://ndownloader.figshare.com/files/24984689",
    protonLungTemplate = "https://ndownloader.figshare.com/files/22707338",
    ctLungTemplate = "https://ndownloader.figshare.com/files/22707335",
    luna16LungPriors = "https://ndownloader.figshare.com/files/28253796",
    priorDktLabels = "https://ndownloader.figshare.com/files/24139802",
    S_template3 = "https://ndownloader.figshare.com/files/22597175",
    priorDeepFlashLeftLabels = "https://ndownloader.figshare.com/files/25422098",
    priorDeepFlashRightLabels = "https://ndownloader.figshare.com/files/25422101",
    adni = "https://ndownloader.figshare.com/files/25516361",
    ixi = "https://ndownloader.figshare.com/files/25516358",
    kirby = "https://ndownloader.figshare.com/files/25620107",
    mni152 = "https://ndownloader.figshare.com/files/25516349",
    nki = "https://ndownloader.figshare.com/files/25516355",
    nki10 = "https://ndownloader.figshare.com/files/25516346",
    oasis = "https://ndownloader.figshare.com/files/25516352"
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
