#' getPretrainedNetwork
#'
#' Downloads pretrained network/weights.
#'
#' @param fileId one of the permitted file ids or pass "show" to list all
#'   valid possibilities. Note that most require internet access to download.
#' @param targetFileName optional target filename
#' @return filename string
#' @author Avants BB
#' @note See \url{https://figshare.com/authors/Nick_Tustison/441144}
#' or \url{https://figshare.com/authors/Brian_Avants/418551}
#' for some more descriptions
#' @examples
#' \dontrun{
#' net <- getPretrainedNetwork( "dbpn4x" )
#' }
#' @export getPretrainedNetwork
getPretrainedNetwork <- function(
  fileId = c("show",
             "dbpn4x",
             "mriSuperResolution",
             "brainExtraction",
             "brainSegmentation",
             "brainSegmentationPatchBased",
             "brainAgeGender",
             "denoising",
             "wholeTumorSegmentationT2Flair",
             "protonLungMri",
             "ctHumanLung",
             "functionalLungMri",
             "hippMapp3rInitial",
             "hippMapp3rRefine"),
  targetFileName )
{


  if( fileId[1] == "show" )
  {
    return( fileId )
  }
  fileId = match.arg(fileId)

  url <- switch(
    fileId,
    dbpn4x = "https://ndownloader.figshare.com/files/13347617",
    mriSuperResolution = "https://ndownloader.figshare.com/files/19430123",
    brainExtraction = "https://ndownloader.figshare.com/files/13729661",
    brainSegmentation = "https://ndownloader.figshare.com/files/13900010",
    brainSegmentationPatchBased = "https://ndownloader.figshare.com/files/14249717",
    brainAgeGender = "https://ndownloader.figshare.com/files/14394350",
    denoising = "https://ndownloader.figshare.com/files/14235296",
    wholeTumorSegmentationT2Flair = "https://ndownloader.figshare.com/files/14087045",
    protonLungMri = "https://ndownloader.figshare.com/files/13606799",
    ctHumanLung = "https://ndownloader.figshare.com/files/20005217",
    functionalLungMri = "https://ndownloader.figshare.com/files/13824167",
    hippMapp3rInitial = "https://ndownloader.figshare.com/files/18068408",
    hippMapp3rRefine = "https://ndownloader.figshare.com/files/18068411"
  )

  if( missing( targetFileName ) )
  {
    targetFileName <- tempfile( fileext = ".h5" )
  }

  if( ! file.exists( targetFileName ) )
  {
    download.file( url, targetFileName  )
  }
  return( targetFileName )
}
