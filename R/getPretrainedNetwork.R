#' getPretrainedNetwork
#'
#' Downloads pretrained network/weights.
#'
#' @param fileId one of the permitted file ids or pass "show" to list all
#'   valid possibilities. Note that most require internet access to download.
#' @param targetFileName optional target filename
#' @param overwrite shoudl the file be overwritten
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
  fileId = c( "show",
              "brainAgeGender",
              "brainAgeFmrib",
              "brainExtraction",
              "brainSegmentation",
              "brainSegmentationPatchBased",
              "ctHumanLung",
              "dbpn4x",
              "denoising",
              "functionalLungMri",
              "hippMapp3rInitial",
              "hippMapp3rRefine",
              "mriSuperResolution",
              "protonLungMri",
              "wholeTumorSegmentationT2Flair" ),
  targetFileName,
  overwrite = FALSE)
{


  if( fileId[1] == "show" )
  {
    return( fileId )
  }
  fileId = match.arg( fileId )

  url <- switch(
    fileId,
    brainAgeGender = "https://ndownloader.figshare.com/files/22179948",
    brainAgeFmrib = "https://ndownloader.figshare.com/files/22429077",
    brainExtraction = "https://ndownloader.figshare.com/files/13729661",
    brainSegmentation = "https://ndownloader.figshare.com/files/13900010",
    brainSegmentationPatchBased = "https://ndownloader.figshare.com/files/142497 17",
    ctHumanLung = "https://ndownloader.figshare.com/files/20005217",
    dbpn4x = "https://ndownloader.figshare.com/files/13347617",
    denoising = "https://ndownloader.figshare.com/files/14235296",
    functionalLungMri = "https://ndownloader.figshare.com/files/13824167",
    hippMapp3rInitial = "https://ndownloader.figshare.com/files/18068408",
    hippMapp3rRefine = "https://ndownloader.figshare.com/files/18068411",
    mriSuperResolution = "https://ndownloader.figshare.com/files/19430123",
    protonLungMri = "https://ndownloader.figshare.com/files/13606799",
    wholeTumorSegmentationT2Flair = "https://ndownloader.figshare.com/files/14087045"
  )

  if( missing( targetFileName ) )
  {
    targetFileName <- file.path( tempdir(), paste0( fileId, ".h5" ) )
  }

  if( ! file.exists( targetFileName ) || overwrite )
  {
    download.file( url, targetFileName, overwrite = overwrite )
  }
  return( targetFileName )
}
