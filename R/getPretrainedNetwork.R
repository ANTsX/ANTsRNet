#' getPretrainedNetwork
#'
#' Downloads antsrnet pretrained network. \url{10.6084/m9.figshare.7246985}
#'
#' @param fileid one of the permitted file ids or pass "show" to list all
#'   valid possibilities. Note that most require internet access to download.
#' @param targetFileName optional target filename
#' @return filename string
#' @author Avants BB
#' @examples
#' \dontrun{
#' net <- getPretrainedNetwork( "dbpn4x" )
#' }
#' @export getPretrainedNetwork
getPretrainedNetwork <- function( fileId,
                                  targetFileName )
{
  if( missing( fileId ) )
    {
    stop( "Missing file id." )
    }

  validList = c( "dbpn4x",
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
                 "hippMapp3rRefine",
                 "show" )

  if( ! fileId %in% validList )
    {
    stop( "No data with the id you passed - try \"show\" to get list of valid ids." )
    }

  if( fileId == "show" )
    {
    return( validList )
    }

  url <- switch( fileId,
                 dbpn4x = "https://ndownloader.figshare.com/files/13347617",
                 brainExtraction = "https://ndownloader.figshare.com/files/13729661",
                 brainSegmentation = "https://ndownloader.figshare.com/files/13900010",
                 brainSegmentationPatchBased = "https://ndownloader.figshare.com/files/14249717",
                 brainAgeGender = "https://ndownloader.figshare.com/files/14394350",
                 denoising = "https://ndownloader.figshare.com/files/14235296",
                 wholeTumorSegmentationT2Flair = "https://ndownloader.figshare.com/files/14087045",
                 protonLungMri = "https://ndownloader.figshare.com/files/13606799",
                 ctHumanLung = "https://ndownloader.figshare.com/files/16874150",
                 functionalLungMri = "https://ndownloader.figshare.com/files/13824167",
                 hippMapp3rInitial = "https://ndownloader.figshare.com/files/18068408",
                 hippMapp3rRefine = "https://ndownloader.figshare.com/files/18068411"
                 )

  if( missing( targetFileName ) )
    {
    tempDirectory <- tempdir()
    targetFileName <- tempfile( tmpdir = tempDirectory, fileext = ".h5" )
    }

  if( ! file.exists( targetFileName ) )
    {
    download.file( url, targetFileName  )
    }
  return( targetFileName )
}
