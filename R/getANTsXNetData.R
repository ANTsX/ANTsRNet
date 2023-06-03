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
              "deepFlashPriors",
              "deepFlashTemplateT1",
              "deepFlashTemplateT1SkullStripped",
              "deepFlashTemplateT2",
              "deepFlashTemplateT2SkullStripped",
              "mprage_hippmapp3r",
              "protonLungTemplate",
              "ctLungTemplate",
              "luna16LungPriors",
              "priorDktLabels",
              "priorDeepFlashLeftLabels",
              "priorDeepFlashRightLabels",
              "protonLobePriors",
              "S_template3",
              "adni",
              "ixi",
              "kirby",
              "mni152",
              "nki",
              "nki10",
              "oasis",
              "magetTemplate",
              "magetTemplateBrainMask",
              "magetCerebellumTemplate",
              "magetCerebellumTemplatePriors",
              "magetCerebellumxTemplate0GenericAffine"
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
    deepFlashPriors = "https://figshare.com/ndownloader/files/31208272",
    deepFlashTemplateT1 = "https://figshare.com/ndownloader/files/31207795",
    deepFlashTemplateT1SkullStripped = "https://figshare.com/ndownloader/files/31339867",
    deepFlashTemplateT2 = "https://figshare.com/ndownloader/files/31207798",
    deepFlashTemplateT2SkullStripped = "https://figshare.com/ndownloader/files/31339870",
    mprage_hippmapp3r = "https://ndownloader.figshare.com/files/24984689",
    protonLungTemplate = "https://ndownloader.figshare.com/files/22707338",
    ctLungTemplate = "https://ndownloader.figshare.com/files/22707335",
    luna16LungPriors = "https://ndownloader.figshare.com/files/28253796",
    priorDktLabels = "https://ndownloader.figshare.com/files/24139802",
    S_template3 = "https://ndownloader.figshare.com/files/22597175",
    priorDeepFlashLeftLabels = "https://ndownloader.figshare.com/files/25422098",
    priorDeepFlashRightLabels = "https://ndownloader.figshare.com/files/25422101",
    protonLobePriors = "https://figshare.com/ndownloader/files/30678452",
    adni = "https://ndownloader.figshare.com/files/25516361",
    ixi = "https://ndownloader.figshare.com/files/25516358",
    kirby = "https://ndownloader.figshare.com/files/25620107",
    mni152 = "https://ndownloader.figshare.com/files/25516349",
    nki = "https://ndownloader.figshare.com/files/25516355",
    nki10 = "https://ndownloader.figshare.com/files/25516346",
    oasis = "https://ndownloader.figshare.com/files/25516352",
    magetTemplate = "https://figshare.com/ndownloader/files/41052572",
    magetTemplateBrainMask = "https://figshare.com/ndownloader/files/41052569",
    magetCerebellumTemplate = "https://figshare.com/ndownloader/files/41052581",
    magetCerebellumTemplatePriors = "https://figshare.com/ndownloader/files/41052578",
    magetCerebellumxTemplate0GenericAffine = "https://figshare.com/ndownloader/files/41052575"            
  )

  if( missing( targetFileName ) )
    {
    if( fileId == "magetCerebellumxTemplate0GenericAffine" )
      {
      targetFileName <- paste0( fileId, ".mat" )
      } else {
      targetFileName <- paste0( fileId, ".nii.gz" )
      }
    }

  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }

  targetFileNamePath <- ""

  # Check if file exists before involving keras, which requires a writeable cache directory
  # Checking first allows singularity containers to use internal data
  if (fs::is_absolute_path(antsxnetCacheDirectory))
    {
    targetFileNamePath <- fs::path_join(c(antsxnetCacheDirectory, targetFileName))
    }
  else
    {
    targetFileNamePath <- fs::path_join( path.expand( c( "~/.keras", antsxnetCacheDirectory, targetFileName ) ) )
    }

  if (fs::file_exists(targetFileNamePath))
    {
    return( as.character( targetFileNamePath) )
    }

  targetFileNamePath <- tensorflow::tf$keras$utils$get_file(
    targetFileName, url, cache_subdir = antsxnetCacheDirectory )

  return( as.character( targetFileNamePath ) )
}
