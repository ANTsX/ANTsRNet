#' Set the ANTsXNet Cache Directory
#'
#' This function sets the cache directory used by ANTsXNet to store downloaded
#' templates and model weights. If the specified directory does not exist,
#' it will be created.
#'
#' @param path A character string specifying the path to the cache directory.
#' @return NULL
#' @export
setANTsXNetCacheDirectory <- function(path) {
  normalizedPath <- normalizePath( path, winslash = "/", mustWork = FALSE )

  if (!dir.exists(normalizedPath)) {
    dir.create(normalizedPath, recursive = TRUE)
  }

  options(antsxnetCacheDirectory = normalizedPath)
}


#' Get the ANTsXNet Cache Directory
#'
#' This function returns the current cache directory used by ANTsXNet to store
#' downloaded templates and model weights. If no directory has been set, it
#' initializes it to the default path "~/.keras/ANTsXNet".
#'
#' @return A character string specifying the path to the cache directory.
#' @export
getANTsXNetCacheDirectory <- function() {
  cacheDir <- getOption("antsxnetCacheDirectory", default = NULL)

  if (is.null(cacheDir)) {
    cacheDir <- normalizePath("~/.keras/ANTsXNet", winslash = "/", mustWork = FALSE)
    options(antsxnetCacheDirectory = cacheDir)
  }

  return(cacheDir)
}


#' getANTsXNetData
#'
#' Download data such as prefabricated templates and spatial priors.
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
              "deepFlashTemplate2T1SkullStripped",
              "deepFlashTemplate2Labels",
              "mprage_hippmapp3r",
              "protonLungTemplate",
              "ctLungTemplate",
              "luna16LungPriors",
              "xrayLungPriors",
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
              "magetCerebellumxTemplate0GenericAffine",
              "mraTemplate",
              "mraTemplateBrainMask",
              "mraTemplateVesselPrior",
              "bsplineT2MouseTemplate",
              "bsplineT2MouseTemplateBrainMask",
              "DevCCF_P56_MRI_T2_50um",
              "DevCCF_P56_MRI_T2_50um_BrainParcellationNickMask",
              "DevCCF_P56_MRI_T2_50um_BrainParcellationTctMask",
              "DevCCF_P04_STPT_50um",
              "DevCCF_P04_STPT_50um_BrainParcellationJayMask",
              "hcpaT1Template",
              "hcpaT2Template",
              "hcpaFATemplate",
              "hcpyaT1Template",
              "hcpyaT2Template",
              "hcpyaFATemplate",
              "hcpyaTemplateBrainMask",
              "hcpyaTemplateBrainSegmentation",
              "hcpinterT1Template",
              "hcpinterT2Template",
              "hcpinterFATemplate",
              "hcpinterTemplateBrainMask",
              "hcpinterTemplateBrainSegmentation"
            ),
  targetFileName)
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
    deepFlashTemplate2T1SkullStripped = "https://figshare.com/ndownloader/files/46461451",
    deepFlashTemplate2Labels = "https://figshare.com/ndownloader/files/46461415",
    mprage_hippmapp3r = "https://ndownloader.figshare.com/files/24984689",
    protonLungTemplate = "https://ndownloader.figshare.com/files/22707338",
    ctLungTemplate = "https://ndownloader.figshare.com/files/22707335",
    luna16LungPriors = "https://ndownloader.figshare.com/files/28253796",
    xrayLungPriors = "https://figshare.com/ndownloader/files/41965815",
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
    magetCerebellumxTemplate0GenericAffine = "https://figshare.com/ndownloader/files/41052575",
    mraTemplate = "https://figshare.com/ndownloader/files/46406695",
    mraTemplateBrainMask = "https://figshare.com/ndownloader/files/46406698",
    mraTemplateVesselPrior = "https://figshare.com/ndownloader/files/46406713",
    hcpaT1Template = "https://figshare.com/ndownloader/files/54248318",
    hcpaT2Template = "https://figshare.com/ndownloader/files/54248324",
    hcpaFATemplate = "https://figshare.com/ndownloader/files/54248321",
    hcpyaT1Template = "https://figshare.com/ndownloader/files/46746142",
    hcpyaT2Template = "https://figshare.com/ndownloader/files/46746334",
    hcpyaFATemplate = "https://figshare.com/ndownloader/files/46746349",
    hcpyaTemplateBrainMask = "https://figshare.com/ndownloader/files/46746388",
    hcpyaTemplateBrainSegmentation = "https://figshare.com/ndownloader/files/46746367",
    hcpinterT1Template = "https://figshare.com/ndownloader/files/49372855",
    hcpinterT2Template = "https://figshare.com/ndownloader/files/49372849",
    hcpinterFATemplate = "https://figshare.com/ndownloader/files/49372858",
    hcpinterTemplateBrainMask = "https://figshare.com/ndownloader/files/49372861",
    hcpinterTemplateBrainSegmentation = "https://figshare.com/ndownloader/files/49372852",
    bsplineT2MouseTemplate = "https://figshare.com/ndownloader/files/44706247",
    bsplineT2MouseTemplateBrainMask = "https://figshare.com/ndownloader/files/44869285",
    DevCCF_P56_MRI_T2_50um = "https://figshare.com/ndownloader/files/44706244",
    DevCCF_P56_MRI_T2_50um_BrainParcellationNickMask = "https://figshare.com/ndownloader/files/44706238",
    DevCCF_P56_MRI_T2_50um_BrainParcellationTctMask = "https://figshare.com/ndownloader/files/47214532",
    DevCCF_P04_STPT_50um = "https://figshare.com/ndownloader/files/46711546",
    DevCCF_P04_STPT_50um_BrainParcellationJayMask = "https://figshare.com/ndownloader/files/46712656"
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

  targetFileNamePath <- fs::path_join( path.expand( c( getANTsXNetCacheDirectory(), targetFileName ) ) )

  if( ! fs::file_exists( targetFileNamePath ) )
    {
    targetFileNamePath <- tensorflow::tf$keras$utils$get_file(
      targetFileName, url, cache_subdir = getANTsXNetCacheDirectory() )
    }

  return( as.character( targetFileNamePath ) )
}
