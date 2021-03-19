#' getPretrainedNetwork
#'
#' Downloads pretrained network/weights.
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
#' net <- getPretrainedNetwork( "dbpn4x" )
#' }
#' @export getPretrainedNetwork
getPretrainedNetwork <- function(
  fileId = c( "show",
              "brainAgeGender",
              "brainAgeFmrib",
              "brainAgeDeepBrainNet",
              "brainExtraction",
              "brainExtractionT2",
              "brainExtractionFLAIR",
              "brainExtractionBOLD",
              "brainExtractionFA",
              "brainExtractionNoBrainer",
              "brainExtractionInfantT1T2",
              "brainExtractionInfantT1",
              "brainExtractionInfantT2",
              "brainSegmentation",
              "brainSegmentationPatchBased",
              "ctHumanLung",
              "dbpn4x",
              "deepFlash",
              "deepFlashLeft8",
              "deepFlashRight8",
              "deepFlashLeft16",
              "deepFlashRight16",
              "deepFlashLeft16new",
              "deepFlashRight16new",
              "denoising",
              "dktInner",
              "dktOuter",
              "dktOuterWithSpatialPriors",
              "elBicho",
              "ewDavidWmhSegmentationWeights",
              "ewDavidWmhSegmentationSlicewiseWeights",
              "functionalLungMri",
              "hippMapp3rInitial",
              "hippMapp3rRefine",
              "koniqMBCS",
              "koniqMS",
              "koniqMS2",
              "koniqMS3",
              "mriSuperResolution",
              "protonLungMri",
              "sixTissueOctantBrainSegmentation",
              "sysuMediaWmhFlairOnlyModel0",
              "sysuMediaWmhFlairOnlyModel1",
              "sysuMediaWmhFlairOnlyModel2",
              "sysuMediaWmhFlairT1Model0",
              "sysuMediaWmhFlairT1Model1",
              "sysuMediaWmhFlairT1Model2",
              "tidsQualityAssessment",
	            "wholeTumorSegmentationT2Flair" ),
  targetFileName, antsxnetCacheDirectory = NULL )
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
    brainAgeDeepBrainNet = "https://ndownloader.figshare.com/files/23573402",
    brainExtraction = "https://ndownloader.figshare.com/files/22944632",
    brainExtractionT2 = "https://ndownloader.figshare.com/files/23066153",
    brainExtractionFLAIR = "https://ndownloader.figshare.com/files/23562194",
    brainExtractionBOLD = "https://ndownloader.figshare.com/files/22761977",
    brainExtractionFA = "https://ndownloader.figshare.com/files/22761926",
    brainExtractionNoBrainer = "https://ndownloader.figshare.com/files/22598039",
    brainExtractionInfantT1T2 = "https://ndownloader.figshare.com/files/22968833",
    brainExtractionInfantT1 = "https://ndownloader.figshare.com/files/22968836",
    brainExtractionInfantT2 = "https://ndownloader.figshare.com/files/22968830",
    brainSegmentation = "https://ndownloader.figshare.com/files/13900010",
    brainSegmentationPatchBased = "https://ndownloader.figshare.com/files/14249717",
    ctHumanLung = "https://ndownloader.figshare.com/files/20005217",
    dbpn4x = "https://ndownloader.figshare.com/files/13347617",
    deepFlash = "https://ndownloader.figshare.com/files/22933757",
    deepFlashLeft8 = "https://ndownloader.figshare.com/files/25441007",
    deepFlashRight8 = "https://ndownloader.figshare.com/files/25441004",
    deepFlashLeft16 = "https://ndownloader.figshare.com/files/25465844",
    deepFlashRight16 = "https://ndownloader.figshare.com/files/25465847",
    deepFlashLeft16new = "https://ndownloader.figshare.com/files/25991681",
    deepFlashRight16new = "https://ndownloader.figshare.com/files/25991678",
    denoising = "https://ndownloader.figshare.com/files/14235296",
    dktInner = "https://ndownloader.figshare.com/files/23266943",
    dktOuter = "https://ndownloader.figshare.com/files/23765132",
    dktOuterWithSpatialPriors = "https://ndownloader.figshare.com/files/24230768",
    elBicho = "https://ndownloader.figshare.com/files/26736779",
    ewDavidWmhSegmentationWeights = "https://ndownloader.figshare.com/files/26827691",
    # ewDavidWmhSegmentationSlicewiseWeights = "https://ndownloader.figshare.com/files/26787152",
    ewDavidWmhSegmentationSlicewiseWeights = "https://ndownloader.figshare.com/files/26896703",
    functionalLungMri = "https://ndownloader.figshare.com/files/13824167",
    hippMapp3rInitial = "https://ndownloader.figshare.com/files/18068408",
    hippMapp3rRefine = "https://ndownloader.figshare.com/files/18068411",
    koniqMBCS = "https://ndownloader.figshare.com/files/24967376",
    koniqMS = "https://ndownloader.figshare.com/files/25461887",
    koniqMS2 = "https://ndownloader.figshare.com/files/25474850",
    koniqMS3 = "https://ndownloader.figshare.com/files/25474847",
    mriSuperResolution = "https://ndownloader.figshare.com/files/24128618",
    protonLungMri = "https://ndownloader.figshare.com/files/13606799",
    sixTissueOctantBrainSegmentation = "https://ndownloader.figshare.com/files/23776025",
    sysuMediaWmhFlairOnlyModel0 = "https://ndownloader.figshare.com/files/22898441",
    sysuMediaWmhFlairOnlyModel1 = "https://ndownloader.figshare.com/files/22898570",
    sysuMediaWmhFlairOnlyModel2 = "https://ndownloader.figshare.com/files/22898438",
    sysuMediaWmhFlairT1Model0 = "https://ndownloader.figshare.com/files/22898450",
    sysuMediaWmhFlairT1Model1 = "https://ndownloader.figshare.com/files/22898453",
    sysuMediaWmhFlairT1Model2 = "https://ndownloader.figshare.com/files/22898459",
    tidsQualityAssessment = "https://ndownloader.figshare.com/files/24292895",
    wholeTumorSegmentationT2Flair = "https://ndownloader.figshare.com/files/14087045"
  )

  if( missing( targetFileName ) )
    {
    targetFileName <- paste0( fileId, ".h5" )
    }
  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- "ANTsXNet"
    }
  targetFileNamePath <- tensorflow::tf$keras$utils$get_file(
    targetFileName, url, cache_subdir = antsxnetCacheDirectory )
  return( targetFileNamePath )
}
