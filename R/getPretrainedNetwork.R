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
              "denoising",
              "dktInner",
              "dktOuter",
              "dktOuterWithSpatialPriors",
              "functionalLungMri",
              "hippMapp3rInitial",
              "hippMapp3rRefine",
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
              "koniqMBCS",
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
    brainAgeDeepBrainNet = "https://ndownloader.figshare.com/files/23573402",
    brainExtraction = "https://ndownloader.figshare.com/files/22944632",
    # brainExtraction = "https://ndownloader.figshare.com/files/13729661",  # old weights
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
    denoising = "https://ndownloader.figshare.com/files/14235296",
    dktInner = "https://ndownloader.figshare.com/files/23266943",
    dktOuter = "https://ndownloader.figshare.com/files/23765132",
    dktOuterWithSpatialPriors = "https://ndownloader.figshare.com/files/24230768",
    functionalLungMri = "https://ndownloader.figshare.com/files/13824167",
    hippMapp3rInitial = "https://ndownloader.figshare.com/files/18068408",
    hippMapp3rRefine = "https://ndownloader.figshare.com/files/18068411",
    koniqMBCS = "https://ndownloader.figshare.com/files/24967013",
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
    targetFileName <- file.path( tempdir(), paste0( fileId, ".h5" ) )
    }

  if( ! file.exists( targetFileName ) || overwrite )
    {
    download.file( url, targetFileName, overwrite = overwrite )
    }
  return( targetFileName )
}
