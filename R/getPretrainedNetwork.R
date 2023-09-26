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
              "antsxnetWmh",
              "antsxnetWmhOr",
              "arterialLesionWeibinShi",
              "brainAgeGender",
              "brainAgeFmrib",
              "brainAgeDeepBrainNet",
              "brainExtraction",
              "brainExtractionT1",
              "brainExtractionT1v1",
              "brainExtractionRobustT1",
              "brainExtractionT2",
              "brainExtractionRobustT2",
              "brainExtractionRobustT2Star",
              "brainExtractionFLAIR",
              "brainExtractionRobustFLAIR",
              "brainExtractionBOLD",
              "brainExtractionRobustBOLD",
              "brainExtractionFA",
              "brainExtractionRobustFA",
              "brainExtractionNoBrainer",
              "brainExtractionInfantT1T2",
              "brainExtractionInfantT1",
              "brainExtractionInfantT2",
              "brainSegmentation",
              "brainSegmentationPatchBased",
              "bratsStage1",
              "bratsStage2",
              "cerebellumWhole",
              "cerebellumTissue",
              "cerebellumLabels",
              "claustrum_axial_0",
              "claustrum_axial_1",
              "claustrum_axial_2",
              "claustrum_coronal_0",
              "claustrum_coronal_1",
              "claustrum_coronal_2",
              "ctHumanLung",
              "dbpn4x",
              "deepFlash",
              "deepFlashLeftT1",
              "deepFlashRightT1",
              "deepFlashLeftBoth",
              "deepFlashRightBoth",
              "deepFlashLeftT1Hierarchical",
              "deepFlashRightT1Hierarchical",
              "deepFlashLeftBothHierarchical",
              "deepFlashRightBothHierarchical",
              "deepFlashLeftT1Hierarchical_ri",
              "deepFlashRightT1Hierarchical_ri",
              "deepFlashLeftBothHierarchical_ri",
              "deepFlashRightBothHierarchical_ri",
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
              "ex5_coronal_weights",
              "ex5_sagittal_weights",
              "allen_brain_mask_weights",
              "elBicho",
              "functionalLungMri",
              "hippMapp3rInitial",
              "hippMapp3rRefine",
              "hyperMapp3r",
              "hypothalamus",
              "koniqMBCS",
              "koniqMS",
              "koniqMS2",
              "koniqMS3",
              "lungCtWithPriorsSegmentationWeights",
              "maskLobes",
              "mriSuperResolution",
              "mriModalityClassification",
              "protonLungMri",
              "protonLobes",
              "sixTissueOctantBrainSegmentation",
              "sixTissueOctantBrainSegmentationWithPriors1",
              "sixTissueOctantBrainSegmentationWithPriors2",
              "sysuMediaWmhFlairOnlyModel0",
              "sysuMediaWmhFlairOnlyModel1",
              "sysuMediaWmhFlairOnlyModel2",
              "sysuMediaWmhFlairT1Model0",
              "sysuMediaWmhFlairT1Model1",
              "sysuMediaWmhFlairT1Model2",
              "tidsQualityAssessment",
              "xrayLungOrientation",
              "xrayLungExtraction",
              "chexnetClassificationModel",
              "chexnetClassificationANTsXNetModel",
              "wholeHeadInpaintingFLAIR",
              "wholeHeadInpaintingPatchBasedT1",
              "wholeHeadInpaintingPatchBasedFLAIR",
	             "wholeTumorSegmentationT2Flair",
              "wholeLungMaskFromVentilation" ),
  targetFileName, antsxnetCacheDirectory = NULL )
{


  if( fileId[1] == "show" )
    {
    return( fileId )
    }
  fileId = match.arg( fileId )

  url <- switch(
    fileId,
    antsxnetWmh = "https://figshare.com/ndownloader/files/42301059",
    antsxnetWmhOr = "https://figshare.com/ndownloader/files/42301056",
    arterialLesionWeibinShi = "https://figshare.com/ndownloader/files/31624922",
    brainAgeGender = "https://ndownloader.figshare.com/files/22179948",
    brainAgeFmrib = "https://ndownloader.figshare.com/files/22429077",
    brainAgeDeepBrainNet = "https://ndownloader.figshare.com/files/23573402",
    brainExtraction = "https://ndownloader.figshare.com/files/22944632",
    brainExtractionT1 = "https://ndownloader.figshare.com/files/27334370",
    brainExtractionT1v1 = "https://ndownloader.figshare.com/files/28057626",
    brainExtractionRobustT1 = "https://figshare.com/ndownloader/files/34821874",
    brainExtractionT2 = "https://ndownloader.figshare.com/files/23066153",
    brainExtractionRobustT2 = "https://figshare.com/ndownloader/files/34870416",
    brainExtractionRobustT2Star = "https://figshare.com/ndownloader/files/34870413",
    brainExtractionFLAIR = "https://ndownloader.figshare.com/files/23562194",
    brainExtractionRobustFLAIR = "https://figshare.com/ndownloader/files/34870407",
    brainExtractionBOLD = "https://ndownloader.figshare.com/files/22761977",
    brainExtractionRobustBOLD = "https://figshare.com/ndownloader/files/34870404",
    brainExtractionFA = "https://ndownloader.figshare.com/files/22761926",
    brainExtractionRobustFA = "https://figshare.com/ndownloader/files/34870410",
    brainExtractionNoBrainer = "https://ndownloader.figshare.com/files/22598039",
    brainExtractionInfantT1T2 = "https://ndownloader.figshare.com/files/22968833",
    brainExtractionInfantT1 = "https://ndownloader.figshare.com/files/22968836",
    brainExtractionInfantT2 = "https://ndownloader.figshare.com/files/22968830",
    brainSegmentation = "https://ndownloader.figshare.com/files/13900010",
    brainSegmentationPatchBased = "https://ndownloader.figshare.com/files/14249717",
    bratsStage1 = "https://figshare.com/ndownloader/files/42384756",
    bratsStage2 = "https://figshare.com/ndownloader/files/42384750",
    cerebellumWhole = "https://figshare.com/ndownloader/files/41460447",
    cerebellumTissue = "https://figshare.com/ndownloader/files/41107724",
    cerebellumLabels = "https://figshare.com/ndownloader/files/41168678",
    claustrum_axial_0 = "https://ndownloader.figshare.com/files/27844068",
    claustrum_axial_1 = "https://ndownloader.figshare.com/files/27844059",
    claustrum_axial_2 = "https://ndownloader.figshare.com/files/27844062",
    claustrum_coronal_0 = "https://ndownloader.figshare.com/files/27844074",
    claustrum_coronal_1 = "https://ndownloader.figshare.com/files/27844071",
    claustrum_coronal_2 = "https://ndownloader.figshare.com/files/27844065",
    ctHumanLung = "https://ndownloader.figshare.com/files/20005217",
    dbpn4x = "https://figshare.com/ndownloader/files/35295394",
    deepFlash = "https://ndownloader.figshare.com/files/22933757",
    deepFlashLeft8 = "https://ndownloader.figshare.com/files/25441007",
    deepFlashRight8 = "https://ndownloader.figshare.com/files/25441004",
    deepFlashLeft16 = "https://ndownloader.figshare.com/files/25465844",
    deepFlashRight16 = "https://ndownloader.figshare.com/files/25465847",
    deepFlashLeft16new = "https://ndownloader.figshare.com/files/25991681",
    deepFlashRight16new = "https://ndownloader.figshare.com/files/25991678",
    deepFlashLeftT1 = "https://ndownloader.figshare.com/files/28966269",
    deepFlashRightT1 = "https://ndownloader.figshare.com/files/28966266",
    deepFlashLeftBoth = "https://ndownloader.figshare.com/files/28966275",
    deepFlashRightBoth = "https://ndownloader.figshare.com/files/28966272",
    deepFlashLeftT1Hierarchical = "https://figshare.com/ndownloader/files/31226449",
    deepFlashRightT1Hierarchical = "https://figshare.com/ndownloader/files/31226452",
    deepFlashLeftBothHierarchical = "https://figshare.com/ndownloader/files/31226458",
    deepFlashRightBothHierarchical = "https://figshare.com/ndownloader/files/31226455",
    deepFlashLeftT1Hierarchical_ri = "https://figshare.com/ndownloader/files/33198794",
    deepFlashRightT1Hierarchical_ri = "https://figshare.com/ndownloader/files/33198800",
    deepFlashLeftBothHierarchical_ri = "https://figshare.com/ndownloader/files/33198803",
    deepFlashRightBothHierarchical_ri = "https://figshare.com/ndownloader/files/33198809",
    denoising = "https://ndownloader.figshare.com/files/14235296",
    dktInner = "https://ndownloader.figshare.com/files/23266943",
    dktOuter = "https://ndownloader.figshare.com/files/23765132",
    dktOuterWithSpatialPriors = "https://ndownloader.figshare.com/files/24230768",
    e13x5_coronal_weights = "https://figshare.com/ndownloader/files/38035968",
    e13x5_sagittal_weights = "https://figshare.com/ndownloader/files/38035965",
    allen_brain_mask_weights = "https://figshare.com/ndownloader/files/36999880",
    elBicho = "https://ndownloader.figshare.com/files/26736779",
    functionalLungMri = "https://ndownloader.figshare.com/files/13824167",
    hippMapp3rInitial = "https://ndownloader.figshare.com/files/18068408",
    hippMapp3rRefine = "https://ndownloader.figshare.com/files/18068411",
    hyperMapp3r = "https://figshare.com/ndownloader/files/38790702",
    hypothalamus = "https://ndownloader.figshare.com/files/28344378",
    koniqMBCS = "https://ndownloader.figshare.com/files/24967376",
    koniqMS = "https://figshare.com/ndownloader/files/35295403",
    koniqMS2 = "https://figshare.com/ndownloader/files/35295397",
    koniqMS3 = "https://figshare.com/ndownloader/files/35295400",
    lungCtWithPriorsSegmentationWeights = "https://ndownloader.figshare.com/files/28357818",
    maskLobes = "https://figshare.com/ndownloader/files/30678458",
    mriSuperResolution = "https://figshare.com/ndownloader/files/35290684",
    mriModalityClassification = "https://figshare.com/ndownloader/files/41691681",
    protonLungMri = "https://ndownloader.figshare.com/files/13606799",
    protonLobes = "https://figshare.com/ndownloader/files/30678455",
    sixTissueOctantBrainSegmentation = "https://ndownloader.figshare.com/files/23776025",
    sixTissueOctantBrainSegmentationWithPriors1 = "https://ndownloader.figshare.com/files/28159869",
    sysuMediaWmhFlairOnlyModel0 = "https://ndownloader.figshare.com/files/22898441",
    sysuMediaWmhFlairOnlyModel1 = "https://ndownloader.figshare.com/files/22898570",
    sysuMediaWmhFlairOnlyModel2 = "https://ndownloader.figshare.com/files/22898438",
    sysuMediaWmhFlairT1Model0 = "https://ndownloader.figshare.com/files/22898450",
    sysuMediaWmhFlairT1Model1 = "https://ndownloader.figshare.com/files/22898453",
    sysuMediaWmhFlairT1Model2 = "https://ndownloader.figshare.com/files/22898459",
    tidsQualityAssessment = "https://figshare.com/ndownloader/files/35295391",
    xrayLungOrientation = "https://figshare.com/ndownloader/files/41965821",
    xrayLungExtraction = "https://figshare.com/ndownloader/files/41965818",
    chexnetClassificationModel = "https://figshare.com/ndownloader/files/42460332",
    chexnetClassificationANTsXNetModel = "https://figshare.com/ndownloader/files/42460335",
    wholeHeadInpaintingT1 = "https://figshare.com/ndownloader/files/39255422",
    wholeHeadInpaintingFLAIR = "https://figshare.com/ndownloader/files/39255419",
    wholeHeadInpaintingPatchBasedT1 = "https://figshare.com/ndownloader/files/39337442",
    wholeHeadInpaintingPatchBasedFLAIR = "https://figshare.com/ndownloader/files/39337439",
    wholeTumorSegmentationT2Flair = "https://ndownloader.figshare.com/files/14087045",
    wholeLungMaskFromVentilation = "https://ndownloader.figshare.com/files/28914441"
  )

  if( missing( targetFileName ) )
    {
    targetFileName <- paste0( fileId, ".h5" )
    }
  if( is.null( antsxnetCacheDirectory ) )
    {
    antsxnetCacheDirectory <- fs::path_join( path.expand( c( "~/.keras/ANTsXNet" ) ) )
    }
  targetFileNamePath <- fs::path_join( path.expand( c( antsxnetCacheDirectory, targetFileName ) ) )

  if( ! fs::file_exists( targetFileNamePath ) )
    {
    targetFileNamePath <- tensorflow::tf$keras$utils$get_file(
      targetFileName, url, cache_subdir = antsxnetCacheDirectory )
    }

  return( targetFileNamePath )
}
