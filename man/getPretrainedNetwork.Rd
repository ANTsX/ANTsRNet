% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/getPretrainedNetwork.R
\name{getPretrainedNetwork}
\alias{getPretrainedNetwork}
\title{getPretrainedNetwork}
\usage{
getPretrainedNetwork(
  fileId = c("show", "antsxnetWmh", "antsxnetWmhOr", "arterialLesionWeibinShi",
    "brainAgeGender", "brainAgeFmrib", "brainAgeDeepBrainNet", "brainExtraction",
    "brainExtractionT1", "brainExtractionT1v1", "brainExtractionRobustT1",
    "brainExtractionT2", "brainExtractionRobustT2", "brainExtractionRobustT2Star",
    "brainExtractionFLAIR", "brainExtractionRobustFLAIR", "brainExtractionBOLD",
    "brainExtractionRobustBOLD", "brainExtractionFA", "brainExtractionRobustFA",
    "brainExtractionNoBrainer", "brainExtractionInfantT1T2", 
    
    "brainExtractionInfantT1", "brainExtractionInfantT2", "brainSegmentation",
    "brainSegmentationPatchBased", "bratsStage1", "bratsStage2", "cerebellumWhole",
    "cerebellumTissue", "cerebellumLabels", "claustrum_axial_0", "claustrum_axial_1",
    "claustrum_axial_2", "claustrum_coronal_0", "claustrum_coronal_1",
    "claustrum_coronal_2", "ctHumanLung", "dbpn4x", "deepFlash", "deepFlashLeftT1",
    "deepFlashRightT1", "deepFlashLeftBoth", "deepFlashRightBoth",
    "deepFlashLeftT1Hierarchical", "deepFlashRightT1Hierarchical", 
    
    "deepFlashLeftBothHierarchical", "deepFlashRightBothHierarchical",
    "deepFlashLeftT1Hierarchical_ri", "deepFlashRightT1Hierarchical_ri",
    "deepFlashLeftBothHierarchical_ri", "deepFlashRightBothHierarchical_ri",
    "deepFlashLeft8", "deepFlashRight8", "deepFlashLeft16", "deepFlashRight16",
    "deepFlashLeft16new", "deepFlashRight16new", "denoising", "dktInner", "dktOuter",
    "dktOuterWithSpatialPriors", "ex5_coronal_weights", "ex5_sagittal_weights",
    "allen_brain_mask_weights", "functionalLungMri", "hippMapp3rInitial", 
    
    "hippMapp3rRefine", "hyperMapp3r", "hypothalamus",
    "inpainting_sagittal_rmnet_weights", "inpainting_coronal_rmnet_weights",
    "inpainting_axial_rmnet_weights", "inpainting_axial_rmnet_flair_weights",
    "inpainting_coronal_rmnet_flair_weights", "inpainting_sagittal_rmnet_flair_weights",
    "koniqMBCS", "koniqMS", "koniqMS2", "koniqMS3", "lesion_whole_brain",
    "lungCtWithPriorsSegmentationWeights", "maskLobes", "mriSuperResolution",
    "mriModalityClassification", "mouseMriBrainExtraction", "mouseT2wBrainExtraction3D",
    
     "mouseT2wBrainParcellation3DNick", "protonLungMri", "protonLobes",
    "pulmonaryArteryWeights", "pulmonaryAirwayWeights",
    "sixTissueOctantBrainSegmentation", "sixTissueOctantBrainSegmentationWithPriors1",
    "sixTissueOctantBrainSegmentationWithPriors2", "sysuMediaWmhFlairOnlyModel0",
    "sysuMediaWmhFlairOnlyModel1", "sysuMediaWmhFlairOnlyModel2",
    "sysuMediaWmhFlairT1Model0", "sysuMediaWmhFlairT1Model1",
    "sysuMediaWmhFlairT1Model2", "tidsQualityAssessment", "xrayLungOrientation",
    "xrayLungExtraction", 
     "chexnetClassificationModel",
    "chexnetClassificationANTsXNetModel", "wholeHeadInpaintingFLAIR",
    "wholeHeadInpaintingPatchBasedT1", "wholeHeadInpaintingPatchBasedFLAIR",
    "wholeTumorSegmentationT2Flair", "wholeLungMaskFromVentilation"),
  targetFileName,
  antsxnetCacheDirectory = NULL
)
}
\arguments{
\item{fileId}{one of the permitted file ids or pass "show" to list all
valid possibilities. Note that most require internet access to download.}

\item{targetFileName}{optional target filename}

\item{antsxnetCacheDirectory}{destination directory for storing the downloaded
template and model weights.  Since these can be resused, if
\code{is.null(antsxnetCacheDirectory)}, these data will be downloaded to the
subdirectory ~/.keras/ANTsXNet/.}
}
\value{
filename string
}
\description{
Downloads pretrained network/weights.
}
\note{
See \url{https://figshare.com/authors/Nick_Tustison/441144}
or \url{https://figshare.com/authors/Brian_Avants/418551}
for some more descriptions
}
\examples{
\dontrun{
net <- getPretrainedNetwork( "dbpn4x" )
}
}
\author{
Avants BB
}
