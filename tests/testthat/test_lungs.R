library(httr)

download_with_user_agent <- function(url, destfile) {
  res <- GET(url, write_disk(destfile, overwrite = TRUE), user_agent("Mozilla/5.0"))
  stop_for_status(res)
  return(destfile)
}

# test_that("lungExtraction works with CT modality", {
#   skip_on_cran()
#   skip_if_not_installed("ANTsRNet")
#   skip_if_not_installed("ANTsR")
#   library(ANTsRNet); library(ANTsR)

#   ct_file <- tempfile(fileext = ".nii.gz")
#   download_with_user_agent("https://figshare.com/ndownloader/files/42934234", ct_file)
#   ct <- antsImageRead(ct_file)
#   seg <- lungExtraction(ct, modality = "ct")

#   expect_s4_class(seg, "antsImage")
# })

test_that("lungExtraction works with proton and derived masks", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  proton_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/42934228", proton_file)
  proton <- antsImageRead(proton_file)

  lobe_seg <- lungExtraction(proton, modality = "protonLobes")
  expect_type(lobe_seg, "list")
  expect_s4_class(lobe_seg$segmentationImage, "antsImage")

  lung_mask <- thresholdImage(lobe_seg$segmentationImage, 0, 0, 0, 1)
  lobes_from_mask <- lungExtraction(lung_mask, modality = "maskLobes")
  expect_type(lobes_from_mask, "list")
  expect_s4_class(lobes_from_mask$segmentationImage, "antsImage")
})

test_that("lungExtraction works with X-ray (CXR) input", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  cxr_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/42934237", cxr_file)
  cxr <- antsImageRead(cxr_file)

  seg <- lungExtraction(cxr, modality = "xray")
  expect_s4_class(seg$segmentationImage, "antsImage")
})

test_that("lungExtraction works with ventilation images", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  vent_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/42934231", vent_file)
  vent <- antsImageRead(vent_file)

  seg <- lungExtraction(vent, modality = "ventilation")
  expect_s4_class(seg, "antsImage")
})

test_that("elBicho runs on ventilation + mask", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  proton_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/42934228", proton_file)
  proton <- antsImageRead(proton_file)
  lung_seg <- lungExtraction(proton, modality = "proton")
  lung_mask <- thresholdImage(lung_seg$segmentationImage, 0, 0, 0, 1)

  vent_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/42934231", vent_file)
  vent <- antsImageRead(vent_file)

  eb <- elBicho(vent, lung_mask)
  expect_s4_class(eb$segmentationImage, "antsImage")
})

test_that("chexNet returns prediction scores with/without TB", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  cxr_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/42934237", cxr_file)
  cxr <- antsImageRead(cxr_file)

  pred1 <- chexnet(cxr, useANTsXNetVariant = FALSE)
  pred2 <- chexnet(cxr, useANTsXNetVariant = TRUE, includeTuberculosisDiagnosis = FALSE)
  # pred3 <- chexnet(cxr, useANTsXNetVariant = TRUE, includeTuberculosisDiagnosis = TRUE)

  expect_type(pred1, "list")
  expect_type(pred2, "list")
})
