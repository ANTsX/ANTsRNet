test_that("lungExtraction works with CT modality", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet); library(ANTsR)

  ct_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934234", destfile = ct_file, mode = "wb")
  ct <- antsImageRead(ct_file)
  seg <- lungExtraction(ct, modality = "ct")

  expect_s4_class(seg, "antsImage")
})

test_that("lungExtraction works with proton and derived masks", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  proton_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934228", destfile = proton_file, mode = "wb")
  proton <- antsImageRead(proton_file)

  lobe_seg <- lungExtraction(proton, modality = "protonLobes")
  expect_type(lobe_seg, "list")
  expect_s4_class(lobe_seg$segmentation_image, "antsImage")

  lung_mask <- thresholdImage(lobe_seg$segmentation_image, 0, 0, 0, 1)
  lobes_from_mask <- lungExtraction(lung_mask, modality = "maskLobes")
  expect_type(lobes_from_mask, "list")
  expect_s4_class(lobes_from_mask$segmentation_image, "antsImage")
})

test_that("lungExtraction works with X-ray (CXR) input", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  cxr_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934237", destfile = cxr_file, mode = "wb")
  cxr <- antsImageRead(cxr_file)

  seg <- lungExtraction(cxr, modality = "xray")
  expect_s4_class(seg, "antsImage")
})

test_that("lungExtraction works with ventilation images", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  ventilation_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934231", destfile = ventilation_file, mode = "wb")
  vent <- antsImageRead(ventilation_file)

  seg <- lungExtraction(vent, modality = "ventilation")
  expect_s4_class(seg, "antsImage")
})

test_that("elBicho runs on ventilation + mask", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  proton_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934228", destfile = proton_file, mode = "wb")
  proton <- antsImageRead(proton_file)
  lung_seg <- lungExtraction(proton, modality = "proton")
  lung_mask <- thresholdImage(lung_seg$segmentation_image, 0, 0, 0, 1)

  ventilation_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934231", destfile = ventilation_file, mode = "wb")
  vent <- antsImageRead(ventilation_file)

  eb <- elBicho(vent, lung_mask)
  expect_s4_class(eb, "antsImage")
})

test_that("chexNet returns prediction scores with/without TB", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  cxr_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/42934237", destfile = cxr_file, mode = "wb")
  cxr <- antsImageRead(cxr_file)

  pred1 <- chexNet(cxr, useANTsXNetVariant = FALSE)
  pred2 <- chexNet(cxr, useANTsXNetVariant = TRUE, includeTuberculosisDiagnosis = TRUE)

  expect_type(pred1, "double")
  expect_type(pred2, "double")
})
