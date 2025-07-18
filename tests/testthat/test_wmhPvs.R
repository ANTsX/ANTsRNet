test_that("SYSU Media WMH segmentation runs", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/40251796", destfile = t1_file, mode = "wb")
  download.file("https://figshare.com/ndownloader/files/40251793", destfile = flair_file, mode = "wb")

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  wmh <- sysuMediaWMHSegmentation(flair, t1)
  expect_s4_class(wmh, "antsImage")
})

test_that("Hypermapp3r WMH segmentation runs", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/40251796", destfile = t1_file, mode = "wb")
  download.file("https://figshare.com/ndownloader/files/40251793", destfile = flair_file, mode = "wb")

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  wmh <- hypermapp3rSegmentation(t1, flair)
  expect_s4_class(wmh, "antsImage")
})

test_that("SHIVA WMH segmentation runs with all models", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/40251796", destfile = t1_file, mode = "wb")
  download.file("https://figshare.com/ndownloader/files/40251793", destfile = flair_file, mode = "wb")

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  wmh <- shivaWMHSegmentation(flair, t1, whichModel = "all")
  expect_s4_class(wmh, "antsImage")
})

test_that("SHIVA PVS segmentation runs with all models", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/48675367", destfile = t1_file, mode = "wb")
  download.file("https://figshare.com/ndownloader/files/48675352", destfile = flair_file, mode = "wb")

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  pvs <- shivaPVSSegmentation(t1, flair, whichModel = "all")
  expect_s4_class(pvs, "antsImage")
})
