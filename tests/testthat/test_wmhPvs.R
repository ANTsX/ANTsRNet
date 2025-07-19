library(httr)

download_with_user_agent <- function(url, destfile) {
  res <- GET(url, write_disk(destfile, overwrite = TRUE), user_agent("Mozilla/5.0"))
  stop_for_status(res)
  return(destfile)
}

test_that("SYSU Media WMH segmentation runs", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/40251796", t1_file)
  download_with_user_agent("https://figshare.com/ndownloader/files/40251793", flair_file)

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  wmh <- sysuMediaWmhSegmentation(flair, t1)
  expect_s4_class(wmh, "antsImage")
})

test_that("Hypermapp3r WMH segmentation runs", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/40251796", t1_file)
  download_with_user_agent("https://figshare.com/ndownloader/files/40251793", flair_file)

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  wmh <- hyperMapp3rSegmentation(t1, flair)
  expect_s4_class(wmh, "antsImage")
})

test_that("SHIVA WMH segmentation runs with all models", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/40251796", t1_file)
  download_with_user_agent("https://figshare.com/ndownloader/files/40251793", flair_file)

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  wmh <- shivaWmhSegmentation(flair, t1, whichModel = "all")
  expect_s4_class(wmh, "antsImage")
})

test_that("SHIVA PVS segmentation runs with all models", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsR); library(ANTsRNet)

  t1_file <- tempfile(fileext = ".nii.gz")
  flair_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/48675367", t1_file)
  download_with_user_agent("https://figshare.com/ndownloader/files/48675352", flair_file)

  t1 <- antsImageRead(t1_file)
  flair <- antsImageRead(flair_file)

  pvs <- shivaPvsSegmentation(t1, flair, whichModel = "all")
  expect_s4_class(pvs, "antsImage")
})
