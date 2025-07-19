library(httr)

download_with_user_agent <- function(url, destfile) {
  res <- GET(url, write_disk(destfile, overwrite = TRUE), user_agent("Mozilla/5.0"))
  stop_for_status(res)
  return(destfile)
}

test_that("mouse brain extraction (T2) runs correctly", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  mouse_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/45289309", mouse_file)
  mouse <- antsImageRead(mouse_file)

  mouse_n4 <- n4BiasFieldCorrection(mouse,
                                    rescaleIntensities = TRUE,
                                    shrinkFactor = 2,
                                    convergence = list(iters = c(50, 50, 50, 50), tol = 0.0),
                                    splineParam = 20)

  mask <- mouseBrainExtraction(mouse_n4, modality = "t2")
  expect_s4_class(mask, "antsImage")
})

test_that("mouse brain parcellation (nick and tct) works", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  mouse_file <- tempfile(fileext = ".nii.gz")
  download_with_user_agent("https://figshare.com/ndownloader/files/45289309", mouse_file)
  mouse <- antsImageRead(mouse_file)

  mouse_n4 <- n4BiasFieldCorrection(mouse,
                                    rescaleIntensities = TRUE,
                                    shrinkFactor = 2,
                                    convergence = list(iters = c(50, 50, 50, 50), tol = 0.0),
                                    splineParam = 20)

  parc_nick <- mouseBrainParcellation(mouse_n4, mask = NULL,
                                      whichParcellation = "nick",
                                      returnIsotropicOutput = TRUE)

  parc_tct <- mouseBrainParcellation(mouse_n4, mask = NULL,
                                     whichParcellation = "tct",
                                     returnIsotropicOutput = TRUE)

  expect_s4_class(parc_nick$segmentationImage, "antsImage")
  expect_s4_class(parc_tct$segmentationImage, "antsImage")
})

