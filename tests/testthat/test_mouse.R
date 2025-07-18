test_that("mouse brain extraction (T2) runs correctly", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  mouse_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/45289309", destfile = mouse_file, mode = "wb")
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

  library(ANTsRNet)
  library(ANTsR)

  mouse_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/45289309", destfile = mouse_file, mode = "wb")
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

  expect_s4_class(parc_nick, "antsImage")
  expect_s4_class(parc_tct, "antsImage")
})

test_that("mouse cortical thickness returns image", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  mouse_file <- tempfile(fileext = ".nii.gz")
  download.file("https://figshare.com/ndownloader/files/45289309", destfile = mouse_file, mode = "wb")
  mouse <- antsImageRead(mouse_file)

  mouse_n4 <- n4BiasFieldCorrection(mouse,
                                    rescaleIntensities = TRUE,
                                    shrinkFactor = 2,
                                    convergence = list(iters = c(50, 50, 50, 50), tol = 0.0),
                                    splineParam = 20)

  thickness <- mouseCorticalThickness(mouse_n4,
                                      mask = NULL,
                                      returnIsotropicOutput = TRUE)

  expect_s4_class(thickness, "antsImage")
})
