test_that("MRI super-resolution runs and returns antsImage", {
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
  t1_lr <- resampleImage(t1, c(4, 4, 4), useVoxels = FALSE)
  t1_sr <- mriSuperResolution(t1_lr, expansionFactor = c(1, 1, 2))
  expect_s4_class(t1_sr, "antsImage")
})

test_that("T1w neural image QA returns numeric score", {
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
  qa_score <- tidNeuralImageAssessment(t1)
  expect_type(qa_score, "list")
})

test_that("PSNR and SSIM return valid similarity values", {
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  r16 <- antsImageRead(getANTsRData("r16"))
  r64 <- antsImageRead(getANTsRData("r64"))

  psnr_val <- PSNR(r16, r64)
  expect_equal(psnr_val, 10.37418, tolerance = 1e-3)
  ssim_val <- SSIM(r16, r64)
  expect_equal(ssim_val, 0.5654819, tolerance = 1e-3)
})
