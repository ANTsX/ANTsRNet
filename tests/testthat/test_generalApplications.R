test_that("MRI super-resolution runs and returns antsImage", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
  t1_lr <- resampleImage(t1, c(4, 4, 4), useVoxels = FALSE)
  t1_sr <- mriSuperResolution(t1_lr, expansionFactors = c(1, 1, 2))
  expect_s4_class(t1_sr, "antsImage")
})

test_that("T1w neural image QA returns numeric score", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
  qa_score <- tidNeuralImageAssessment(t1)
  expect_type(qa_score, "double")
  expect_length(qa_score, 1)
})

test_that("PSNR and SSIM return valid similarity values", {
  skip_on_cran()
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsRNet)
  library(ANTsR)

  r16 <- antsImageRead(getANTsRData("r16"))
  r64 <- antsImageRead(getANTsRData("r64"))

  psnr_val <- psnr(r16, r64)
  ssim_val <- ssim(r16, r64)

  expect_type(psnr_val, "double")
  expect_type(ssim_val, "double")
  expect_length(psnr_val, 1)
  expect_length(ssim_val, 1)
})
