# test_that("deepAtropos runs with default input", {
# #   skip_if_not_installed("ANTsRNet")
#   skip_if_not_installed("ANTsR")
#   library(ANTsRNet); library(ANTsR)

#   t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
#   seg <- deepAtropos(t1, verbose = FALSE)

#   expect_type(seg, "list")
#   expect_s4_class(seg$segmentationImage, "antsImage")
# })

# test_that("deepAtropos accepts list input with NULLs", {
# #   skip_if_not_installed("ANTsRNet")
#   skip_if_not_installed("ANTsR")
#   library(ANTsRNet); library(ANTsR)

#   t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
#   seg <- deepAtropos(list(t1, NULL, NULL), verbose = FALSE)

#   expect_type(seg, "list")
#   expect_s4_class(seg$segmentationImage, "antsImage")
# })

# test_that("DKT labeling versions 0 and 1 work", {
# #   skip_if_not_installed("ANTsRNet")
#   skip_if_not_installed("ANTsR")
#   library(ANTsRNet); library(ANTsR)

#   t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
#   dkt0 <- desikanKillianyTourvilleLabeling(t1, version = 0)
#   dkt1 <- desikanKillianyTourvilleLabeling(t1, version = 1)

#   expect_s4_class(dkt0, "antsImage")
#   expect_s4_class(dkt1, "antsImage")
# })

# test_that("Harvard-Oxford Atlas labeling works", {
# #   skip_if_not_installed("ANTsRNet")
#   skip_if_not_installed("ANTsR")
#   library(ANTsRNet); library(ANTsR)

#   t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
#   hoa <- harvardOxfordAtlasLabeling(t1)
# })

test_that("deepFlash returns expected segmentation", {
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
  df <- deepFlash(t1, verbose = FALSE)
  expect_type(df, "list")
})

test_that("claustrum segmentation runs", {
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")
  library(ANTsRNet); library(ANTsR)

  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))
  seg <- claustrumSegmentation(t1)
  expect_s4_class(seg, "antsImage")
})
