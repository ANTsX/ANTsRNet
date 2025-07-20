test_that("brainExtraction runs correctly across modalities", {
  skip_on_cran()  # avoid download on CRAN
  skip_if_not_installed("ANTsRNet")
  skip_if_not_installed("ANTsR")

  library(ANTsR)
  library(ANTsRNet)

  # Download and read test image
  t1 <- antsImageRead(getANTsXNetData('mprage_hippmapp3r'))

  # Define modalities to test
  # modalities <- c("t1", "t1threetissue", "t1hemi", "t1lobes")
  modalities <- c("t1")

  for (mod in modalities) {
    bext <- brainExtraction(t1, modality = mod, verbose = FALSE)

    if (mod %in% c("t1")) {
      expect_s4_class(bext, "antsImage")
    } else {
      expect_type(bext, "list")
      expect_true("segmentationImage" %in% names(bext))
      expect_s4_class(bext$segmentationImage, "antsImage")
    }
  }
})
