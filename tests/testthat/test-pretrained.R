testthat::context("Downloading a pre-trained model")
testthat::test_that("mriSuperResolution loads", {
  res = getPretrainedNetwork("mriSuperResolution")
  testthat::expect_true(file.exists(res))
  model = keras::load_model_hdf5(res)
  testthat::expect_is(model, "keras.engine.training.Model" )
})
