testthat::context("Downloading a pre-trained model")
testthat::test_that("mriSuperResolution loads", {
  res = getPretrainedNetwork("mriSuperResolution")
  testthat::expect_true(file.exists(res))
  model = keras::load_model_hdf5(res)
  testthat::expect_is(model, "keras.engine.training.Model" )
})

# testthat::test_that("mriSuperResolution loads", {
#   all_files = getPretrainedNetwork()
#   all_files = setdiff(all_files, c("show", "mriSuperResolution"))
#   all_files = c("ctHumanLung")
#   all_networks = sapply(all_files, getPretrainedNetwork)
#   testthat::expect_true(all(file.exists(all_networks)))
#   keras::load_model_hdf5(all_networks[1])
#   models = lapply(all_networks, keras::load_model_hdf5)
#   model = keras::load_model_hdf5(res)
#   testthat::expect_is(model, "keras.engine.training.Model" )
# })
