testthat::context("AlexModels-3D")

testthat::test_that("Creating 3D Models", {
  if (keras::is_keras_available()) {
    model <- createAlexNetModel3D(
      inputImageSize = c(20L, 20L, 19L, 1L),
      numberOfClassificationLabels = 2,
      numberOfDenseUnits = 256,
      batch_size = 1)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 46963394L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()

    model <- createAlexNetModel3D(
      inputImageSize = c(20L, 20L, 20L, 1L),
      numberOfClassificationLabels = 3,
      numberOfDenseUnits = 256,
      batch_size = 1)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 46963651L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()

    model <- createAlexNetModel3D(
      inputImageSize = c(20L, 20L, 20L, 1L),
      numberOfClassificationLabels = 2,
      mode = "regression",
      numberOfDenseUnits = 256,
      batch_size = 1 )
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 46963394L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()

  }
})


testthat::test_that("Creating Big 3D Models", {
  if (keras::is_keras_available()) {
    testthat::skip_on_travis()
    model <- createAlexNetModel3D(
      inputImageSize = c(20L, 20L, 20L, 1L),
      numberOfClassificationLabels = 2,
      mode = "regression",
      numberOfDenseUnits = 1024,
      batch_size = 1 )
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 164734658L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()
  }
})
