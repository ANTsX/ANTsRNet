testthat::context("AlexModels-3D")

testthat::test_that("Creating 3D Models", {
  if (keras::is_keras_available()) {
    model <- createAlexNetModel3D( inputImageSize = c(20L, 20L, 20L, 3L),
                                   numberOfClassificationLabels = 20,
                                   mode = "regression",
                                   batch_size = 1 )
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 647945492L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()

    model <- createAlexNetModel3D( inputImageSize = c(20L, 20L, 20L, 3L),
                                   numberOfClassificationLabels = 2,
                                   batch_size = 1)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 647871746L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()

    model <- createAlexNetModel3D( inputImageSize = c(20L, 20L, 20L, 3L),
                                   numberOfClassificationLabels = 3,
                                   batch_size = 1)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 647875843L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()

  }
})
