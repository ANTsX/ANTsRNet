testthat::context("AlexModels")

testthat::test_that("Creating 2D Models", {
  if (keras::is_keras_available()) {
    model <- createAlexNetModel2D( inputImageSize = c(20L, 20L, 3L),
                                   numberOfClassificationLabels = 20,
                                   mode = "regression" )
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 123120596L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc()

    model <- createAlexNetModel2D( inputImageSize = c(20L, 20L, 3L),
                                   numberOfClassificationLabels = 2)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 123046850L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc()

    model <- createAlexNetModel2D( inputImageSize = c(20L, 20L, 3L),
                                   numberOfClassificationLabels = 3)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 123050947L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc()

  }
})

testthat::test_that("Creating 3D Models", {
  if (keras::is_keras_available()) {
    model <- createAlexNetModel3D( inputImageSize = c(20L, 20L, 20L, 3L),
                                   numberOfClassificationLabels = 20,
                                   mode = "regression" )
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 647945492L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc()
    model <- createAlexNetModel3D( inputImageSize = c(20L, 20L, 20L, 3L),
                                   numberOfClassificationLabels = 2)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 647871746L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc()
    model <- createAlexNetModel3D( inputImageSize = c(20L, 20L, 20L, 3L),
                                   numberOfClassificationLabels = 3)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 647875843L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc()

  }
})
