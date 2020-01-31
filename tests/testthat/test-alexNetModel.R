testthat::context("AlexModels-2D")

testthat::test_that("Creating 2D Models", {
  if (keras::is_keras_available()) {
    model <- createAlexNetModel2D( inputImageSize = c(20L, 20L, 1L),
                                   numberOfClassificationLabels = 2,
                                   batch_size = 1)
    cat("First Model is done")
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 123023618L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()


    model <- createAlexNetModel2D( inputImageSize = c(20L, 20L, 1L),
                                   numberOfClassificationLabels = 3,
                                   batch_size = 1)
    cat("Second Model is done")
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 123027715L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()


    model <- createAlexNetModel2D( inputImageSize = c(20L, 20L, 1L),
                                   numberOfClassificationLabels = 10,
                                   mode = "regression",
                                   batch_size = 1)
    testthat::expect_is(model, "keras.engine.training.Model" )
    testthat::expect_equal(model$count_params(), 123056394L)
    testthat::expect_equal(length(model$weights), 16L)
    rm(model); gc(); gc()
    Sys.sleep(2); gc(); gc()


  }
})
