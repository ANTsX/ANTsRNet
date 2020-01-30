library(testthat)
library(ANTsRNet)

have_keras = keras::is_keras_available()
test_check("ANTsRNet")
