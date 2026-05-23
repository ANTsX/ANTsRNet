#' Supervised grader / scoring of T1-weighted brain MRI.
#'
#' @param image raw 3-D T1 brain image.
#' @param verbose print progress to the screen.
#' @return data frame with letter grade and numeric score predictions.
#' @author Tustison NJ
#' @examples
#' \dontrun{
#' library( ANTsRNet )
#' image <- antsImageRead( getANTsXNetData( "mprage_hippmapp3r" ) )
#' gradeDf <- t1Grader( image )
#' }
#' @import keras
#' @export
t1Grader <- function( image,
                      verbose = FALSE )
  {

  if( image@dimension != 3 )
    {
    stop( "Image dimension must be 3." )
    }

  ################################
  #
  # Prétraitement et Recalage
  #
  ################################

  t1 <- iMath( image - min( image ), "Normalize" )
  bxt <- thresholdImage( t1, 0.01, 1.0 )
  t1 <- rankIntensity( t1, mask = bxt )

  templatebFile <- getANTsXNetData( "S_template3_brain" )
  templateb <- antsImageRead( templatebFile )
  templateb <- resampleImage( cropImage( templateb ), c( 1, 1, 1 ) )
  templateb <- padImageByFactor( templateb, 8 )
  templatebsmall <- resampleImage( templateb, c( 2, 2, 2 ) )

  reg <- antsRegistration( fixed = templatebsmall, moving = t1,
                           typeofTransform = "Similarity", verbose = verbose )

  ilist <- list( list( templateb ) )
  nsim <- 16

  uu <- randomlyTransformImageData( templateb, ilist,
                                    numberOfSimulations = nsim,
                                    transformType = "scaleShear",
                                    sdAffine = 0.075 )

  fwdaffgd <- readAntsrTransform( reg$fwdtransforms[1] )

  scoreNums <- matrix( c( 3, 2, 1, 0 ), ncol = 1 )

  ################################
  #
  # Chargement du modèle et des poids (Keras)
  #
  ################################

  weightsFileName <- getPretrainedNetwork( "resnet_grader" )

  model <- createResNetModel3D( inputImageSize = c( dim( templateb ), 1 ),
                                numberOfOutputs = 4, 
                                mode = "classification",
                                lowestResolution = 32,
                                cardinality = 1,
                                squeezeAndExcite = FALSE )

  model$load_weights( weightsFileName )

  getGrade <- function( score, probs )
    {
    grade <- 'f'
    if( score >= 2.25 ) {
      grade <- 'a'
    } else if( score >= 1.5 ) {
      grade <- 'b'
    } else if( score >= 0.75 ) {
      grade <- 'c'
    }

    probgradeindex <- which.max( probs )
    probgrade <- c( 'a', 'b', 'c', 'f' )[probgradeindex]
    
    return( list( grade = grade, probgrade = probgrade, score = as.numeric( score ) ) )
    }

  gradelistNum <- c()
  gradelistProb <- c()
  gradeScore <- c()

  for( k in seq_len( nsim ) )
    {
    simtx <- uu$simulatedTransforms[[k]]
    
    # Composition de la transformation aléatoire et du recalage
    cmptx <- composeAntsrTransforms( list( simtx, fwdaffgd ) )
    
    subjectsim <- applyAntsrTransformToImage( cmptx, t1, templateb, interpolation = "linear" )
    subjectsim <- addNoiseToImage( subjectsim, noiseModel = "additivegaussian", noiseParameters = c( 0, 0.01 ) )

    batchX <- array( data = 0, dim = c( 1, dim( templateb ), 1 ) )
    batchX[1, , , , 1] <- as.array( subjectsim )

    preds <- model %>% predict( batchX, verbose = verbose )

    predsnum <- preds %*% scoreNums
    locgrades <- getGrade( predsnum[1, 1], preds[1, ] )

    gradelistNum <- c( gradelistNum, locgrades$grade )
    gradelistProb <- c( gradelistProb, locgrades$probgrade )
    gradeScore <- c( gradeScore, locgrades$score )
    }

  mostFrequent <- function( x )
    {
    ux <- unique( x )
    ux[which.max( tabulate( match( x, ux ) ) )]
    }

  mydf <- data.frame(
    NumericGrade = gradelistNum,
    ProbGrade = gradelistProb,
    NumericScore = gradeScore,
    grade = mostFrequent( gradelistProb )
  )

  smalldf <- data.frame(
    gradeLetter = mydf$grade[1],
    gradeNum = mean( mydf$NumericScore )
  )

  return( smalldf )
  }