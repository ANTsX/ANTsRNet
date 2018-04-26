#' @export

unetImageBatchGenerator <- R6::R6Class( "UnetImageBatchGenerator",

  public = list( 
    
    imageList = NULL,

    segmentationList = NULL,

    transformList = NULL,

    referenceImageList = NULL,

    referenceTransformList = NULL,

    pairwiseIndices = NULL,

    initialize = function( imageList = NULL, segmentationList = NULL, 
      transformList = NULL, referenceImageList = NULL, 
      referenceTransformList = NULL )
      {
        
      if( !usePkg( "ANTsR" ) )
        {
        stop( "Please install the ANTsR package." )
        }

      if( !is.null( imageList ) )
        {
        self$imageList <- imageList
        } else {
        stop( "Input images must be specified." )
        }

      if( !is.null( segmentationList ) )
        {
        self$segmentationList <- segmentationList
        } else {
        stop( "Input segmentation images must be specified." )
        }

      if( !is.null( transformList ) )
        {
        self$transformList <- transformList
        } else {
        stop( "Input transforms must be specified." )
        }

      if( is.null( referenceImageList ) || 
        is.null( referenceTransformList ) )
        {
        self$referenceImageList <- imageList
        self$referenceTransformList <- transformList
        } else {
        self$referenceImageList <- referenceImageList
        self$referenceTransformList <- referenceTransformList
        }

      self$pairwiseIndices <- expand.grid( source = 1:length( self$imageList ), 
        reference = 1:length( self$referenceImageList ) )  

      },

    generate = function( batchSize = 32L )    
      {

      # shuffle the source data
      sampleIndices <- sample( length( self$imageList ) )
      self$imageList <- self$imageList[sampleIndices]
      self$segmentationList <- self$segmentationList[sampleIndices]
      self$transformList <- self$transformList[sampleIndices]

      # shuffle the reference data
      sampleIndices <- sample( length( self$referenceImageList ) )
      self$referenceImageList <- self$referenceImageList[sampleIndices]
      self$referenceTransformList <- self$referenceTransformList[sampleIndices]

      currentPassCount <- 1L

      function() 
        {
        # Shuffle the data after each complete pass 

        if( currentPassCount >= nrow( self$pairwiseIndices ) )
          {
          # shuffle the source data
          sampleIndices <- sample( length( self$imageList ) )
          self$imageList <- self$imageList[sampleIndices]
          self$segmentationList <- self$segmentationList[sampleIndices]
          self$transformList <- self$transformList[sampleIndices]

          # shuffle the reference data
          sampleIndices <- sample( length( self$referenceImageList ) )
          self$referenceImageList <- self$referenceImageList[sampleIndices]
          self$referenceTransformList <- self$referenceTransformList[sampleIndices]

          currentPassCount <- 1L
          }

        rowIndices <- currentPassCount + 0:( batchSize - 1L )

        outOfBoundsIndices <- which( rowIndices > nrow( self$pairwiseIndices ) )
        while( length( outOfBoundsIndices ) > 0 )
          {
          rowIndices[outOfBoundsIndices] <- rowIndices[outOfBoundsIndices] - 
            nrow( self$pairwiseIndices )
          outOfBoundsIndices <- which( rowIndices > nrow( self$pairwiseIndices ) )
          }
        batchIndices <- self$pairwiseIndices[rowIndices,]

        batchImages <- self$imageList[batchIndices$source]
        batchSegmentations <- self$segmentationList[batchIndices$source]
        batchTransforms <- self$transformList[batchIndices$source]

        batchReferenceImages <- self$referenceImageList[batchIndices$reference]
        batchReferenceTransforms <- self$referenceTransformList[batchIndices$reference]

        imageSize <- dim( batchImages[[1]] )

        batchX <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )
        batchY <- array( data = 0, dim = c( batchSize, imageSize ) )

        currentPassCount <<- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          sourceX <- batchImages[[i]]
          sourceY <- batchSegmentations[[i]]
          sourceXfrm <- batchTransforms[[i]]

          referenceX <- batchReferenceImages[[i]]
          referenceXfrm <- batchReferenceTransforms[[i]]

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- c( referenceXfrm$invtransforms[1], 
            referenceXfrm$invtransforms[2], sourceXfrm$fwdtransforms[1],
            sourceXfrm$fwdtransforms[2] )

          warpedImageX <- antsApplyTransforms( referenceX, sourceX, 
            interpolator = "linear", transformlist = transforms,
            whichtoinvert = boolInvert )          
          warpedImageY <- antsApplyTransforms( referenceX, sourceY, 
            interpolator = "nearestNeighbor", transformlist = transforms,
            whichtoinvert = boolInvert )

          warpedArrayX <- as.array( warpedImageX )
          warpedArrayX <- ( warpedArrayX - min( as.vector( warpedArrayX ) ) ) / 
            ( max( as.vector( warpedArrayX ) ) - min( as.vector( warpedArrayX ) ) )
          warpedArrayY <- as.array( warpedImageY )

          warpedArrayX3D <- abind( warpedArrayX, warpedArrayX, warpedArrayX, along = 3 )
          warpedArrayY3D <- abind( warpedArrayY, warpedArrayY, warpedArrayY, along = 3 )

          if( runif( 1 ) < 0.5 )  
            {
            warpedArrayX3D <- image_read( warpedArrayX3D ) %>%
              image_flop() %>%  
              .[[1]] %>% as.numeric()
            warpedArrayY3D <- image_read( warpedArrayY3D ) %>%
              image_flop() %>%  
              .[[1]] %>% as.numeric()
            }

          rotateAngleInDegrees <- runif( 1, 0, 359 )

          warpedArrayX3D <- image_read( warpedArrayX3D ) %>%
            image_negate() %>% 
            image_rotate( rotateAngleInDegrees ) %>%  
            image_negate() %>% 
            .[[1]] %>% as.numeric()
          warpedArrayY3D <- image_read( warpedArrayY3D ) %>%
            image_negate() %>% 
            image_rotate( rotateAngleInDegrees ) %>%  
            image_negate() %>% 
            .[[1]] %>% as.numeric()

          # Crop to original size
          dimOriginal <- dim( warpedArrayX )
          dimRotated <- dim( warpedArrayX3D )
          geometry <- c()
          geometry[1] <- dimOriginal[1]
          geometry[2] <- dimOriginal[2]
          geometry[3] <- round( 0.25 * ( dimRotated[1] - dimOriginal[1] ) )
          geometry[4] <- round( 0.25 * ( dimRotated[2] - dimOriginal[2] ) )

          geometryString <- paste0( geometry[1], 'x', geometry[2], 
            '+', geometry[3], '+', geometry[4] )

          warpedArrayX3D <- image_read( warpedArrayX3D ) %>%
            image_crop( geometryString ) %>%  
            .[[1]] %>% as.numeric()
          warpedArrayY3D <- image_read( warpedArrayY3D ) %>%
            image_crop( geometryString ) %>%  
            .[[1]] %>% as.numeric()

          warpedArrayX <- warpedArrayX3D[,,1]
          warpedArrayX <- ( warpedArrayX - mean( warpedArrayX ) ) / sd( warpedArrayX )

          warpedArrayY <- warpedArrayY3D[,,1]
          warpedArrayY[which( warpedArrayY < 0.5 )] <- 0
          warpedArrayY[which( warpedArrayY >= 0.5 )] <- 1

          # antsImageWrite( as.antsImage( warpedArrayX, reference = warpedImageX ), "~/Desktop/warpedX.nii.gz" )
          # antsImageWrite( as.antsImage( warpedArrayY, reference = warpedImageX ), "~/Desktop/warpedY.nii.gz" )

          # myPrompt <- paste0( "Angle = ", rotateAngleInDegrees, "\n" )
          # readline( prompt = myPrompt ) 

          batchX[i,,, 1] <- warpedArrayX
          batchY[i,,] <- warpedArrayY
          }

        segmentationLabels <- sort( unique( as.vector( batchY ) ) )

        encodedBatchY <- encodeY( batchY, segmentationLabels ) 

        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )