#' @export

ssdImageBatchGenerator3D <- R6::R6Class( "SsdImageBatchGenerator3D",

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
      if( !usePkg( "abind" ) )
        {
        stop( "Please install the abind package." )
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

    generate = function( batchSize = 32L, paddingSize = NULL, 
      anchorBoxes = NULL, variances = rep( 1.0, 6 ) )    
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

        referenceX <- antsImageRead( batchReferenceImages[[1]], dimension = 3 )
        imageSize <- dim( referenceX )
        channelSize <- length( batchImages[[1]] )

        if( !is.null( paddingSize ) )
          {
          batchX <- array( data = 0, dim = c( batchSize, paddingSize + imageSize, channelSize ) )
          } else {
          batchX <- array( data = 0, dim = c( batchSize, imageSize, channelSize ) )
          }
        batchY <- list()

        currentPassCount <<- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          subjectBatchImages <- batchImages[[i]]  

          referenceX <- antsImageRead( batchReferenceImages[[i]], dimension = 3 )
          referenceXfrm <- batchReferenceTransforms[[i]]

          sourceXfrm <- batchTransforms[[i]]

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- c( referenceXfrm$invtransforms[1], 
            referenceXfrm$invtransforms[2], sourceXfrm$fwdtransforms[1],
            sourceXfrm$fwdtransforms[2] )

          for( j in seq_len( channelSize ) )
            {  
            sourceX <- antsImageRead( subjectBatchImages[j], dimension = 3 )

            warpedImageX <- antsApplyTransforms( referenceX, sourceX, 
              interpolator = "linear", transformlist = transforms,
              whichtoinvert = boolInvert )          

            warpedArrayX <- as.array( warpedImageX )
            warpedArrayX <- ( warpedArrayX - mean( as.vector( warpedArrayX ) ) ) / 
              sd( as.vector( warpedArrayX ) )

            if( !is.null( paddingSize ) )
              {
              for( d in 1:length( paddingSize ) )
                {
                if( paddingSize[d] > 0 )  
                  {
                  paddingSizeDim <- dim( warpedArrayX )
                  paddingSizeDim[d] <- paddingSize[d]  
                  zerosArray <- array( 0, dim = c( paddingSizeDim ) )  
                  warpedArrayX <- abind( warpedArrayX, zerosArray, along = d )
                  }
                }
              }  

            batchX[i,,,,j] <- warpedArrayX
            }
          sourceY <- antsImageRead( batchSegmentations[[i]], dimension = 3 )

          warpedImageY <- antsApplyTransforms( referenceX, sourceY, 
            interpolator = "nearestNeighbor", transformlist = transforms,
            whichtoinvert = boolInvert  )

          measures <- labelGeometryMeasures( warpedImageY )

          #
          # Note:  need to change the order when ANTsR updates it's ANTs version
          #
          boxes <- data.frame( classId = rep( 1, nrow( measures ) ),
            xmin = measures$BoundingBoxLower_x, xmax = measures$BoundingBoxLower_y, 
            ymin = measures$BoundingBoxLower_z, ymax = measures$BoundingBoxUpper_x, 
            zmin = measures$BoundingBoxUpper_y, zmax = measures$BoundingBoxUpper_z ) 
          batchY[[i]] <- boxes
          }

        encodedBatchY <- encodeY3D( batchY, anchorBoxes, imageSize, variances )  
        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )