





controlObject <- trainControl(method = "boot", number = 10 , 
                              summaryFunction = twoClassSummary , classProbs = TRUE)

model.1 <- train( x = Xtrain_quant.reduced , y = ytrain.cat , 
                           method = "glm", metric = "ROC", trControl = controlObject)

model.2 <- train( x = Xtrain_quant.reduced , y = ytrain.cat,  
                method = "lda", metric = "ROC" , trControl = controlObject)

model.3 <- train( x = Xtrain_quant.reduced , y = ytrain.cat,  
                  method = "pls", tuneGrid = expand.grid(.ncomp = 1:10), 
                  metric = "ROC" , trControl = controlObject)

ll = list(M1 = model.1, M2 = model.2 , M3 = model.3 )
##ll = list(model.1, model.2 , model.3 )


mod = c(model.1)
mod = c(mod,model.2)
mod = c(mod,model.3)

name.mod = c("M1")
name.mod = c(name.mod,"M2")
name.mod = c(name.mod,"M3")

ll = list(mod )
names(ll) = name.mod

cvValues <- resamples( ll )
summary(cvValues)
splom(cvValues, metric = "ROC")
xyplot(cvValues, metric = "ROC")  ### <<<<<<<<<<<<<<<<<<<<<<<<<<--------------------- il modello in alto ? il migliore 
parallelplot(cvValues, metric = "ROC")
dotplot(cvValues, metric = "ROC")
rocDiffs <- diff(cvValues, metric = "ROC")
summary(rocDiffs)
dotplot(rocDiffs, metric = "ROC")

