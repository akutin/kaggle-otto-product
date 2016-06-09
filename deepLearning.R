install.packages("h2o")
library(h2o)

train <- read.csv("input/train.csv")

h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads = -1)

# split data set
set.seed(2909)
product.sample <- sample(1:nrow(train), 50000) # 80/20 split
product.train <- train[product.sample, 2:95]
product.test <- train[-product.sample, 2:95]
# plotting split
barplot( rbind(table(product.train$target),table(product.test$target)), beside = T)

# build model
product.h2o.train <- as.h2o(product.train)
product.model <- h2o.deeplearning(x = 1:93,  # column numbers for predictors
                   y = 94,   # column number for label
                   training_frame = product.h2o.train, # data in H2O format
                   activation = "Rectifier", # or 'Tanh'
                   input_dropout_ratio = 0.01, # % of inputs dropout
                   hidden_dropout_ratios = c(0.01,0.01,0.01), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(30,50,30), # three layers of 50 nodes
                   epochs = 50) # max. no. of epochs


# confusion matix
h2o.confusionMatrix(product.model, product.h2o.train)
h2o.confusionMatrix(product.model, product.h2o.test)

# predictions
product.h2o.test <- as.h2o(product.test)
product.predict.train <- h2o.predict(product.model, product.h2o.train)
product.predict.test <- h2o.predict(product.model, product.h2o.test)

h2o.shutdown(prompt = F)
