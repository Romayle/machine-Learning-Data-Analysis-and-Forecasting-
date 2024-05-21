library("neuralnet")
library("readxl")
# Read the data from the Excel file
data <- read_excel("data/ExchangeUSD.xlsx")
# Split the data into training and testing sets
train_data <- data[1:400, ]
test_data <- data[401:500, ]

# c)
# Normalize the data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
#training/testing
# normalizing done before io matrix created
train_data$`USD/EUR` <- normalize(train_data$`USD/EUR`)
test_data$`USD/EUR` <- normalize(test_data$`USD/EUR`)
evaluate_mlp <- function(input_size, hidden_layers) {
  
  
  # b) construct a matrix (I/O) for the MLP
  #train I/O Matrix
  train_data_with_lags <- data.frame(
    exchange_rate = train_data$`USD/EUR`[(input_size + 1):nrow(train_data)],
    lag1 = train_data$`USD/EUR`[1:(nrow(train_data) - input_size)],
    lag2 = train_data$`USD/EUR`[2:(nrow(train_data) - input_size + 1)],
    lag3 = train_data$`USD/EUR`[3:(nrow(train_data) - input_size + 2)],
    lag4 = train_data$`USD/EUR`[4:(nrow(train_data) - input_size + 3)],
    lag5 = train_data$`USD/EUR`[5:(nrow(train_data) - input_size + 4)]
  )
  
  
  train_data_with_lags <- train_data_with_lags[, 1:(input_size + 1)]
  
  # Create the MLP model
  mlp_model <- neuralnet(exchange_rate ~ ., data = train_data_with_lags, hidden = hidden_layers, 
                         linear.output = TRUE)
  
  # b) construct a matrix (I/O) for the MLP
  # testing I/O Matrix
  # Evaluate the model on the test data
  test_data_with_lags <- data.frame(
    exchange_rate = test_data$`USD/EUR`[(input_size + 1):nrow(test_data)],
    lag1 = test_data$`USD/EUR`[1:(nrow(test_data) - input_size)],
    lag2 = test_data$`USD/EUR`[2:(nrow(test_data) - input_size + 1)],
    lag3 = test_data$`USD/EUR`[3:(nrow(test_data) - input_size + 2)],
    lag4 = test_data$`USD/EUR`[4:(nrow(test_data) - input_size + 3)],
    lag5 = test_data$`USD/EUR`[5:(nrow(test_data) - input_size + 4)]
  )
  
  test_data_with_lags <- test_data_with_lags[, 1:(input_size + 1)]
  
  predicted_values <- compute(mlp_model, test_data_with_lags)$net.result
  
  
  #part D ------- For each case, the testing performance (i.e. evaluation) of the networks will be
  # calculated using the standard statistical indices (RMSE, MAE, MAPE and sMAPE – symmetric MAPE).
  # Calculate evaluation metrics
  rmse <- sqrt(mean((predicted_values - test_data_with_lags$exchange_rate)^2))
  mae <- mean(abs(predicted_values - test_data_with_lags$exchange_rate))
  
  mape <- mean(abs((predicted_values - test_data_with_lags$exchange_rate) / 
                     test_data_with_lags$exchange_rate)) * 100
  smape <- mean(abs(predicted_values - test_data_with_lags$exchange_rate) / ((abs(predicted_values) + 
                                                                                abs(test_data_with_lags$exchange_rate)) / 2)) * 100
  
  return(list(rmse = rmse, mae = mae, mape = mape, smape = smape))
}
# b)
# Experiment with different input vector sizes and network structures
input_sizes <- c(1, 2, 3, 4, 5)
hidden_layer_configs <- list(c(5), c(10), c(5, 3), c(5, 10), c(10, 5))
# f)
# Create a data frame to store the results
results <- data.frame(
  input_size = integer(),
  hidden_layers = character(),
  rmse = numeric(),
  mae = numeric(),
  mape = numeric(),
  smape = numeric(),
  description = character()
)
# Train and evaluate MLP models

model_count <- 0
for (input_size in input_sizes) {
  for (hidden_layers in hidden_layer_configs) {
    
    
    
    # f) restricting the total number of developed NNs to 12-15 models
    
    if (model_count >= 15) {
      break
    }
    
    
    
    # d) experimenting with various MLP models, utilising these different input vectors and various internal network structures (such as hidden layers, nodes, linear/nonlinear output, activation function, etc.).
    model_eval <- evaluate_mlp(input_size, hidden_layers)
    
    description <- paste("Input size:", input_size, "Hidden layers:", paste(hidden_layers, collapse = ", "))
    
    results <- rbind(results, data.frame(
      input_size = input_size,
      hidden_layers = paste(hidden_layers, collapse = ", "),
      rmse = model_eval$rmse,
      mae = model_eval$mae,
      mape = model_eval$mape,
      smape = model_eval$smape,
      description = description
      
    ))
    
    
    model_count <- model_count + 1
  }
}
# Print the comparison table
print(results)
# Calculate total number of weight parameters for each model
results$total_weights <- sapply(1:nrow(results), function(i) {
  hidden_layers <- as.numeric(strsplit(results$hidden_layers[i], ",")[[1]])
  if (length(hidden_layers) == 1) {
    total_weights <- (results$input_size[i] + 1) * hidden_layers + (hidden_layers + 1) * 1
  } else {
    total_weights <- (results$input_size[i] + 1) * hidden_layers[1] + 
      (hidden_layers[1] + 1) * hidden_layers[2] +
      (hidden_layers[2] + 1) * 1
  }
  return(sum(total_weights))
})
# Printed the results for my ease
print(results[, c("input_size", "hidden_layers", "total_weights")])
# g) Check the "efficiency" of the best one-hidden layer and two-hidden layer networks

one_hidden_layer_models <- results[sapply(strsplit(results$hidden_layers, ","), length) == 1, ]
two_hidden_layer_models <- results[sapply(strsplit(results$hidden_layers, ","), length) == 2, ]
print(one_hidden_layer_models)
print(two_hidden_layer_models)
best_one_hidden_layer <-
  one_hidden_layer_models[which.min(one_hidden_layer_models$total_weights), ]
# Print the details of the best one-hidden layer model
cat("Best one-hidden layer network with lowest number of weight parameters:\n")
print(best_one_hidden_layer)
best_two_hidden_layer <-
  two_hidden_layer_models[which.min(two_hidden_layer_models$total_weights), ]
# Print the details of the best one-hidden layer model
cat("Best two-hidden layer network with lowest number of weight parameters:\n")
print(best_two_hidden_layer)