# 1st Subtask
library(readxl)
library(cluster)
library(NbClust)
library(factoextra)
# Load the dataset
wine_data <- read_excel("data/Whitewine_v6.xlsx")
# (DATA PRE-PROCESSING)
# a)
# Function to find outliers
outlier <- function(data) {
  # Calculate quartiles and IQR for each column
  quartiles <- apply(data, 2, quantile, probs = c(0.25, 0.75))
  iqr <- quartiles[2, ] - quartiles[1, ]
  
  # Define lower and upper fences
  lower_fence <- quartiles[1, ] - 1.5 * iqr
  upper_fence <- quartiles[2, ] + 1.5 * iqr
  
  # Detect outliers
  outlier_indices <- apply(data, 1, function(row) {
    any(row < lower_fence) || any(row > upper_fence)
  })
  
  return(which(outlier_indices))
}
outlier_indices <- outlier(wine_data)
wine_data_removed_outliers <- wine_data[-outlier_indices, ]
#head(wine_data_removed_outliers, n=3)
# Scale Data
min_values <- apply(wine_data_removed_outliers[,1:11],2, min)
max_values <- apply(wine_data_removed_outliers[,1:11],2, max)
scaled_data <- scale(wine_data_removed_outliers[,1:11], center = min_values, scale = max_values -
                       min_values)
#print(scaled_data)
# c)
# Visualize NBclust Method
nb <- NbClust(scaled_data, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = 
                "kmeans", index = "all")
# Visualize Elbow Method
fviz_nbclust(scaled_data, kmeans, method = "wss") + labs(subtitle = "Elbow Method")
# Visualize Gap Method

fviz_nbclust(scaled_data, kmeans, method = "gap_stat") + labs(subtitle = "Gap Method")
# Visualize Silhouette Method
fviz_nbclust(scaled_data, kmeans, method = "silhouette") + labs(subtitle = "Silhouette Method")
# d)
# For k=2 clusters
# Step 3: Perform K-means clustering
# Choose the optimal k based on the above methods
optimal_k <- 2 # for example, choose the value obtained from the elbow method
# Perform k-means clustering
kmeans_result <- kmeans(scaled_data, centers = optimal_k)
# Show kmeans output
print(kmeans_result)
# Cluster centers
cat("\nCluster Centers:\n")
print(kmeans_result$centers)
# Clustered results
cat("\nClustered Results:\n")
print(kmeans_result$cluster)

# Calculate BSS and WSS
BSS <- kmeans_result$betweenss # Between-cluster sum of squares
TSS <- kmeans_result$totss # Total sum of squares
WSS <- kmeans_result$tot.withinss # Within-cluster sum of squares
# Ratio of BSS to TSS
BSS_ratio <- BSS / TSS
cat("\nRatio of BSS to TSS:", BSS_ratio, "\n")
# Print BSS and WSS
cat("\nBetween-cluster sum of squares (BSS):", BSS, "\n")
cat("Within-cluster sum of squares (WSS):", WSS, "\n")
fviz_cluster(kmeans_result, data = scaled_data)
# e)
# Define 'k' as the number of clusters
k <- length(unique(kmeans_result$cluster))
# Compute silhouette width
sil_width <- silhouette(kmeans_result$cluster, dist(scaled_data))
# Plot silhouette plot
plot(sil_width, col = 1:k, border = NA)
# Average silhouette width score
avg_sil_width <- mean(sil_width[, "sil_width"])

cat("\nAverage Silhouette Width Score:", avg_sil_width, "\n")
# 2nd Subtask
# f) Applying PCA
pca_model <- prcomp(scaled_data, scale = TRUE)
# Output eigenvalues and eigenvectors
print(summary(pca_model))
# Print eigenvalues and eigenvectors
print(pca_model$sdev) # Eigenvalues (square roots of the eigenvalues)
print(pca_model$rotation) # Eigenvectors
# Cumulative proportion of variance explained
cumulative_prop_var <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))
print(cumulative_prop_var)
# Choose PCs with cumulative proportion > 0.85
num_pcs <- min(which(cumulative_prop_var >= 0.85))
print(paste("Number of PCs chosen:", num_pcs))
# Create a new dataset with selected PCs
pca_data <- data.frame(pca_model$x[, 1:num_pcs])
# g) Determining the number of clusters for PCA dataset

# NbClust
nb <- NbClust(pca_data, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = 
                "kmeans", index = "all")
# Visualize Elbow method
fviz_nbclust(pca_data, kmeans, method = "wss") + labs(subtitle = "Elbow Method for PCA-based 
dataset")
# Gap statistic
fviz_nbclust(pca_data, kmeans, method = "gap_stat") + labs(subtitle = "Gap Method for PCA-based 
dataset")
#Visualize Silhouette method
fviz_nbclust(pca_data, kmeans, method = "silhouette") + labs(subtitle = "Silhouette Method for PCAbased dataset")
# h) K-means clustering on PCA dataset
k <- 3 # Choose the optimal number of clusters based on the previous methods
set.seed(123)
kmeans_model_pca <- kmeans(pca_data, k, nstart = 10)
# Print cluster centers, clustered results, and BSS/TSS ratio
print(kmeans_model_pca$centers)
print(kmeans_model_pca$cluster)
bss <- sum(kmeans_model_pca$withinss)
tss <- sum(kmeans_model_pca$totss)
print(paste("BSS/TSS ratio:", bss / tss))

# Calculate BSS and WSS
bss <- sum(kmeans_model_pca$betweenss)
wss <- sum(kmeans_model_pca$withinss)
print(paste("BSS:", bss))
print(paste("WSS:", wss))
# i) Silhouette plot for PCA dataset
silhouette_plot <- fviz_silhouette(kmeans_model_pca)
print(silhouette_plot)
avg_silhouette_width <- mean(silhouette_width(kmeans_model_pca))
print(paste("Average Silhouette Width:", avg_silhouette_width))
# j) Calinski-Harabasz Index
calinski_harabasz_index <- calinhara(pca_data, kmeans_model_pca$cluster)
print(paste("Calinski-Harabasz Index:", calinski_harabasz_index))
?prcomp()