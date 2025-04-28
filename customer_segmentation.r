# Customer Segmentation using Clustering in R
# This script demonstrates how to perform customer segmentation using
# different clustering algorithms in R

# Install required packages if not already installed
required_packages <- c("tidyverse", "cluster", "factoextra", "dbscan", "NbClust", "corrplot", "GGally")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(cluster)      # Clustering algorithms
library(factoextra)   # Clustering visualization
library(dbscan)       # DBSCAN clustering
library(NbClust)      # Determining optimal number of clusters
library(corrplot)     # Correlation visualization
library(GGally)       # Advanced plotting

# Function to read data
read_customer_data <- function(file_path) {
  # Check file extension
  if (grepl("\\.csv$", file_path)) {
    data <- read.csv(file_path)
  } else if (grepl("\\.(xlsx|xls)$", file_path)) {
    data <- readxl::read_excel(file_path)
  } else {
    stop("Unsupported file format. Please use CSV or Excel files.")
  }
  return(data)
}

# Function to preprocess data
preprocess_data <- function(data, features = NULL) {
  # Select features or use all numeric columns
  if (is.null(features)) {
    numeric_cols <- sapply(data, is.numeric)
    if (sum(numeric_cols) == 0) {
      stop("No numeric columns found in the dataset")
    }
    data_subset <- data[, numeric_cols]
  } else {
    data_subset <- data[, features]
    # Convert to numeric if possible
    for (col in names(data_subset)) {
      if (!is.numeric(data_subset[[col]])) {
        data_subset[[col]] <- as.numeric(as.character(data_subset[[col]]))
      }
    }
  }
  
  # Handle missing values
  data_subset <- data_subset %>%
    mutate_all(~ifelse(is.na(.), mean(., na.rm = TRUE), .))
  
  # Scale the data
  scaled_data <- scale(data_subset)
  
  return(list(
    original = data,
    processed = data_subset,
    scaled = scaled_data
  ))
}

# Function to determine optimal number of clusters
find_optimal_clusters <- function(scaled_data) {
  # Elbow method
  wss <- sapply(1:10, function(k) {
    kmeans(scaled_data, centers = k, nstart = 25)$tot.withinss
  })
  
  # Silhouette method
  silhouette_scores <- sapply(2:10, function(k) {
    km <- kmeans(scaled_data, centers = k, nstart = 25)
    ss <- silhouette(km$cluster, dist(scaled_data))
    mean(ss[, 3])
  })
  
  # Plot results
  par(mfrow = c(1, 2))
  plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Within Sum of Squares", 
       main = "Elbow Method")
  
  plot(2:10, silhouette_scores, type = "b", xlab = "Number of Clusters", 
       ylab = "Average Silhouette Width", main = "Silhouette Method")
  
  # Reset plot parameters
  par(mfrow = c(1, 1))
  
  # Return a recommendation
  elbow_recommendation <- which(diff(diff(wss)) > 0)[1] + 1
  silhouette_recommendation <- which.max(silhouette_scores) + 1
  
  return(list(
    wss = wss,
    silhouette_scores = silhouette_scores,
    elbow_recommendation = elbow_recommendation,
    silhouette_recommendation = silhouette_recommendation
  ))
}

# Function to perform K-means clustering
perform_kmeans <- function(data, scaled_data, k) {
  set.seed(123)  # For reproducibility
  km <- kmeans(scaled_data, centers = k, nstart = 25)
  
  # Add cluster information to original data
  data_with_clusters <- data %>%
    mutate(Cluster = as.factor(km$cluster))
  
  # Calculate silhouette score
  sil <- silhouette(km$cluster, dist(scaled_data))
  avg_sil <- mean(sil[, 3])
  
  # Return results
  return(list(
    model = km,
    data = data_with_clusters,
    silhouette = avg_sil
  ))
}

# Function to perform hierarchical clustering
perform_hierarchical <- function(data, scaled_data, k) {
  # Compute distance matrix
  dist_matrix <- dist(scaled_data, method = "euclidean")
  
  # Perform hierarchical clustering
  hc <- hclust(dist_matrix, method = "ward.D2")
  
  # Cut the dendrogram to get k clusters
  clusters <- cutree(hc, k = k)
  
  # Add cluster information to original data
  data_with_clusters <- data %>%
    mutate(Cluster = as.factor(clusters))
  
  # Calculate silhouette score
  sil <- silhouette(clusters, dist_matrix)
  avg_sil <- mean(sil[, 3])
  
  # Return results
  return(list(
    model = hc,
    clusters = clusters,
    data = data_with_clusters,
    silhouette = avg_sil
  ))
}

# Function to perform DBSCAN clustering
perform_dbscan <- function(data, scaled_data, eps = 0.5, minPts = 5) {
  # Perform DBSCAN clustering
  db <- dbscan::dbscan(scaled_data, eps = eps, minPts = minPts)
  
  # Add cluster information to original data
  data_with_clusters <- data %>%
    mutate(Cluster = as.factor(db$cluster))
  
  # Count noise points (cluster -1)
  noise_count <- sum(db$cluster == 0)
  
  # Calculate silhouette score if there are at least 2 clusters and no noise
  if (length(unique(db$cluster)) > 1 && noise_count == 0) {
    sil <- silhouette(db$cluster, dist(scaled_data))
    avg_sil <- mean(sil[, 3])
  } else {
    avg_sil <- NA
  }
  
  # Return results
  return(list(
    model = db,
    data = data_with_clusters,
    noise_count = noise_count,
    silhouette = avg_sil
  ))
}

# Function to visualize clustering results
visualize_clusters <- function(data_with_clusters, method_name) {
  # PCA for dimensionality reduction
  pca_result <- prcomp(data_with_clusters %>% 
                       select_if(is.numeric) %>% 
                       select(-Cluster, -any_of("id")), 
                     scale. = TRUE)
  
  # Create a data frame with PCA results and cluster information
  pca_df <- as.data.frame(pca_result$x[, 1:2])
  pca_df$Cluster <- data_with_clusters$Cluster
  
  # Plot the clusters
  ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.7) +
    theme_minimal() +
    labs(title = paste("Customer Segments using", method_name),
         x = "Principal Component 1",
         y = "Principal Component 2")
}

# Function to analyze cluster profiles
analyze_clusters <- function(data_with_clusters) {
  # Calculate mean values for each numeric variable by cluster
  cluster_summary <- data_with_clusters %>%
    group_by(Cluster) %>%
    summarise_if(is.numeric, mean, na.rm = TRUE)
  
  # Calculate cluster sizes
  cluster_sizes <- data_with_clusters %>%
    count(Cluster) %>%
    mutate(percentage = n / sum(n) * 100)
  
  return(list(
    profiles = cluster_summary,
    sizes = cluster_sizes
  ))
}

# Example usage
# Uncomment and modify the following code to run the analysis

# # Read data
# file_path <- "path/to/your/customer_data.csv"
# customer_data <- read_customer_data(file_path)
# 
# # Explore data
# str(customer_data)
# summary(customer_data)
# 
# # Preprocess data
# features <- c("feature1", "feature2", "feature3")  # Replace with your actual feature names
# preprocessed_data <- preprocess_data(customer_data, features)
# 
# # Find optimal number of clusters
# optimal_k <- find_optimal_clusters(preprocessed_data$scaled)
# print(paste("Elbow method recommends:", optimal_k$elbow_recommendation, "clusters"))
# print(paste("Silhouette method recommends:", optimal_k$silhouette_recommendation, "clusters"))
# 
# # Choose k based on recommendations
# k <- optimal_k$silhouette_recommendation
# 
# # Perform K-means clustering
# kmeans_result <- perform_kmeans(preprocessed_data$original, preprocessed_data$scaled, k)
# cat("K-means silhouette score:", kmeans_result$silhouette, "\n")
# 
# # Visualize K-means clusters
# kmeans_viz <- visualize_clusters(kmeans_result$data, "K-means")
# print(kmeans_viz)
# 
# # Analyze K-means cluster profiles
# kmeans_profiles <- analyze_clusters(kmeans_result$data)
# print(kmeans_profiles$profiles)
# print(kmeans_profiles$sizes)
# 
# # Perform hierarchical clustering
# hierarchical_result <- perform_hierarchical(preprocessed_data$original, preprocessed_data$scaled, k)
# cat("Hierarchical clustering silhouette score:", hierarchical_result$silhouette, "\n")
# 
# # Visualize hierarchical clusters
# hierarchical_viz <- visualize_clusters(hierarchical_result$data, "Hierarchical Clustering")
# print(hierarchical_viz)
# 
# # Perform DBSCAN clustering
# dbscan_result <- perform_dbscan(preprocessed_data$original, preprocessed_data$scaled, eps = 0.5, minPts = 5)
# cat("DBSCAN found", length(unique(dbscan_result$model$cluster)), "clusters with", 
#     dbscan_result$noise_count, "noise points\n")
# 
# # Visualize DBSCAN clusters
# dbscan_viz <- visualize_clusters(dbscan_result$data, "DBSCAN")
# print(dbscan_viz)
# 
# # Export results
# write.csv(kmeans_result$data, "customer_segments_kmeans.csv", row.names = FALSE)
