module ModelEstimation
    using MLJ, MLJBase, Clustering, LinearAlgebra, ..SubsetSelection

    #-----------------------------------------------------------
    # Definition of the Model structure
    #-----------------------------------------------------------
    # Parameters:
    # models: Estimated models for each cluster
    # optimal_features: Optimal features for each cluster
    # clusters: Found clusters in the dataset
    # model_string: String representation of the model
    #-----------------------------------------------------------
    struct Model
        models::Dict{Int64, Any}
        optimal_features::Dict{Int64, Vector{Int64}}
        clusters::Dict{Int64, Any}
        model_string::String
    end
    
    #-----------------------------------------------------------
    # Definition of the train_model function
    #-----------------------------------------------------------
    # Parameters:
    # X: Matrix of features
    # y: Target Vector 
    # model: Model to be used
    # n_clusters: Number of clusters to be found
    #-----------------------------------------------------------
    # Returns:
    # Model structure
    #-----------------------------------------------------------
    function train_model(X::Matrix, y::Vector, model=(@load LogisticClassifier pkg=MLJLinearModels verbosity=0)(), n_clusters::Int64=4)
        n_features = size(X, 2)
        
        # Calculate the distance matrix between the observations and perform the hierarchical clustering
        G = X * X';
        D = sqrt.(diag(G) .+ diag(G)' .- 2 .* G);
        clust = hclust(D, linkage=:complete)
        clusters = cutree(clust, k=n_clusters)

        model_string = string(typeof(model))

        # Calculate the mean of the observations in each cluster
        cluster_dict = Dict{Int64, Any}()
        for cluster in clusters
            cluster_dict[cluster] = mean(X[findall(x -> x == cluster, clusters), :], dims=1)
        end

        # Estimate the model for each cluster
        models = Dict()
        optimal_features = Dict()
        for cluster in unique(clusters)
            cluster_idx = findall(x -> x == cluster, clusters)
            X_aux = X[cluster_idx, :] 
            y_aux = categorical(y[cluster_idx])
        
            subset_input = SubsetSelection.SubsetSelectionInput(n_features, X_aux, y_aux, model)
        
            optimal_features[cluster], models[cluster] = SubsetSelection.bestSubsetSelection(subset_input)
        end

        return Model(models, optimal_features, cluster_dict, model_string)
    end

    #-----------------------------------------------------------
    # Definition of the test_model function
    #-----------------------------------------------------------
    # Parameters:
    # X: Matrix of features
    # y: Target Vector
    # model: Model structure
    #-----------------------------------------------------------
    # Returns:
    # Dictionary with the accuracy, recall, precision and f1score
    #-----------------------------------------------------------
    function test_model(X::Matrix, y::Vector, model::ModelEstimation.Model)
        n_observations = size(X, 1)
        
        models = model.models
        optimal_features = model.optimal_features
        clusters = model.clusters

        model_string = model.model_string

        ŷ = categorical(ones(n_observations))

        # Assign each observation to one of the estimated clusters
        assigned_clusters = []
        for i in 1:n_observations
            cluster = reduce((x, y) -> norm(X[i, :]' - clusters[x]) <= norm(X[i, :]' - clusters[y]) ? x : y, keys(clusters))

            push!(assigned_clusters, cluster)
        end

        # Predict the target for each observation using the model estimated for the cluster it was assigned to
        for cluster in unique(keys(clusters))
            idx_clusters = findall(x -> x == cluster, assigned_clusters)
            opt_features = optimal_features[cluster]
            X_aux = X[idx_clusters, opt_features]
            
            if model_string in ["MLJLinearModels.LogisticClassifier", "MLJXGBoostInterface.XGBoostClassifier"]
                ŷ_dist = predict(models[cluster], MLJ.table(X_aux))
                ŷ_aux = mode.(ŷ_dist)
            else
                ŷ_aux = predict(models[cluster], MLJ.table(X_aux))
            end
            
            ŷ[idx_clusters] = ŷ_aux
        end
        
        return Dict("accuracy"=>accuracy(ŷ, y), 
        "recall"=>recall(ŷ, y), 
        "precision"=>precision(ŷ, y),
        "f1score"=>f1score(ŷ, y)
        )
    end
end