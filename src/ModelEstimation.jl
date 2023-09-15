module ModelEstimation
    using MLJ, MLJBase, Clustering, LinearAlgebra, ..SubsetSelection

    struct Model
        models::Dict{Int64, Any}
        optimal_features::Dict{Int64, Vector{Int64}}
        clusters::Dict{Int64, Any}
    end
    
    function train_model(X::Matrix, y::Vector, model=(@load LogisticClassifier pkg=MLJLinearModels verbosity=0)(), n_clusters::Int64=4)
        n_features = size(X, 2)
        
        G = X * X';
        D = sqrt.(diag(G) .+ diag(G)' .- 2 .* G);

        clust = hclust(D, linkage=:complete)
        clusters = cutree(clust, k=n_clusters)

        cluster_dict = Dict{Int64, Any}()
        for cluster in clusters
            cluster_dict[cluster] = mean(X[findall(x -> x == cluster, clusters), :], dims=1)
        end

        models = Dict()
        optimal_features = Dict()
        for cluster in unique(clusters)
            cluster_idx = findall(x -> x == cluster, clusters)
            X_aux = X[cluster_idx, :] 
            y_aux = categorical(y[cluster_idx])
        
            subset_input = SubsetSelection.SubsetSelectionInput(n_features, X_aux, y_aux, model)
        
            optimal_features[cluster], models[cluster] = SubsetSelection.bestSubsetSelection(subset_input)
        end

        return Model(models, optimal_features, cluster_dict)
    end

    function test_model(X::Matrix, y::Vector, model::ModelEstimation.Model)
        n_observations = size(X, 1)
        
        models = model.models
        optimal_features = model.optimal_features
        clusters = model.clusters

        ŷ = categorical(ones(n_observations))
        assigned_clusters = []
        for i in 1:n_observations
            cluster = reduce((x, y) -> norm(X[i, :]' - clusters[x]) <= norm(X[i, :]' - clusters[y]) ? x : y, keys(clusters))

            push!(assigned_clusters, cluster)
        end

        for cluster in unique(keys(clusters))
            idx_clusters = findall(x -> x == cluster, assigned_clusters)
            opt_features = optimal_features[cluster]
            X_aux = X[idx_clusters, opt_features]

            ŷ_dist = predict(models[cluster], MLJ.table(X_aux))

            ŷ_aux = mode.(ŷ_dist)
            ŷ[idx_clusters] = ŷ_aux
        end
        
        return Dict("accuracy"=>accuracy(ŷ, y), 
        "recall"=>recall(ŷ, y), 
        "precision"=>precision(ŷ, y),
        "f1score"=>f1score(ŷ, y)
        )
    end
end