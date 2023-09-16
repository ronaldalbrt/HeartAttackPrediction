module SubsetSelection
    using MLJ, Combinatorics, MLJBase

    #-----------------------------------------------------------
    # Definition of the SubsetSelectionInput structure
    #-----------------------------------------------------------
    # Parameters:
    # n_features: Number of features
    # X: Matrix of features
    # y: Target Vector
    # model: Model to be used
    #-----------------------------------------------------------
    struct SubsetSelectionInput
        n_features::Int64
        X::Matrix{Float64}
        y
        model::Model
    end

    #-----------------------------------------------------------
    # Definition of the bestSubsetSelection function
    #-----------------------------------------------------------
    # Parameters:
    # input: SubsetSelectionInput structure
    #-----------------------------------------------------------
    # Returns:
    # best_features: Optimal features
    # best_model: Estimated model
    #-----------------------------------------------------------
    function bestSubsetSelection(input::SubsetSelectionInput)
        n_features = input.n_features
        X = input.X
        y = input.y
        model = input.model

        best_features = Vector{Vector{Int64}}([])
        out_of_sample_acc = Dict{Vector{Int64}, Float64}()
        best_model_dict = Dict{Vector{Int64}, Any}()

        # Iterate over the maximum number of features
        for i in 1:n_features
            best_acc = 0
            curr_features = []
            best_model = nothing

            # Iterate over all possible combinations of features and estimate a model for each combination
            for features in combinations(1:n_features, i)
                curr_X = MLJ.table(X[:, features])

                curr_mach = machine(model, curr_X, y, scitype_check_level=0)

                fit!(curr_mach, verbosity=0)
                
                if string(typeof(model)) in ["MLJLinearModels.LogisticClassifier", "MLJXGBoostInterface.XGBoostClassifier"]
                    ŷ_dist = predict(curr_mach, curr_X)
                    ŷ = mode.(ŷ_dist)
                else
                    ŷ = predict(curr_mach, curr_X)
                end
                acc = accuracy(ŷ, y)

                # Update the best model if the current model is better
                if acc > best_acc
                    best_acc = acc
                    curr_features = features
                    best_model = curr_mach
                end
            end
            push!(best_features, curr_features)

            # Estimate the out-of-sample accuracy for the best model
            out_of_sample_acc[curr_features] = crossValidation(input, curr_features)
            best_model_dict[curr_features] = best_model
        end
        # Select the best model based on the out-of-sample accuracy
        best_features = reduce((x, y) -> out_of_sample_acc[x] >= out_of_sample_acc[y] ? x : y, keys(out_of_sample_acc))

        # Return the best features and the best model
        return best_features, best_model_dict[best_features]
    end 

    #-----------------------------------------------------------
    # Definition of the crossValidation function
    #-----------------------------------------------------------
    # Parameters:
    # input: SubsetSelectionInput structure
    # features: Features to be used
    # nfolds: Number of folds
    # shuffle: Shuffle the data
    # rng: Random seed
    #-----------------------------------------------------------
    # Returns:
    # Mean accuracy
    #-----------------------------------------------------------
    function crossValidation(input::SubsetSelectionInput, features::Vector{Int64}; nfolds::Int64=10, shuffle::Bool=true, rng::Int64=7)
        cv = CV(nfolds=nfolds, shuffle=shuffle, rng=rng)

        X = input.X[:, features]
        y = input.y
        model = input.model

        accuracies = []

        # Split the data into training and test sets with the given number of folds
        for (itrain, itest) in MLJBase.train_test_pairs(cv, 1:length(y))
            curr_mach = machine(model, MLJ.table(X), y, scitype_check_level=0)
            fit!(curr_mach, rows=itrain, verbosity=0)

            if string(typeof(model)) in ["MLJLinearModels.LogisticClassifier", "MLJXGBoostInterface.XGBoostClassifier"]
                ŷ_dist = predict(curr_mach, rows=itest)
                ŷ = mode.(ŷ_dist)
            else
                ŷ = predict(curr_mach, rows=itest)
            end

            # Calculate the accuracy for the current fold
            acc = accuracy(ŷ, y[itest])
            push!(accuracies, acc)
        end

        # Return the mean accuracy
        return mean(accuracies)
    end
end