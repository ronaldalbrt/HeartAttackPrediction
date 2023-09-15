module SubsetSelection
    using MLJ, Combinatorics, MLJBase

    struct SubsetSelectionInput
        n_features::Int64
        X::Matrix{Float64}
        y::Vector{Any}
        model::Model
    end

    function bestSubsetSelection(input::SubsetSelectionInput)
        n_features = input.n_features
        X = input.X
        y = input.y
        model = input.model

        best_features = Vector{Vector{Int64}}([])
        out_of_sample_acc = Dict{Vector{Int64}, Float64}()
        best_model_dict = Dict{Vector{Int64}, Any}()

        for i in 1:n_features
            best_acc = 0
            curr_features = []
            best_model = nothing

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

                println("Features: ", features, " Accuracy: ", acc)

                if acc > best_acc
                    best_acc = acc
                    curr_features = features
                    best_model = curr_mach
                end

            end
            push!(best_features, curr_features)

            out_of_sample_acc[curr_features] = crossValidation(input, curr_features)
            best_model_dict[curr_features] = best_model
        end

        best_features = reduce((x, y) -> out_of_sample_acc[x] >= out_of_sample_acc[y] ? x : y, keys(out_of_sample_acc))


        return best_features, best_model_dict[best_features]
    end 


    function crossValidation(input::SubsetSelectionInput, features::Vector{Int64}; nfolds::Int64=10, shuffle::Bool=true, rng::Int64=7)
        cv = CV(nfolds=nfolds, shuffle=shuffle, rng=rng)

        X = input.X[:, features]
        y = input.y
        model = input.model

        accuracies = []

        for (itrain, itest) in MLJBase.train_test_pairs(cv, 1:length(y))
            curr_mach = machine(model, MLJ.table(X), y, scitype_check_level=0)
            fit!(curr_mach, rows=itrain, verbosity=0)

            if string(typeof(model)) in ["MLJLinearModels.LogisticClassifier", "MLJXGBoostInterface.XGBoostClassifier"]
                ŷ_dist = predict(curr_mach, rows=itest)
                ŷ = mode.(ŷ_dist)
            else
                ŷ = predict(curr_mach, rows=itest)
            end

            acc = accuracy(ŷ, y[itest])
            push!(accuracies, acc)
        end

        return mean(accuracies)
    end
end