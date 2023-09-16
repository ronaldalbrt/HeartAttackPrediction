using Pkg
Pkg.activate(".")
Pkg.instantiate()

using HeartAttackPrediction, JSON, CSV, MLJ, DataFrames, Random

# Set random seed for code reproducibility
Random.seed!(7)

# Load the dataset
df = CSV.read("data/dataset_ic2023.csv", DataFrame)
df = rename(df, "Colesterol Total" => :ColesterolTotal)

# Partition the dataset into train and test sets 
df_train, df_test = MLJ.partition(df, 0.9, multi=true)

# Define the features and the target
features = [:ColesterolTotal, :Idade, :Glicemia]
target = :Desfecho

# Convert the data to the appropriate format
X = Matrix(df_train[!, features])
y = df_train[:, target]
X_test = Matrix(df_test[!, features])
y_test = df_test[:, target]

# Load the models
logistic = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
svm = @load SVC pkg=LIBSVM verbosity=0
boost = @load XGBoostClassifier pkg=XGBoost verbosity=0

models = Dict("LogisticRegression"=>logistic(), "GradientBoosting"=>boost(), "SVM"=>svm())

# Define the number of clusters to be tested
n_cluster = [2, 4]

# Train and test the models for each number of clusters and save the results in JSON file
for model_key in keys(models)
    for cluster in n_cluster
        model = HeartAttackPrediction.ModelEstimation.train_model(X, y, models[model_key], cluster)

        results =  HeartAttackPrediction.ModelEstimation.test_model(X_test, y_test, model)

        open("test/results/$model_key-$cluster.json", "w") do f
            JSON.print(f, results)
        end
    end
end