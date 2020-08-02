using DataFrames, Plots, CSV, DecisionTree, Statistics

df = DataFrame(CSV.File("waveform.data", header=false))

#Define hat function
function hat(start, stop)
    y1 = [0 for i in 1:1:start]
    y2 = [i for i in 1:1:6]
    y3 = [i for i in 5:-1:0]
    y4 = [0 for i in stop:1:20]
    return vcat(y1, y2, y3, y4)
end

t = [i for i in 1:1:21]
p1 = plot(hcat(t, t, t), [hat(1,13) hat(9,21) hat(5,17)], lw=2, label=["H1" "H2" "H3"])
savefig(p1, "Basis Hat Functions.pdf")

#u ∈ (0,1)
#eps ∈ N(0,1)
#Class 0: x = u*h1 + (1-u)*h2 + eps
#Class 1: x = u*h1 + (1-u)*h3 + eps
#Class 2: x = u*h2 + (1-u)*h3 + eps

#Example of classes
p2 = scatter(hcat(t, t, t), [i for i in convert(Matrix, df[1:3, 1:end-1])]', layout=(3,1), label=["Class 2" "Class 1" "Class 0"])
savefig(p2, "Examples of Classes.pdf")

X_train, y_train = df[1:4000, 1:end-1], df[1:4000, end]
X_test, y_test = df[4001:end, 1:end-1], df[4001:end, 1:end-1]

# the data loaded are of type Array{Any}
# cast them to concrete types for better performance
features = float.(convert(Matrix, X_train))
labels   = string.(y_train)

### train full-tree classifier ###
model = build_tree(labels, features, 3, 3)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)
# pretty print of the tree
print_tree(model, 3)
# apply learned model
apply_tree(model, X_test[end])
# apply model to all the sames
preds = apply_tree(model, features)
# generate confusion matrix, along with accuracy and kappa scores
confusion_matrix(labels, preds)

### train random forest classifier ###

model = build_forest(labels, features, 2, 100, 0.5, 3)

# apply learned model
apply_forest_proba(model, X_test[end], ["0", "1", "2"])

# run 3-fold cross validation for forests, using 2 random features per split
n_folds=3
n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

y_pred = parse.(Int64, apply_forest(model, convert(Matrix, X_test)))


plot(t, (hat(1,13)+hat(9,21))/2, label="True", title="H1 + H2")
tmp0 = convert(Matrix, X_test[findall(==(0), y_pred),:])
val0 = mean(tmp0, dims=1)'
scatter!(t, val0, label="Pred")
savefig("Result_Class0.pdf")


plot(t, (hat(1,13)+hat(5, 17))/2, label="True", title="H1 + H3")
tmp1 = convert(Matrix, X_test[findall(==(1), y_pred),:])
val1 = mean(tmp1, dims=1)'
scatter!(t, val1, label="Pred")
savefig("Result_Class1.pdf")


plot(t, (hat(9, 21)+hat(5, 17))/2, label="True", title="H1 + H3")
tmp2 = convert(Matrix, X_test[findall(==(2), y_pred),:])
val2 = mean(tmp2, dims=1)'
scatter!(t, val2, label="Pred")
savefig("Result_Class2.pdf")