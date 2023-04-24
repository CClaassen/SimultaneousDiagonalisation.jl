#Functions for doing classifications
 
function ols2(X::AbstractVecOrMat{<:Real}, label::AbstractVector)
    n,p = size(X)

    if 0 in label
        label = label .+ 1
    end

    X2 = [ones(n) X]
    β = (X2'X2)\(X2'label)

    label_pred = floor.(Integer, round.(X2*β, digits= 0))
    label_pred[label_pred .< minimum(label)] .= minimum(label)
    label_pred[label_pred .> maximum(label)] .= maximum(label)

    acc = mean(label_pred .== label)
    return label_pred, acc
end

function ols2(X::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real}, label_X::AbstractVector, label_Y::AbstractVector; unique_set::AbstractVector = unique(label_X), ret_test::Bool = false)
    n_train, n_test = length(label_X), length(label_Y)

    if 0 in label_X || 0 in label_Y
        label_X = label_X .+ 1
        label_Y = label_Y .+ 1
        unique_set = unique_set .+ 1
    end

    n_x,p_x = size(X)
    n_y,p_y = size(Y)

    X2 = [ones(n_x) X]
    β = (X2'X2)\(X2'label_X)

    Y2 = [ones(n_y) Y]

    train_pred = floor.(Integer, round.(X2*β, digits= 0))
    test_pred = floor.(Integer, round.(Y2*β, digits= 0))

    train_pred[train_pred .< minimum(label_X)] .= minimum(label_X)
    train_pred[train_pred .> maximum(label_X)] .= maximum(label_X)
    test_pred[test_pred .< minimum(label_X)] .= minimum(label_X)
    test_pred[test_pred .> maximum(label_X)] .= maximum(label_X)

    train_acc = mean(train_pred .== label_X)
    test_acc = mean(test_pred .== label_Y)

    total_acc = (n_train*train_acc + n_test*test_acc) / (n_train + n_test)
    conf_mat = confusion_matrix(label_Y, test_pred, unique_set = unique_set)

    if !ret_test
        return train_acc, test_acc, total_acc, conf_mat
    else
        return train_acc, test_acc, total_acc, conf_mat, test_pred
    end
end

function logit2(X::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real}, label_X::AbstractVector, label_Y::AbstractVector; unique_set::AbstractVector = unique(label_X), ret_test::Bool = false)
    n_train, n_test = length(label_X), length(label_Y)

    if 0 in label_X || 0 in label_Y
        label_X = label_X .+ 1
        label_Y = label_Y .+ 1
        unique_set = unique_set .+ 1
    end

    n_x,p_x = size(X)
    n_y,p_y = size(Y)

    X2 = [ones(n_x) X]
    Y2 = [ones(n_y) Y]

    p_x += 1
    p_y += 1

    c = length(unique(label_X))

    b_init = zeros(p_x,c-1)

    #opt_func = OnceDifferentiable(vars -> loglike(X2, [reshape(vars, (p_x,c-1)) zeros(p_x)], label_X, c = c), vec(b_init); autodiff=:forward);
    opt_func = TwiceDifferentiable(vars -> loglike(X2, [reshape(vars, (p_x,c-1)) zeros(p_x)], label_X, c = c), vec(b_init); autodiff=:forward);
    #res = Optim.optimize(opt_func, vec(b_init), ConjugateGradient(), Optim.Options(iterations = 2500))
    res = Optim.optimize(opt_func, vec(b_init), LBFGS(), Optim.Options(iterations = 2500))
    param = Optim.minimizer(res)
    param = [reshape(param, (p_x,c-1)) zeros(p_x)]

    logit(X,β) = exp.(X*β) ./ sum(exp.(X*β), dims = 2)
    soft_pred_X = logit(X2,param)
    soft_pred_Y = logit(Y2,param)
    train_pred = argmax.(RowVecs(soft_pred_X))
    test_pred = argmax.(RowVecs(soft_pred_Y))

    train_acc = mean(train_pred .== label_X)
    test_acc = mean(test_pred .== label_Y)

    total_acc = (n_train*train_acc + n_test*test_acc) / (n_train + n_test)
    conf_mat = confusion_matrix(label_Y, test_pred, unique_set = unique_set)

    if !ret_test
        return train_acc, test_acc, total_acc, conf_mat
    else
        return train_acc, test_acc, total_acc, conf_mat, test_pred
    end
end

function logit2(X::AbstractVecOrMat{<:Real}, label::AbstractVector)
    n, p = size(X)

    if 0 in label
        label = label .+ 1
    end

    X2 = [ones(n) X]
    p += 1
    c = length(unique(label))

    b_init = zeros(p,c-1)

    opt_func = TwiceDifferentiable(vars -> loglike(X2, [reshape(vars, (p,c-1)) zeros(p)], label, c = c), vec(b_init); autodiff=:forward);
    res = Optim.optimize(opt_func, vec(b_init), BFGS(), Optim.Options(iterations = 2500))
    param = Optim.minimizer(res)
    param = [reshape(param, (p,c-1)) zeros(p)]

    logit(X,β) = exp.(X*β) ./ sum(exp.(X*β), dims = 2)
    soft_pred = logit(X2,param)
    label_pred = argmax.(RowVecs(soft_pred))

    acc = mean(label_pred .== label)
    return label_pred, acc
end


function loglike(X::AbstractVecOrMat{<:Real}, beta::AbstractVecOrMat{<:Real}, label::AbstractVector; c::Integer = length(unique(label)))
    n,p = size(X)

    function logit_func(ind)
        res = exp.((X[ind,:]'beta))
        return res / sum(res)
    end

    res = 0.0
    for i in 1:n
        res += log(logit_func(i)[label[i]])
    end

    return -res
end


#Function for using SVM, returning in-sample predictions
function svm2(X::AbstractVecOrMat{<:Real}, label::AbstractVector)
    model = svmtrain(X', label, kernel = LIBSVM.Kernel.Linear)
    label_pred, dec_val = svmpredict(model, X')
    acc = mean(label_pred .== label)
    return label_pred, acc
end

#Function for using SVM, returning bith in- and out-of-sample predictions
function svm2(X::AbstractVecOrMat{<:Real}, Y::AbstractVecOrMat{<:Real}, label_X::AbstractVector, label_Y::AbstractVector; unique_set::AbstractVector = unique(label_X), ret_test::Bool = false)
    n_train, n_test = length(label_X), length(label_Y)
    model = svmtrain(X', label_X, kernel = LIBSVM.Kernel.Linear)
    train_pred, _ = svmpredict(model, X')
    test_pred, _ = svmpredict(model, Y')
    train_acc = mean(train_pred .== label_X)
    test_acc = mean(test_pred .== label_Y)
    total_acc = (n_train*train_acc + n_test*test_acc) / (n_train + n_test)
    conf_mat = confusion_matrix(label_Y, test_pred, unique_set = unique_set)

    if !ret_test
        return train_acc, test_acc, total_acc, conf_mat
    else
        return train_acc, test_acc, total_acc, conf_mat, test_pred
    end
end
