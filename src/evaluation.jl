#Functions for cross-validation and evaluation metrics

function k_fold(X::AbstractVecOrMat{<:Real}, label::AbstractVector, k::Integer = 10, eval_func::Function = svm2; seed::Integer = -1)
    seed = seed < 0 ? abs(rand(Int32)) : seed
    rng = Xoshiro(seed)

    if k == 1
        train_acc, test_acc, total_acc, conf_mat = eval_func(X,X,label,label)
        return round.([train_acc test_acc total_acc], digits = 4), conf_mat
    end

    ord = sortperm(label)
    X, label = X[ord, :], label[ord]

    n, p = size(X)
    unique_set = unique(label)
    n_label = length(unique_set)
    ind = split_data_ind(n, k, rng = rng)

    res_acc = zeros(k, 3)
    res_conf_mat = fill(0, n_label, n_label)

    all_ind = collect(1:k)
    cur_splits = fill(0, k-1)
    for i in 1:k
        cur_splits = setdiff(all_ind, i)
        train_ind = reduce(vcat, ind[cur_splits])
        test_ind = ind[i]

        train_acc, test_acc, total_acc, conf_mat = eval_func(X[train_ind, :], X[test_ind, :], label[train_ind], label[test_ind], unique_set = unique_set)
        res_acc[i, :] .= collect((train_acc, test_acc, total_acc))
        res_conf_mat += conf_mat
    end

    return round.(res_acc, digits = 4), res_conf_mat
end

function stratified_k_fold(X::AbstractVecOrMat{<:Real}, label::AbstractVector, k::Integer = 10, eval_func::Function = svm2; seed::Integer = -1)
    seed = seed < 0 ? abs(rand(Int32)) : seed
    rng = Xoshiro(seed)

    if k == 1
        train_acc, test_acc, total_acc, conf_mat = eval_func(X,X,label,label)
        return round.([train_acc test_acc total_acc], digits = 4), conf_mat
    end

    ord = sortperm(label)
    X, label = X[ord, :], label[ord]

    n, p = size(X)
    unique_set = unique(label)
    n_label = length(unique_set)
    ind, label_split = stratified_split_data_ind(label, k, sorted = true, rng = rng)

    res_acc = zeros(k, 3)
    res_conf_mat = fill(0, n_label, n_label)

    all_ind = collect(1:k)
    cur_splits = fill(0, k-1)
    for i in 1:k
        cur_splits = setdiff(all_ind, i)
        train_ind = reduce(vcat, ind[cur_splits])
        test_ind = ind[i]

        train_acc, test_acc, total_acc, conf_mat = eval_func(X[train_ind, :], X[test_ind, :], label[train_ind], label[test_ind], unique_set = unique_set)
        res_acc[i, :] .= collect((train_acc, test_acc, total_acc))
        res_conf_mat += conf_mat
    end

    return round.(res_acc, digits = 4), res_conf_mat
end

function split_data_ind(n::Integer, k::Integer = 10; rng::AbstractRNG = Xoshiro())

    n_k = floor(Integer, n/k)
    n_rem = n - n_k*k

    obs_k = fill(n_k, k)
    obs_k[1:n_rem] .+= 1
    obs_k = [0; cumsum(obs_k)]

    ind = randperm(rng, n)
    return [ind[obs_k[i]+1:obs_k[i+1]] for i in 1:k]
end

function stratified_split_data_ind(label::AbstractVector, k::Integer = 10; sorted::Bool = false,  rng::AbstractRNG = Xoshiro())

    label_sorted = sorted ? label : sort(label)

    label_stats = countmap(label)
    labels, freqs = collect(keys(label_stats)), collect(values(label_stats))
    labels_ord = sortperm(labels)
    labels, freqs = labels[labels_ord], freqs[labels_ord]
    sum_freqs = cumsum(freqs)
    n_labels = length(labels)

    cur_res = split_data_ind(freqs[1], k, rng = rng)
    res = copy(cur_res)
    n_splits = length.(res)
    new_labels = ind2labels(label_sorted, cur_res, k)
    for i in 2:n_labels
        split_ord = sortperm(n_splits)

        cur_res = broadcast(.+, split_data_ind(freqs[i], k, rng = rng), sum_freqs[i-1])
        res = vcat.(res[split_ord], cur_res)
        n_splits = length.(res)
        new_labels = vcat.(new_labels[split_ord], ind2labels(label_sorted, cur_res, k))
    end

    return res, new_labels
end

function ind2labels(label_sorted::AbstractVector, cur_res::Vector{Vector{Int64}}, k::Integer)
    return [label_sorted[cur_res[i]] for i in 1:k]
end



function confusion_matrix(label::AbstractVector, pred::AbstractVector; unique_set::AbstractVector = unique(label))
    miss_label = setdiff(unique_set, label)
    miss_pred = setdiff(unique_set, pred)

    n1, n2, n3, n4 = length(label), length(pred), length(miss_label), length(miss_pred)
    ord_label = sortperm([unique(label); miss_label])
    ord_pred = sortperm([unique(pred); miss_pred])


    CM = ([mat_cat_encode(label) zeros(Int, n1, n3)][:, ord_label])'*([mat_cat_encode(pred) zeros(Int, n2, n4)][:, ord_pred])

    return CM
end

#For two classes the mcc is within [-1,1], for multiclass the minimum can be up to 0 maximum
function mcc(X::AbstractMatrix{<:Integer})
    t_k = sum(X, dims = 1)
    p_k = sum(X, dims = 2)
    c = sum(diag(X))
    s = sum(X)

    matthew_corr = (c*s - dot(t_k,p_k)) / (sqrt(s^2-dot(p_k,p_k))*sqrt(s^2- dot(t_k,t_k)))

    return matthew_corr
end

function diagnostics(X::AbstractMatrix{<:Integer}; method::Symbol = :none, pos_class::Integer = 1)
    n,p = size(X)

    res = NamedTuple[]
    for i in 1:n
        push!(res, diagnostics2D(reduce_mat(X,i)))
    end

    if method in (:none, :binary, :micro, :macro, :weighted)
            if method == :none
                return res
            end

            if method == :binary
                return res[pos_class]
            end

            if method == :micro
                TP, TN, FP, FN = 0, 0, 0, 0
                for tup in res
                    TP += tup.TP
                    TN += tup.TN
                    FP += tup.FP
                    FN += tup.FN
                end
                return diagnostics2D([TP FN; FP TN])
            end

            if method == :macro
                key = keys(res[1])
                value = collect(res[1])

                for i in 2:n
                    value += collect(res[i])
                end
                value = value / n
                return (; zip(key, value)...)
            end

            if method == :weighted
                key = keys(res[1])
                P_temp = res[1].P
                value = P_temp*collect(res[1])
                P = P_temp

                for i in 2:n
                    P_temp = res[i].P
                    value += P_temp*collect(res[i])
                    P += P_temp
                end
                value = value / P
                return (; zip(key, value)...)
            end
    else
        throw(ArgumentError("Method keyword must be one of:(:none, :micro, :macro, :weighted)."))
    end

end

function diagnostics2D(X::AbstractMatrix{<:Integer})
    #Extract info from confusing matrix
    TP, TN, FP, FN = X[1], X[4], X[2], X[3]
    P, N = TP+FN, FP+TN
    PP, PN = TP+FP, FN+TN
    T = P+N

    #Calculate some metrics based on these metrics
    TPR = TP/P
    FNR = FN/P
    TNR = TN/N
    FPR = FP/N

    PPV = TP/PP
    FDR = FP/PP
    NPV = TN/PN
    FOR = FN/PN

    ACC = (TP+TN)/T
    PREV = P/(P+N)
    TS = TP/(TP+FN+FP)

    LRp = TPR/FPR
    LRn = FNR/TNR
    DOR = (TPR/FPR)/(FNR/TNR)

    INFO = TPR+TNR-1.0
    MARK = PPV+NPV-1.0

    FMI = sqrt(PPV*TPR)
    PT = (sqrt(TPR*FPR)-FPR)/(TPR-FPR)
    bACC = (TPR+TNR)/2.0
    F_1 = 2*TP/(2*TP+FP+FN)

    MCC = sqrt(TPR*TNR*PPV*NPV)-sqrt(FNR*FPR*FOR*FDR)

    res = (
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    P = P,
    N = N,
    PP = PP,
    PN = PN,
    T = T,

    TPR = TPR,
    FNR = FNR,
    TNR = TNR,
    FPR = FPR,

    PPV = PPV,
    FDR = FDR,
    NPV = NPV,
    FOR = FOR,

    ACC = ACC,
    PREV = PREV,
    TS = TS,

    LRp = LRp,
    LRn = LRn,
    DOR = DOR,

    INFO = INFO,
    MARK = MARK,

    FMI = FMI,
    PT = PT,
    bACC = bACC,
    F_1 = F_1,

    MCC = MCC
    )


    return res

    #https://en.wikipedia.org/wiki/Confusion_matrix#cite_note-:1-31
end

function reduce_mat(X::AbstractMatrix{<:Integer}, i::Integer)
    n,p = size(X)

    T = sum(X)
    TP = X[i,i]
    FP = sum(X[:,i]) - TP
    FN = sum(X[i,:]) - TP
    TN = T - TP - FN - FP
    #TN = sum(X[setdiff(1:n,i),setdiff(1:p,i)])

    return [TP FN; FP TN]
end

function get_all_eval(coll::AbstractVector, crit::Symbol)
    return [getindex(coll[i], crit) for i in 1:length(coll)]
end

function get_all_eval(coll::AbstractVector, crit::Vector{Symbol})
    n,p = length(coll), length(crit)
    return reshape([getindex(coll[i], crit[j]) for i in 1:n for j in 1:p], (p,n))'
end
