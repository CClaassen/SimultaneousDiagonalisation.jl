#Functions for both traditional and robust estimation of location and scale
 
#Function for location based on the regular mean
function mean1(X::VecOrMat{<:Real})
    return raw_moment(X, 1)
end

#Function for scatter based on the regular covariance
function cov2(X::Matrix{<:Real}; O::Matrix{<:Real} = raw_moment(X, 1))
    n,p = size(X)
    X_c = X .- O

    return self_inner(X_c) / (n-1)
end

#Function for location based on the third moment
function mean3(X::AbstractVecOrMat{<:Real})
    n,p = X isa Vector ? (length(X), 1) : size(X)
    d = mahalanobis2(X, raw_moment(X, 1); S = (n-1)*cov2(X)/n)
    return raw_moment(d .* X, 1) / p
end

#Function for scatter based on the fourth moments
function cov4(X::Matrix{<:Real}; O::Matrix{<:Real} = mean1(X))
    n,p = size(X)

    X_c = X .- O
    S = sqrt(cov2(X))
    rad = sqrt.(sum((X_c * inv(S)).^2, dims = 2))
    Y = rad .* X_c
    return (1.0 / (n * (p + 2))) *  self_inner(Y)
end

#Function for local covariance
function lcov(X::AbstractVecOrMat{<:Real}, kernel_function::Function = epanechnikov_kernel_w; scale::Real = 1.0, dist_metric::Metric = euclidean, ret_loc::Bool = false)
    Y = kernel_smoother(X, kernel_function, weighted_mean, scale = scale, dist_metric = dist_metric)
    if !ret_loc
        return cov(Y)
    else
        return cov(Y), mean1(Y)
    end
end

#Function for robustified local covariance
function rlcov(X::AbstractVecOrMat{<:Real}, kernel_function::Function = epanechnikov_kernel_w; scale::Real = 1.0, dist_metric::Metric = euclidean, ret_loc::Bool = false)
    Y = kernel_smoother(X, kernel_function, geometric_median, scale = scale, dist_metric = dist_metric)
    if !ret_loc
        return cov(Y)
    else
        return cov(Y), mean1(Y)
    end
end

#Function for adaptive local covariance
function alcov(X::AbstractVecOrMat{<:Real}, kernel_function::Function = epanechnikov_kernel_w, nn_param::Tuple{Int64, Float64} = (3, 1.0); scale::Real = 1.0, dist_metric::Metric = euclidean, ret_loc::Bool = false)
    Y = adaptive_kernel_smoother(X, kernel_function, weighted_mean, nn_param, scale = scale, dist_metric = dist_metric)
    if !ret_loc
        return cov(Y)
    else
        return cov(Y), mean1(Y)
    end
end

#Function for adaptive robustified local covariance
function arlcov(X::AbstractVecOrMat{<:Real}, kernel_function::Function = epanechnikov_kernel_w, nn_param::Tuple{Int64, Float64} = (3, 1.0); scale::Real = 1.0, dist_metric::Metric = euclidean, ret_loc::Bool = false)
    Y = adaptive_kernel_smoother(X, kernel_function, geometric_median, nn_param, scale = scale, dist_metric = dist_metric)

    if !ret_loc
        return cov(Y)
    else
        return cov(Y), mean1(Y)
    end
end


#Function for robust scatter based on (approximation of) 1-step reweighted MCD
#TODO: maybe convert c-step to seperate function
function fastMCD(X::Matrix{<:Real}, h::Integer = opt_h(X); iterations::Integer = 500, inner_steps::Integer = 2, n_best_h::Integer = 10, rng_seed::Integer = 0)
    #TODO: Add check: opt_h(X) <= h <=n

    n, p = size(X)
    #breakdown_val = (n - h + 1) / n

    if h == n
        return mean(X, dims = 1), cov(X)
    end

    #Add in parellization
    #prng = rng_seed == 0 ? Xoshiro() : Random.seed!(rng_seed)
    #ind0_dist = Categorical(ones(n) / n)

    #Determine best inititial subsets
    best_ind = zeros(Int64, h, n_best_h)
    best_det = fill(Inf, n_best_h)
    cur_worst_ind = 1
    cur_worst_det = Inf

    for i in 1:iterations
        #Init

        #Construct initial subset H1 from random (p+1)*n H0
        ind0 = rand(prng, ind0_dist, p+1)
        H0 = @view X[ind0, :]
        S0 = cov(H0)

        while det(S0) == 0.0
            push!(ind0 , rand(prng, ind0_dist, 1)[1])
            H0 = @view X[ind0, :]
            S0 = cov(H0)
        end

        T0 = mean(H0, dims = 1)

        D0_2 = colwise(SqMahalanobis(S0), X', vec(T0))
        ind1 = sortperm(D0_2)[1:h]
        H1 = @view X[ind1, :]
        T1, S1 = mean(H1, dims = 1), cov(H1)

        #Perform a few concentration steps to to determine is this is a good subset
        for j in 1:inner_steps
            D1_1 = colwise(Mahalanobis(S1), X', vec(T1))
            ind1 = sortperm(D1_1)[1:h]
            H1 = @view X[ind1, :]
            T1, S1 = mean(H1, dims = 1), cov(H1)
        end

        cur_det = det(S1)
        if  cur_det < cur_worst_det
            best_ind[:, cur_worst_ind] = ind1
            best_det[cur_worst_ind] = cur_det

            cur_worst_ind = argmax(best_det)
            cur_worst_det = best_det[cur_worst_ind]
        end
    end

    #Run all best subsets to convergence
    for k in 1:n_best_h
        prev_det = Inf
        cur_det = best_det[k]

        H1 = @view X[best_ind[:,k], :]
        T1, S1 = mean(H1, dims = 1), cov(H1)

        while !(cur_det <= prev_det || cur_det == 0.0)
            D1_1 = colwise(Mahalanobis(S1), X', vec(T1))
            ind1 = sortperm(D1_1)[1:h]
            H1 = @view X[ind1, :]
            T1, S1 = mean(H1, dims = 1), cov(H1)

            prev_det = cur_det
            cur_det = det(S1)
        end

        best_ind[:, k] = ind1
        best_det[k] = cur_det
    end

    #1-step reweighted estimators
    best_ind_min = best_ind[:, argmin(best_det)]
    H_min = @view X[best_ind_min, :]
    T_MCD = vec(mean(H_min, dims = 1))
    S_MCD = cov(H_min)
    DS_2 = colwise(SqMahalanobis(S_MCD), X', vec(T_MCD))
    S_MCD *= (median(DS_2) / quantile(Chisq(p), 0.5))
    D_MCD = colwise(Mahalanobis(S_MCD), X', vec(T_MCD))
    w = FrequencyWeights((D_MCD) .< sqrt(quantile(Chisq(p), 0.975))*1)

    return mean(X, w, dims = 1), cov(X, w, corrected = true)
end

#Alternative input for fastMCD
function fastMCD(X::Matrix{<:Real}, prop::Float64 = 0.5; iterations::Integer = 500, inner_steps::Integer = 2, n_best_h::Integer = 10, rng_seed::Integer = 0)
    n = size(X, 1)
    h = floor(Int64, (1 - prop) *n) + 1

    return fastMCD(X, h; iterations = iterations, inner_steps = inner_steps, n_best_h = n_best_h, rng_seed = rng_seed)
end

#Alternative input for fastMCD
function fastMCD(x::Vector{<:Real}, h::Integer = opt_h(X))
    #TODO: Add check: opt_h(X) <= h <=n

     x_dev = abs.(x .- median(x))
     ord = sortperm(x_dev)

     return mean(x[ord][1:h]), var(x[ord][1:h])
end

#Function for robust scatter based on (approximation of) 1-step reweighted MVE
function fastMVE()

end

#Calculate the optimal h for (fast)MCD given sample size
function opt_h(X::Matrix{<:Real})
    return floor(Int64, 0.5*(sum(size(X)) + 1))
end

#Calculate the theoretical breakdown for (fast)MCD given parameters
function breakdown(X::Matrix{<:Real}, h::Integer = opt_h(X))
    return (size(X, 1) - h + 1) / n
end
