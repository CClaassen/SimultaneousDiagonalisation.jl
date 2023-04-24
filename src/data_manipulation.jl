#Functions to help with data manipulation for: moments, categorical encoding and data transformations
 
#Raw moment
function raw_moment(x::Vector{<:Real}, k::Integer)
    #@assert k >= 0
    return sum(x.^ k) / length(x)
end

#Central moment
function central_moment(x::Vector{<:Real}, k::Integer; o::Real = raw_moment(x, 1))
    #@assert k >= 0
    return k == 1 ? 0.0 : raw_moment(x .- o, k)
end

#Standardized
function standard_moment(x::Vector{<:Real}, k::Integer; o::Real = raw_moment(x, 1), s::Real = central_moment(x, 2, o=o))
    #@assert k >= 0
    return k == 1 ? 0.0 : (k == 2 ? 1.0 : raw_moment(x .- o, k) / s ^ (k /2))
end

#Raw moment per column
#TODO: reduce allocations?
function raw_moment(X::Matrix{<:Real}, k::Integer)
    return Matrix(raw_moment.(Vector.(eachcol(X)), k)')
end

#Central moment per column
function central_moment(X::Matrix{<:Real}, k::Integer; O::Matrix{<:Real} = raw_moment(X, 1))
    return raw_moment(X .- O, k)
end

#Standardized moment per column
function standard_moment(X::Matrix{<:Real}, k::Integer; O::Matrix{<:Real} = raw_moment(X, 1), S::Matrix{<:Real} = central_moment(X, 2, O=O))
    return raw_moment(X .- O, k) ./ S .^(k/2)
end

#Squared Mahalnobis distance
function mahalanobis2(X::Matrix{<:Real}, Y::Matrix{<:Real}; S::Matrix{<:Real}= cov2(X))
    return pairwise(Mahalanobis(S), X, Y, dims=1)
end

#Mahalnobis distance
function mahalanobis1(X::Matrix{<:Real}, Y::Matrix{<:Real}; S::Matrix{<:Real} = cov2(X))
    return pairwise(SqMahalanobis(S), X, X, dims=1)
end

#Encode label via vector of integers
function vec_cat_encode(x::AbstractVector)
    n = length(x)
    ret = zeros(Int64, n)

    mat_cat = mat_cat_encode(x)
    p = size(mat_cat)[2]
    z = collect(1:1:p)

    for i in 1:n
        ret[i] = dot(z,mat_cat[i,:])
    end

    return ret
end

#Encode label via one-hot encoding
function mat_cat_encode(x::AbstractVector)
    ret = unique(x) .== permutedims(x)
    return ret'
end

function transform_loc(X::AbstractVecOrMat{<:Real})
    return X .- mean1(X)
end

#01 transformations
function transform_01(x::AbstractVector{<:Real})
    min_val, max_val = minimum(x), maximum(x)
    return (x .- min_val) / (max_val - min_val)
end

function transform_01(X::AbstractMatrix{<:Real})
    return mapslices(transform_01, X, dims = 1)
end

#Z-score transformations
function transform_z(x::AbstractVector{<:Real})
        return (x .- mean(x)) / std(x)
end

function transform_z(X::AbstractMatrix{<:Real})
    return mapslices(transform_z, X, dims = 1)
end

#Robust transformations
function transform_rob(x::AbstractVector{<:Real}; med = median(x), mad_n = mad(x, normalize = true), dist = Normal(med, mad_n))
    cor = quantile(dist, 0.75)
    return cor * (x .- med) / mad_n
end

function transform_rob(X::AbstractMatrix{<:Real})
    return mapslices(transform_rob, X, dims = 1)
end

function weighted_mean(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real} = Real[])
    if isempty(W)
        W = ones(n, n) ./ fill(n,n)
    end

    return W*X
end

function geometric_median(X::AbstractMatrix{<:Real}, w::AbstractVector{<:Real} = Real[]; iter_cap::Int=1000, eps::Number=1e-6)
    n,p = size(X)

    if isempty(w)
        w = ones(n)
    end

    if sum(w)  == 1.0
        return vec(mean(X, weights(w), dims=1))
    end

    old = vec(mean(X, weights(w), dims=1))
    new = zeros(p)

    iter = 0
    while norm(new - old) > eps
        num = zeros(p)
        denom = 0.0

        for i in 1:n
            d = norm(new - X[i,:]) / w[i]
            num += X[i,:] / d
            denom += inv(d)
        end

        old = new
        new = num / denom
        iter += 1
    end

    return new
end

function geometric_median(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real} = Real[]; iter_cap::Int=1000, eps::Number=1e-6)
    n,p = size(X)
    res = zeros(n, p)

    if isempty(W)
        W = ones(n, n)
    end

    #Calculate the geometric median for all weight vectors
    for i in 1:n
        res[i,:] = geometric_median(X, W[i,:], iter_cap = iter_cap, eps = eps)
    end

    return res
end

function weighted_median(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real} = [])
    n, p = size(X)
    #check if weights are nXn ?
    res = zeros(n,p)

    #weight = sort.(RowVecs(W), rev = true)
    ind = sortperm.(RowVecs(W), rev = true)

    for i in 1:n
        cur_w = W[i, ind[i]]

        pivot_ind = 1
        lower_w, upper_w = 0.0, 1.0 - cur_w[1]
        while lower_w < 0.5
            lower_w += cur_w[pivot_ind]
            upper_w -= cur_w[pivot_ind+1]
            pivot_ind += 1
        end

        res[i, :] = X[ind[i][pivot_ind - 1], :]
    end

    return res
end

#Auxiliary function for transforming data to pairwise distance for a given metric
#Metric can be any from https://github.com/JuliaStats/Distances.jl
function data2dist(X::AbstractVecOrMat{<:Real}; dist_metric::Metric = euclidean)
    return pairwise((x,y) -> dist_metric(x,y), RowVecs(X))
end

function fix_signs!(V; dims::Int = 1)
    if dims in (1,2)
        return V .*= 2*(V[findmax(abs.(V), dims = dims)[2]] .> 0.0) .-1
    else
        error("Array dimension to fix signs over invalid.")
    end
end
