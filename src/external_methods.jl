#Functions for other mdethods from other packages
#using TSne, UMAP, LIBSVM, Flux
using RCall

#Function for performing t-SNE
function tsne2(X::AbstractMatrix{<:Real}, n_components::Integer = 2, perplexity::Real = 30.0, reduce_dims::Integer = 0; max_iter::Integer = 1000,
    pca_init::Bool = true, distance::Bool = false, stop_cheat_iter = floor(Int64, 0.25*max_iter), progress = false)
    return tsne(X, n_components, reduce_dims, max_iter, perplexity;
    pca_init = pca_init, distance = distance, stop_cheat_iter = stop_cheat_iter, progress = progress)
end

#Function for performing UMAP
function umap2(X::AbstractMatrix{<:Real}, n_components::Integer = 2, n_neighbours::Integer = 15, min_dist::Real = 0.1; max_iter::Integer = 300,
    metric::SemiMetric = Euclidean())

    return Matrix(transpose(umap(X', n_components, n_neighbors = n_neighbours, metric = metric, min_dist = min_dist, n_epochs = max_iter)))
end

function fastMCD2(X::Matrix{<:Real}, alpha::Float64 = 0.5; seed::Integer = 1, nsamp::Integer = 1000)
    R"library(robustbase)"
    R"library(rrcov)"
    @rput X
    @rput alpha
    @rput seed
    @rput nsamp
    R"res <- CovMcd(X, alpha = alpha, seed = set.seed(seed), nsamp = nsamp)"
    R"mu <- res$center"
    R"sig <- res$cov"
    @rget mu
    @rget sig

    return sig, mu
end

function fastMVE2(X::Matrix{<:Real}, alpha::Float64 = 0.5; seed::Integer = 1, nsamp::Integer = 1000)
    R"library(robustbase)"
    R"library(rrcov)"
    @rput X
    @rput alpha
    @rput seed
    @rput nsamp
    R"res <- CovMve(X, alpha = alpha, seed = set.seed(seed), nsamp = nsamp)"
    R"mu <- res$center"
    R"sig <- res$cov"
    @rget mu
    @rget sig

    return sig, mu
end
