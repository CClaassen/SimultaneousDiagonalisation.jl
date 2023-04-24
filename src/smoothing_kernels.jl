#Library for kernel weighing [11 variants]
 
#Function to smoothen data via weighting kernels
function kernel_smoother(X::AbstractVecOrMat{<:Real}, kernel_function::Function = epanechnikov_kernel_w, location_function::Function = weighted_mean; scale::Real = 1.0, dist_metric::Metric = euclidean)

    dist = (scale !== 0.0)*(scale !== 1.0) ? scale*data2dist(X, dist_metric = dist_metric) : data2dist(X, dist_metric = dist_metric)
    normalized_weights = kernel_function(dist)

    return location_function(X, normalized_weights)
end

#Function to smoothen data via weighting kernels with added shrinkage of outliers
function adaptive_kernel_smoother(X::AbstractVecOrMat{<:Real}, kernel_function::Function = epanechnikov_kernel_w, location_function::Function = weighted_mean, nn_param::Tuple{Int64, Float64} = (3, 1.0); scale::Real = 1.0, dist_metric::Metric = euclidean)
    #params: cut-off?, α, min_nn
    min_nn, α = nn_param[1], nn_param[2]

    n = size(X)[1]
    dist0 = (scale !== 0.0)*(scale !== 1.0) ? scale*data2dist(X, dist_metric = dist_metric) : data2dist(X, dist_metric = dist_metric)
    min_dist = partialsort.(RowVecs(dist0), min_nn)
    new_scale = minimum.(RowVecs([fill(scale, n) α*inv.(min_dist)]))
    #dist = dist0 .* new_scale
    normalized_weights = kernel_function(dist0 .* new_scale)

    return location_function(X, normalized_weights)
end

#Function for normalized weights of uniform kernel
function uniform_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 0.5 : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of triangular kernel
function triangular_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 1.0-abs(x) : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of epanechnikov kernel
function epanechnikov_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 0.75*(1.0-x^2) : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Alias of epanechnikov kernel
function parabolic_kernel_w(X::AbstractVecOrMat{<:Real})
    return epanechnikov_kernel_w(X)
end

#Function for normalized weights of quartic kernel
function quartic_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 0.9375*(1.0-x^2)^2 : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Alias of quartic kernel
function biweight_kernel_w(X::AbstractVecOrMat{<:Real})
    return quartic_kernel_w(X)
end

#Function for normalized weights of triweight kernel
function triweight_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 1.09375*(1.0-x^2)^3 : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of tricube kernel
function tricube_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 70.0/81.0*(1.0-abs(x)^3)^3 : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of cosine kernel
function cosine_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = abs(x) <= 1.0 ? 0.25*pi*cos(0.5*pi*x) : 0.0
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of gaussian kernel
function gaussian_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = inv(sqrt(2*pi))*exp(-0.5*x^2)
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of logistic kernel
function logistic_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = inv(2+exp(x)+exp(-x))
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of sigmoid kernel
function sigmoid_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = 2.0/pi*inv(exp(x)+exp(-x))
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end

#Function for normalized weights of silverman kernel
function silverman_kernel_w(X::AbstractVecOrMat{<:Real})
    kernel_func(x::Real) = 0.5*exp(-abs(x)/sqrt(2))*sin(abs(x)/sqrt(2)+0.25*pi)
    weight = kernel_func.(X)
    return weight ./ sum(weight, dims = 2)
end
