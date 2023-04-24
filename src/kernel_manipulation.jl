#Function for manipulating kernels
 
#Function to scale the input
function scale_input(k::Kernel, s::Real = 1.0)
    return k ∘ ScaleTransform(s)
end

#Function to scale the output bt setting the kernel variance
function scale_output(k::Kernel, l::Real = 1.0)
    return l * k
end

#Function to scale input dimensions seperately
function ard_tranform(k::Kernel, v::AbstractVector{<:Real})
    return k ∘ ARDTransform(v)
end

#Function to scale input dimensions seperately
function linear_tranform(k::Kernel, V::AbstractMatrix{<:Real})
    return k ∘ LinearTransform(X)
end

#Center an instantiated kernel in feature space
function kernel_centered(K::AbstractMatrix{<:Real})
    n,p = size(K)
    C = eye(p) - ones(p,p) / n
    return C*K*C
end

#Normalize the kernel
function kernel_normalized(k::Kernel)
    return NormalizedKernel(k)
end

#Function to convert an instantiated kernel to a (pseudo) distance
function kernel2distance(K_x::Matrix{<:Real}, K_y::Matrix{<:Real}, K_xy::Matrix{<:Real})
    #Note: only a pseudo metric as d(x,y) = 0 is possible
    return sqrt(K_x + K_y - 2.0*K_xy)
end

#Perform a Nystrom approximate decomposition of the kernel based on a select few indices
function nystrom_ind(k::Kernel, X::Union{RowVecs,ColVecs}, ind::AbstractVector{<:Integer})
    F = nystrom(k, X, ind)
    return F.W, F.C
end

#Perform a Nystrom approximate decomposition of the kernel based on random subset of observations
function nystrom_ratio(k::Kernel, X::Union{RowVecs,ColVecs}, ratio::Real = 1.0)
    F = nystrom(k, X, ratio)
    return F.W, F.C
end

#Return a kernel consisting of the sum of multiple kernels
function kernel_sum(k::Vector{Kernel})
    return KernelSum(k)
end

#Return a kernel consisting of the sum of multiple kernels
function kernel_prod(k::Vector{Kernel})
    return KernelProduct(k)
end

function median_trick(X::AbstractVecOrMat{<:Real})
    return inv(2*median(data2dist(X)))
end

function quantile_trick(X::AbstractVecOrMat{<:Real}, r::Real = 0.5)
    return inv(2*quantile(vec(data2dist(X)), r))
end
