#RKHS kernels [19 variants]
 
#Main function to generate Reproducing/PSD kernels
function kernel(X::AbstractVecOrMat{<:Real}, k::Kernel = RBFKernel(), input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0,
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    #Check if observations are in column order instead of rows
    X2 = use_cols ? ColVecs(X) : RowVecs(X)

    #Apply possible output transformation
    if output_scale ≠ 1.0
        k = scale_output(k, output_scale)
    end

    #Apply possible input transformation
    if input_scale isa Vector{<:Real}
        k = ard_transform(k, input_scale)
    elseif input_scale isa Matrix{<:Real}
        k = linear_transform(k, input_scale)
    elseif input_scale ≠ 1.0
        k = scale_input(k, input_scale)
    end

    if normalize
        k = kernel_normalized(k)
    end

    if 0.0 < use_nystrom <= 1.0
        W, C = nystrom_ratio(k, X2, use_nystrom)
        K = C'*W*C
    else
        K = kernelmatrix(k, X2)
    end

    K = center ? kernel_centered(K) : K

    return K
end

#Function for a linear kernel
function linear_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0,  param::Tuple = (0.0,),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = LinearKernel(c = param[1])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a polynomial kernel
function polynomial_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (2, 0.0),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = PolynomialKernel(degree = param[1], c = param[2])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a rational kernel
function rational_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (2.0, Euclidean()),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = RationalKernel(α = param[1], metric = param[2])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a rational quadratic kernel
function rational_quadratic_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0,  param::Tuple = (2.0, Euclidean()),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = RationalQuadraticKernel(α = param[1], metric = param[2])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a gamma rational quadratic kernel
function gamma_rational_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (2.0, 1.0, Euclidean()),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = GammaRationalKernel(α = param[1], γ = param[2], metric = param[3])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for an Abelian kernel
function abelian_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = ExponentialKernel(metric = Cityblock())

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Laplacian / exponential kernel
function laplacian_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = ExponentialKernel()

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Gaussian / RBF / Squared Exponential kernel
function gaussian_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = SqExponentialKernel()

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a gamma exponential kernel
function gamma_exponential_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (1.0, Euclidean()),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = SqExponentialKernel(γ = param[1], metric = param[2])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Gibbs kernel #TODO: fix parameter func
function gibbs_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (x -> 1.0,),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = GibbsKernel(lengthscale = param[1])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for an exponentiated kernel
function exponentiated_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = ExponentiatedKernel()

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a cosine kernel
function cosine_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (Euclidean(),),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = CosineKernel(metric = param[1])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a periodic kernel
function periodic_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (ones(size(X)[2]),),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = PeriodicKernel(r = param[1])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Neural Network kernel
function neural_network_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = NeuralNetworkKernel()

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a fractional brownian motion kernel
function fbm_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (0.5,),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = FBMKernel(h = param[1])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Mahalanobis kernel
function mahalanobis_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    Q = cholesky(cov(X))
    k = SqExponentialKernel() ∘ LinearTransform(sqrt(2.0) .* Q.U)

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Wiener kernel
function wiener_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (0,),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = WienerKernel(i = param[1])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Gabor kernel #TODO: check param arguments
function gabor_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (IdentityTransform(), IdentityTransform()),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = gaborkernel(sqexponential_transform = param[1], cosine_transform = param[2])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end

#Function for a Matern kernel
function matern_kernel(X::AbstractVecOrMat{<:Real}, input_scale::Union{Real, AbstractVecOrMat{<:Real}} = 1.0, param::Tuple = (1.5, Euclidean()),
    output_scale::Real = 1.0; center::Bool = true, normalize::Bool = false, use_nystrom::Real = 0.0, use_cols::Bool = false)

    k = MaternKernel(ν = param[1], metric = param[2])

    return kernel(X, k, input_scale, output_scale; center = center, normalize = normalize, use_nystrom = use_nystrom, use_cols = use_cols)
end
