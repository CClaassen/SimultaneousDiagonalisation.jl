module SimultaneousDiagonalisation

using LinearAlgebra
using StatsBase, Statistics
using Distributions, Random #, Combinatorics, StableRNGs
using Distances, KernelFunctions, SmoothingSplines
using Optim, TSne, UMAP, LIBSVM
#using DelimitedFiles, Plots

#Resolve multi-export issue
Kernel = KernelFunctions.Kernel

##Set the correct directory
#cd(@__DIR__)

#Include all source files, grouped by contents

include("factorisations.jl")
include("component_selection.jl")
include("normality_tests.jl")

include("scatters.jl")
include("smoothing_kernels.jl")
include("reproducing_kernels.jl")
include("kernel_manipulation.jl")

include("classification.jl")
include("evaluation.jl")
include("external_methods.jl")

include("figures.jl")

include("data_manipulation.jl")
include("utilities.jl")

include("experiments.jl") #Not for eventual package
include("external_data.jl") #Not for eventual package


#Export important function, other functions can be accessed via SimultaneousDiagonalisation.FUNCTIONNAME

export gpca, ics
export component_selection
export normal_test

export mean1, cov2, mean3, cov4, fastMCD, fastMVE, lcov, rlcov, alcov, arlcov
export kernel_smoother, adaptive_kernel_smoother
export linear_kernel, polynomial_kernel, laplacian_kernel, gaussian_kernel, rational_kernel, cosine_kernel, neural_network_kernel, mahalanobis_kernel
export median_trick, quantile_trick

export ols2, logit2, svm2
export k_fold, stratified_k_fold, diagnostics, diagnostics2D, get_all_eval, mcc
export tsne2, umap2

export scatter_plot, scatter_plot_ind, contour_plot, contour_plot_ind, heatmap_plot, component_plot, bshape_plot

#No export from data_manipulation.jl
#No export from utilities.jl

#No export from experiments.jl
#No export from external_data.jl

end
