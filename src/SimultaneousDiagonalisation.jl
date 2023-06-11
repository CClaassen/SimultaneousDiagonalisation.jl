module SimultaneousDiagonalisation

using LinearAlgebra
using StatsBase, Statistics
using Distributions, Random #, StableRNGs
using Distances, KernelFunctions, SmoothingSplines
using Optim, TSne, UMAP, LIBSVM
#using DelimitedFiles, Plots

#Resolve multi-export issue
Kernel = KernelFunctions.Kernel


#Include all source files, grouped here by contents

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

include("experiments.jl") #Only for reproducing results, nothing exported for this package
include("external_data.jl") #Only for reproducing results, nothing exported for this package


#Export important functions defined by this package, other functions can be accessed via SimultaneousDiagonalisation.FUNCTIONNAME
export
        #factorisations
        gpca,
        ics,

        #component_selection
        component_selection,

        #normality_tests
        normal_test,

        #scatters
        mean1,
        cov2,
        mean3,
        cov4,
        fastMCD,
        fastMVE,
        lcov,
        rlcov,
        alcov,
        arlcov,

        #smoothing_kernels
        kernel_smoother,
        adaptive_kernel_smoother,

        #reproducing_kernels
        linear_kernel,
        polynomial_kernel,
        laplacian_kernel,
        gaussian_kernel,
        rational_kernel,
        cosine_kernel,
        neural_network_kernel,
        mahalanobis_kernel,

        #kernel_manipulation
        median_trick,
        quantile_trick,

        #classification
        ols2,
        logit2,
        svm2,

        #evaluation
        k_fold,
        stratified_k_fold,
        diagnostics,
        diagnostics2D,
        get_all_eval,
        mcc,

        #external_methods
        tsne2,
        umap2,

        #figures
        scatter_plot,
        scatter_plot_ind,
        contour_plot,
        contour_plot_ind,
        heatmap_plot,
        component_plot,
        bshape_plot

        #No exports from data_manipulation

        #No exports from utilities

        #No exports from experiments

        #No exports from external_data

#If module is included locally, apply 'using .SimultaneousDiagonalisation' in REPL
end
