#Functions for various plot and figures
 
using Plots

#Plot the output of a decomposition
function scatter_plot(X::Matrix{<:Real}, Y::Matrix{<:Real}, label1::AbstractVector = ones(size(X,2)), label2::AbstractVector = label1; title::String = "", flip::Tuple{Bool, Bool} = (true, false))
    !(size(X) == size(Y)) && throw(ArgumentError("Embeddings must have the same amount of dimensions."))

    n, p = size(X)
    plt = Matrix{Plots.Plot{Plots.GRBackend}}(undef, p, p)
    color_ind1, color_ind2 = ones(Int64, n), ones(Int64, n)

    n_z = length(unique(label1))
    clr = palette(get_default_colour())

    color_ind1 = vec_cat_encode(label1)
    color_ind2 = vec_cat_encode(label2)

    #TODO: improve plot and add colouring support
    group_ind1 = mat_cat_encode(color_ind1)
    group_ind2 = mat_cat_encode(color_ind2)
    g = size(group_ind1)[2]


    for (i,j) in [(i,j) for i in 1:p, j in 1:p]
        if i == j && j < p+1
            #plt[i,j] = bshape_plot([X[:,i][group_ind1[:,k]] for k in 1:g])
            plt[i,j] = bshape_plot([X[:,i] Y[:,i]], color_ind = 6)
        elseif i > j && i <= p
            if flip[1]
                plt[i,j] = scatter(X[:,j], X[:,i], color = clr[color_ind1], ticks = false)
            else
                plt[i,j] = scatter(X[:,i], X[:,j], color = clr[color_ind1], ticks = false)
            end
        else
            if flip[2]
                plt[i,j] = scatter(Y[:,j], Y[:,i], color = clr[color_ind2], ticks = false)
            else
                plt[i,j] = scatter(Y[:,i], Y[:,j], color = clr[color_ind2], ticks = false)
            end
        end
    end

    ##TODO: only Works until p = 12
    if p < 6
        scl = minimum((0.825, 0.3 + 0.125p))
    else
        scl = (0.825 + 0.125 * (p - 6))
    end
    Plots.scalefontsizes(scl)

    plt2 = plot(permutedims(plt)..., layout = (p,p), size = (200*p, 200*p), legend = false, plot_title = title, titlefontsize = 2 + p)

    Plots.scalefontsizes(1.0/scl)

    return plt2
end

function scatter_plot_ind(X::Matrix{<:Real}, Y::Matrix{<:Real}, label1::AbstractVector = ones(size(X,2)), label2::AbstractVector = label1,
    ind::Tuple{Vararg{Integer}} = (1,2,3), vals1::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]), vals2::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]);
    title::Tuple{String,String} = ("",""), flip::Tuple{Bool, Bool} = (true, false))

    !(minimum((size(X)[2], size(Y)[2])) >= length(ind)) && throw(ArgumentError("Invalid indices to access given matrices."))
    if !prod(isempty.(title)) && !isempty(vals1) && !isempty(vals2)
        title = method_title(title[1], title[2], diag(vals1), diag(vals2), ind, ind)
    elseif !isempty(vals1) && !isempty(vals2)
        title = method_title("Method 1", "Method 2", diag(vals1), diag(vals2), ind, ind)
    else
        title = ""
    end
    return scatter_plot(X[:,collect(ind)], Y[:,collect(ind)], label1, label2, title = title, flip = flip)
end

function scatter_plot_ind(X::Matrix{<:Real}, Y::Matrix{<:Real}, label1::AbstractVector = ones(size(X,2)), label2::AbstractVector = label1,
    ind1::Tuple{Vararg{Integer}} = (1,2,3), ind2::Tuple{Vararg{Integer}} = ind1, vals1::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]), vals2::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]);
    title::Tuple{String,String} = ("",""), flip::Tuple{Bool, Bool} = (true, false))

    !(minimum((size(X)[2], size(Y)[2])) >= length(ind1)) && throw(ArgumentError("Invalid indices to access given matrices."))
    if !prod(isempty.(title)) && !isempty(vals1) && !isempty(vals2)
        title = method_title(title[1], title[2], diag(vals1), diag(vals2), ind1, ind2)
    elseif !isempty(vals1) && !isempty(vals2)
        title = method_title("Method 1", "Method 2", diag(vals1), diag(vals2), ind1, ind2)
    else
        title = ""
    end
    return scatter_plot(X[:,collect(ind1)], Y[:,collect(ind2)], label1, label2, title = title, flip = flip)
end

#Function for generating a title of the plot of the output of a decomposition
function method_title(method1::String, method2::String, kap1::AbstractArray, kap2::AbstractArray, ind1::Tuple{Vararg{Integer}} = collect(1:length(kap1)),
    ind2::Tuple{Vararg{Integer}} = collect(1:length(kap2)); trim_title::Bool = false, trim_val::Bool = false)
    #!(size(kap1) == size(kap2)) && throw(DimensionMismatch("Method dimensions do not match."))

    k(i) = text_subscript("κ", i)
    l(i) = text_subscript("λ", i)

    if trim_title
            title1 =  method1 * " vs. " * method2
    else
            title1 =  method1 * " (lower left) vs. " * method2 * " (upper right)"
    end

    title2 = prod(["$(k(i))" * "=" * "$(round.(kap1[i], digits = 2)), " for i in ind1])[1:end-2]
    title3 = prod(["$(l(i))" * "=" * "$(round.(kap2[i], digits = 2)), " for i in ind2])[1:end-2]

    if trim_val
        title1
    elseif length(ind1) < 6
        return title1 * "\n" * title2 * "  &  " * title3  * "\n"
    else
        return title1 * "\n" * title2 * "\n" * title3  * "\n"
    end
end

function contour_plot(X::Matrix{<:Real}, Y::Matrix{<:Real}, label1::AbstractVector = ones(size(X,2)), label2::AbstractVector = label1;
    title::String = "", flip::Tuple{Bool, Bool} = (true, false), contour_func::Function = svm2)
    !(size(X) == size(Y)) && throw(ArgumentError("Embeddings must have the same amount of dimensions."))

    n, p = size(X)
    plt = Matrix{Plots.Plot{Plots.GRBackend}}(undef, p, p)
    color_ind1, color_ind2 = ones(Int64, n), ones(Int64, n)

    color_ind1 = vec_cat_encode(label1)
    color_ind2 = vec_cat_encode(label2)

    #TODO: improve plot and add colouring support
    group_ind1 = mat_cat_encode(color_ind1)
    group_ind2 = mat_cat_encode(color_ind2)
    g = size(group_ind1)[2]


    for (i,j) in [(i,j) for i in 1:p, j in 1:p]
        if i == j && j < p+1
            #plt[i,j] = bshape_plot([X[:,i][group_ind1[:,k]] for k in 1:g])
            plt[i,j] = bshape_plot([X[:,i] Y[:,i]], color_ind = 6)
        elseif i > j && i <= p
            if flip[1]
                plt[i,j] = contour_plot([X[:,j] X[:,i]], label1, (1,2), contour_step = 100, hide_title = true, show_ticks = false, contour_func = contour_func)
            else
                plt[i,j] = contour_plot([X[:,i] X[:,j]], label1, (1,2), contour_step = 100, hide_title = true, show_ticks = false, contour_func = contour_func)
            end
        else
            if flip[2]
                plt[i,j] = contour_plot([Y[:,j] Y[:,i]], label2, (1,2), contour_step = 100, hide_title = true, show_ticks = false, contour_func = contour_func)
            else
                plt[i,j] = contour_plot([Y[:,i] Y[:,j]], label2, (1,2), contour_step = 100, hide_title = true, show_ticks = false, contour_func = contour_func)
            end
        end
    end

    ##TODO: only Works until p = 12
    if p < 6
        scl = minimum((0.825, 0.3 + 0.125p))
    else
        scl = (0.825 + 0.125 * (p - 6))
    end
    Plots.scalefontsizes(scl)

    plt2 = plot(permutedims(plt)..., layout = (p,p), size = (200*p, 200*p), legend = false, plot_title = title, titlefontsize = 2 + p)

    Plots.scalefontsizes(1.0/scl)

    return plt2
end

function contour_plot_ind(X::Matrix{<:Real}, Y::Matrix{<:Real}, label1::AbstractVector = ones(size(X,2)), label2::AbstractVector = label1,
    ind::Tuple{Vararg{Integer}} = (1,2,3), vals1::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]), vals2::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]);
    title::Tuple{String,String} = ("",""), flip::Tuple{Bool, Bool} = (true, false), contour_func::Function = svm2)

    !(minimum((size(X)[2], size(Y)[2])) >= length(ind)) && throw(ArgumentError("Invalid indices to access given matrices."))

    if contour_func == ols2
        str = "Simple LPM: "
    elseif contour_func == logit2
        str = "Logistic: "
    elseif  contour_func == svm2
        str = "Linear SVM: "
    end

    if !prod(isempty.(title)) && !isempty(vals1) && !isempty(vals2)
        title = method_title(str*title[1], title[2], diag(vals1), diag(vals2), ind, ind)
    elseif !isempty(vals1) && !isempty(vals2)
        title = method_title(str*"Method 1", "Method 2", diag(vals1), diag(vals2), ind, ind)
    else
        title = ""
    end
    return contour_plot(X[:,collect(ind)], Y[:,collect(ind)], label1, label2, title = title, flip = flip, contour_func = contour_func)
end

function contour_plot_ind(X::Matrix{<:Real}, Y::Matrix{<:Real}, label1::AbstractVector = ones(size(X,2)), label2::AbstractVector = label1,
    ind1::Tuple{Vararg{Integer}} = (1,2,3), ind2::Tuple{Vararg{Integer}} = ind1, vals1::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]), vals2::Diagonal{<:Real,<:Vector{<:Real}} = Diagonal(Real[]);
    title::Tuple{String,String} = ("",""), flip::Tuple{Bool, Bool} = (true, false), contour_func::Function = svm2)

    !(minimum((size(X)[2], size(Y)[2])) >= length(ind1)) && throw(ArgumentError("Invalid indices to access given matrices."))

    if contour_func == ols2
        str = "Simple LPM: "
    elseif contour_func == logit2
        str = "Logistic: "
    elseif  contour_func == svm2
        str = "Linear SVM: "
    end

    if !prod(isempty.(title)) && !isempty(vals1) && !isempty(vals2)
        title = method_title(str*title[1], title[2], diag(vals1), diag(vals2), ind1, ind2)
    elseif !isempty(vals1) && !isempty(vals2)
        title = method_title(str*"Method 1", "Method 2", diag(vals1), diag(vals2), ind1, ind2)
    else
        title = ""
    end
    return contour_plot(X[:,collect(ind1)], Y[:,collect(ind2)], label1, label2, title = title, flip = flip, contour_func = contour_func)
end

function contour_plot(X::Matrix{<:Real}, z::AbstractVector, ind::Tuple{Integer, Integer} = (1,2);
    contour_func::Function = svm2, contour_step::Integer = 250, title::String = "", hide_title::Bool = false,
    leg_pos::Union{Bool,Symbol} = :outerright, show_ticks::Bool = true, hide_labels::Bool = true)

    ord = sortperm(z)
    X, z = X[ord,:], z[ord]

    if !(eltype(z)<:Number)
        z_leg = copy(z)
        z = vec_cat_encode(z)
    else
        z_leg = copy(z)
    end

    n_z = length(unique(z))
    clr = palette(get_default_colour())
    Z = X[:, collect(ind)]

    function adj(E::AbstractMatrix{<:Real}, s::Real = 1.06)
        for i in (1,3)
            if prod((E[i]>0,E[i+1]>0))
                E[i] = E[i] * inv(s)
                E[i+1] = E[i+1] * s
            elseif prod((E[i]<0,E[i+1]>0))
                E[i] = E[i] * s
                E[i+1] = E[i+1] * s
            elseif prod((E[i]<0,E[i+1]<0))
                E[i] = E[i] * s
                E[i+1] = E[i+1] * inv(s)
            end
        end
        return E
    end
    extremes = [minimum(Z, dims = 1); maximum(Z, dims = 1)]
    extremes = adj(extremes)
    rng1 = range(extremes[1], extremes[2], length = contour_step)
    rng2 = range(extremes[3], extremes[4], length = contour_step)

    if contour_func in (ols2, logit2, svm2)
        contour = mapreduce(collect, hcat, Iterators.product(rng1, rng2))
        acc, _, _, _, contour_pred = contour_func(Z, contour', z, ones(Integer, size(contour)[2]), ret_test = true)
        acc = round(acc, digits = 4)
    else
        throw(ArgumentError("Classication function ($contour_func) unrecognized."))
    end

    if hide_title
        title = ""
    elseif isempty(title)
        if contour_func == ols2
            str = "Simple LPM"
        elseif contour_func == logit2
            str = "Logistic"
        elseif  contour_func == svm2
            str = "Linear SVM"
        end
        title = str*" Decision Region Boundary" * "\n (Accuracy = $acc)"
    else
        title = title * "\n (Accuracy = $acc) \n"
    end

    if !hide_labels
        x_label = "Component $(ind[1])"
        y_label = "Component $(ind[2])"
    else
        x_label = ""
        y_label = ""
    end

    clr2 = mat_cat_encode(z)*clr[1:n_z]

    if length(unique(contour_pred)) != 1
        plt = contourf(
            rng1,
            rng2,
            contour_pred;
            levels = n_z,
            color = clr[1:n_z],
            alpha = 0.66,
            leg = :none,
            xlabel = x_label,
            ylabel = y_label,
            xlims = (extremes[1],extremes[2]),
            ylims = (extremes[3],extremes[4]),
            widen = false
            )
        scatter!(Z[:,1], Z[:,2], group = z_leg, color = clr2,
        title = title, leg = leg_pos, ticks = show_ticks)
    else
        plt = hspan([extremes[3], extremes[4]], color = clr[contour_pred[1]], fillalpha = 0.3, label = false, legend = false, widen = true)
        vspan!([extremes[1], extremes[2]], color = clr[contour_pred[1]], fillalpha = 0.33, label = false, legend = false, widen = true)
        scatter!(Z[:,1], Z[:,2], group = z_leg, color = clr2, ticks = show_ticks,
        title = title, leg = leg_pos, xlims = (extremes[1],extremes[2]), ylims = (extremes[3],extremes[4]), widen = false)
    end

    return plt
end

function fast_contour_plot(X::Matrix{<:Real}, z::AbstractVector, ind::Tuple{Integer, Integer} = (1,2))
    return contour_plot(X, z, ind; contour_step = 100)
end

function heatmap_plot(X::Matrix{<:Integer}, z::AbstractVector; title::String = "")
    n,p = size(X)
    #sort!(z)
    labels = sort!(unique(z))
    #clr = cgrad([distinguishable_colors(5)[4], distinguishable_colors(5)[5]])
    clr = cgrad([:teal, :darkgreen])

    if eltype(labels)<:Number
        labels = string.(labels)
    else
        labels = convert.(String, labels)
    end
    n_z = length(labels)
    mcc_score = round(mcc(X), digits = 4)

    if p < 6
        scl = minimum((0.825, 0.3 + 0.125p))
    else
        scl = (0.825 + 0.125 * (p - 6))
    end

    if isempty(title)
        if p == 2
            title = "Confusion Matrix of Binary Classification"
        else
            title = "Confusion Matrix of Multiclass Classification"
        end
    end

    Plots.scalefontsizes(scl)

    plt = heatmap(labels, labels, X./ sum(X, dims = 2), xmirror = true, yflip = true,
    color = clr, size = (n_z*150,n_z*150), tickfontsize=floor(Integer,8+n_z/2),
    xlabel = "\$Predicted\$ \$Class\$", ylabel = "\$Expected\$ \$Class\$", labelfontsize = floor(Integer,12+n_z/2),
    title = title * "\n (Matthew's Correlation = $mcc_score)")
    annotate!([(j -0.5, i-0.5, text(X[i,j], floor(Integer,9+n_z/2), :white)) for i in 1:n for j in 1:p])

    Plots.scalefontsizes(inv(scl))
    return plt
end

function component_plot(X::Matrix{<:Real}, ind::Integer, label::AbstractVector; title::String = "", mix::Bool = true, leg_pos::Union{Bool,Symbol} = :outerright)
    n,p = size(X)
    clr = get_default_colour()

    if isempty(title)
        title = "Visualisation of component $ind"
    end

    if mix
        ord = randperm(Xoshiro(1234), n)
    else
        ord = 1:n
    end

    color_ind = vec_cat_encode(label)
    plt = scatter(1:n, X[ord,ind], color = clr[color_ind[ord]], group = label[ord], xticks = false, leg = leg_pos, title = title)
    return plt
end

#Functions for making a beta-shape plot (alternative to qq-plot)

#Plot the beta-shape of the data
function bshape_plot(y::Vector{<:Real}; boundary::Real = 0.025, npoints::Integer = 10000, add_ref::Bool = true, ref_dist::UnivariateDistribution = Normal(), color_ind::Integer = 1, plt::Plots.Plot{Plots.GRBackend} = plot())

    n = length(y)
    #y_sort = @view sort(y)

    sup = (quantile(y, boundary), quantile(y, 1.0 - boundary))
    rng = LinRange(sup[1], sup[2], n)

    emp_cdf = ecdf(y)
    xs = LinRange(0.0, 1.0, n)
    ys = bshape.(emp_cdf.(rng))


    spline = fit(SmoothingSpline, xs, ys, inv(npoints))
    xs2 = LinRange(0.0, 1.0, npoints)
    ys2 = SmoothingSplines.predict(spline, xs2)
    ys2[ys2 .< 0.0] .= 0.0
    ys2  ./= maximum(ys2)

    clr = theme_palette(:auto)[color_ind]

    add_ref && bshape_plot(ref_dist; boundary = boundary, npoints = npoints, plt = plt)
    plot!(plt, xs, ys, legend = false, color = clr, linestyle = :dash, linealpha = 0.45)
    return plot!(plt, xs2, ys2, legend = false, color = clr)
    #return plot!(plt, xs2, ys2, legend = false, color = clr, fill = (zero(xs), 0.5, clr))
end

function bshape_plot(X::Matrix{<:Real}; boundary::Real = 0.025, npoints::Integer = 10000, add_ref::Bool = true, ref_dist::UnivariateDistribution = Normal(), color_ind::Integer = 1)
    n,p = size(X)

    plt = bshape_plot(X[:,1]; color_ind = color_ind, boundary = boundary, npoints = npoints, ref_dist = ref_dist)
    for i in 2:p
        bshape_plot(X[:,i]; color_ind = color_ind + i - 1, boundary = boundary, npoints = npoints, ref_dist = ref_dist, plt = plt)
    end

    return plt
end

function bshape_plot(X::Vector{<:Vector{Float64}}; boundary::Real = 0.025, npoints::Integer = 10000, add_ref::Bool = true, ref_dist::UnivariateDistribution = Normal())
    p = length(X)

    plt = bshape_plot(X[1]; color_ind = 1, boundary = boundary, npoints = npoints, ref_dist = ref_dist)
    for i in 2:p
        bshape_plot(X[i]; color_ind = i, boundary = boundary, npoints = npoints, ref_dist = ref_dist, plt = plt)
    end

    return plt
end

function bshape_plot(dist::UnivariateDistribution; boundary::Real = 0.025, npoints::Integer = 10000, plt::Plots.Plot{Plots.GRBackend} = plot(), clr::Symbol = :black, show_legend::Union{Bool,Symbol} = false)

    sup = (quantile(dist, boundary), quantile(dist, 1.0 - boundary))
    rng = LinRange(sup[1], sup[2], npoints)

    xs = LinRange(0.0, 1.0, npoints)
    ys = bshape.(cdf.(dist, rng))

    return plot!(plt, xs, ys, color = clr, linestyle = :dash, legend = show_legend)
end

function bshape(x::Real)
    return 2.0 * x^(1-x) * (1-x)^x
end

function get_default_colour(clr::PlotUtils.ContinuousColorGradient = cgrad([palette(:seaborn_bright).colors.colors; RGB(100/256,0,0); RGB(0.5,0.5,0)]))
    #Other useful Colorschemes: :default, :tab10, :tab20b
    return clr
end

function get_different_colour(n::Integer)
    return cgrad(reverse(distinguishable_colors(n+1)))
end
