#Functions for selecting components based on eigenvalues or components
 
function component_selection(X::AbstractMatrix{<:Real}, Z::AbstractMatrix{<:Real}, λ::AbstractVector{<:Real}, comp_select::Function = retain_all, comp_select_args::Tuple = ())

    #Select the components to keep
    if comp_select in (retain_all, retain_first, retain_last, retain_var, kramer_rule, cluster_priori)
        retain_ind = comp_select(λ, comp_select_args...)
        perm = collect(1:length(retain_ind))
    elseif comp_select in (normality, pick_tsne, marginal_div, joint_div, batch_joint_div)
        retain_ind = comp_select(X, Z, comp_select_args...)
        perm = sortperm(λ[retain_ind], rev = true)
    else
        throw(ArgumentError("Function for component selection unrecognised."))
    end

    return retain_ind, perm
end

#Simple function that does not discard any components
function retain_all(λ::AbstractVector{<:Real})
    return collect(1:length(λ))
end

function retain_first(λ::AbstractVector{<:Real}, k::Integer = length(λ))
    @assert k <= length(λ)
    return collect(1:k)
end

function retain_last(λ::AbstractVector{<:Real}, k::Integer = length(λ))
    @assert k <= length(λ)
    rng = collect(1:length(λ))
    return rng[end-k+1:end]
end

#Retain components such that a certain proportion of variance is retained
function retain_var(λ::AbstractVector{<:Real}, cut_off::Real = 1.0)
    @assert 0.0 <= cut_off <= 1.0

    λ  /= sum(λ)
    var_prop = [sum(λ[1:i]) for i in 1:length(λ)]
    last_ind = 1 + sum(cut_off .> var_prop)

    return collect(1:last_ind)
end

#Only retain components that contribute more than uniform
function kramer_rule(λ::AbstractVector{<:Real})
    λ  /= sum(λ)
    last_ind = sum(inv(length(λ)) .>= λ)

    return collect(1:last_ind)
end

#Only retain q-1 components based on the distance from the eigenvalue median
function cluster_priori(λ::AbstractVector{<:Real}, q::Integer = length(λ))
    return sortperm(abs.(λ .- median(λ)))[1:q-1]
end

#Retain components until a test for normality fails
function normality(X::AbstractMatrix{<:Real}, Y::AbstractVecOrMat{<:Real}, test_function::Function = jarque_bera, α::Float64 = 0.05)
    test_bool = Bool.(normal_test.(ColVecs(Y)))
    last_ind = minimum(findfirst(x -> x == false, test_bool))
    return collect(1:last_ind)
end

#Pick the best components according to loss function of tsne, useful in visualization
function pick_tsne(X::AbstractMatrix{<:Real}, Y::AbstractVecOrMat{<:Real}, red_dim::Integer = 2, perp::Real = 15.0)
    n,p = size(Y)
    best_ind = [0,0]
    best_sol = Inf

    temp_sol = Inf
    @info "Picking best solution from $(length(combinations(1:p,red_dim))) configurations."
    for i in combinations(1:p,red_dim)
        temp_sol = tsne_loss(X, Y[:,i], perp)
        if temp_sol <= best_sol
            best_sol = temp_sol
            best_ind = i
        end
    end

    return best_ind
end

#Select q components that have the least individual divergence from the original data
function marginal_div(X::AbstractMatrix{<:Real}, Y::AbstractVecOrMat{<:Real}, red_dim::Integer = 2, perp::Real = 15.0; div_measure::Function = kl_div)
    n, p = size(Y)
    @assert(red_dim <= p)


    D2_X = data2dist(X).^2
    σ2 = opt_beta(D2_X, perp)[2]

    P = exp.(-0.5 * D2_X ./ σ2) - I
    P_i = P ./ sum(P, dims = 1)
    P_j = P ./ sum(P', dims = 1)
    P = 0.5*(P_i+P_j)

    marginal_div = zeros(p)
    D2_Y = zeros(n,n)
    Q = zeros(n,n)
    for i in 1:p
        D2_Y = data2dist(Y[:,i]).^2
        Q = inv.(1.0 .+ D2_Y) - I
        Q = Q ./ sum(Q, dims = 1)

        marginal_div[i] = div_measure(P, Q)
    end

    ord = sortperm(marginal_div, rev = true)
    opt_ind = ord[1:red_dim]

    #display(plot(marginal_div[ord]))

    return opt_ind
end

#Select q components by iteratively adding components to a base set, based on least marginal div
function joint_div(X::AbstractMatrix{<:Real}, Y::AbstractVecOrMat{<:Real}, red_dim::Integer = 2, perp::Real = 15.0; div_measure::Function = kl_div)
    n, p = size(Y)
    @assert(red_dim <= p)

    D2_X = data2dist(X).^2
    σ2 = opt_beta(D2_X, perp)[2]

    P = exp.(-0.5 * D2_X ./ σ2) - I
    P_i = P ./ sum(P, dims = 1)
    P_j = P ./ sum(P', dims = 1)
    P = 0.5*(P_i+P_j)

    opt_ind = Integer[]
    for d in 1:red_dim
        D2_Y = zeros(n,n)
        Q = zeros(n,n)

        best_div = Inf
        best_ind = 0
        for i in setdiff(1:p, opt_ind)
            cur_ind = vcat(opt_ind, i)
            D2_Y = data2dist(Y[:,cur_ind]).^2
            Q = inv.(1.0 .+ D2_Y) - I
            Q = Q ./ sum(Q, dims = 1)
            cur_div = div_measure(P, Q)

            if  cur_div <= best_div
                best_ind = i
                best_div = cur_div
            end
        end
        push!(opt_ind, best_ind)
    end

    return opt_ind
end

#Faster, but approximate, version of joint_div by adding components in batches
function batch_joint_div(X::AbstractMatrix{<:Real}, Y::AbstractVecOrMat{<:Real}, red_dim::Integer = 2, perp::Real = 15.0;
    exact::Bool = true, batch_size::Integer = ceil(Integer, size(Y)[2] / red_dim), div_measure::Function = kl_div)
    n, p = size(Y)
    @assert(red_dim <= p)

    D2_X = data2dist(X).^2
    σ2 = opt_beta(D2_X, perp)[2]

    P = exp.(-0.5 * D2_X ./ σ2) - I
    P_i = P ./ sum(P, dims = 1)
    P_j = P ./ sum(P', dims = 1)
    P = 0.5*(P_i+P_j)

    opt_ind = Integer[]
    for d in 1:ceil(Integer, red_dim / batch_size)
        D2_Y = zeros(n,n)
        Q = zeros(n,n)

        best_div = fill(Inf, batch_size)
        best_ind = zeros(Integer, batch_size)
        marginal_div
        for i in setdiff(1:p, opt_ind)
            cur_ind = vcat(opt_ind, i)
            D2_Y = data2dist(Y[:,cur_ind]).^2
            Q = inv.(1.0 .+ D2_Y) - I
            Q = Q ./ sum(Q, dims = 1)
            cur_div = div_measure(P, Q)

            max_ind = findmax(best_div)[2]
            if  cur_div <= best_div[max_ind]
                best_ind[max_ind] = i
                best_div[max_ind] = cur_div
            end
        end
        append!(opt_ind, best_ind)
    end


    if exact && (length(opt_ind) > red_dim)
        opt_ind = opt_ind[joint_div(X, Y[:,opt_ind], red_dim, perp, div_measure = div_measure)]
    end

    return opt_ind
end

#Auxilary functions for tsne selection
function tsne_loss(X::AbstractMatrix{<:Real}, Y::AbstractVecOrMat{<:Real}, perp::Real = 15.0)

    D2_X = data2dist(X).^2
    σ2 = opt_beta(D2_X, perp)[2]

    P = exp.(-0.5 * D2_X ./ σ2) - I
    P_i = P ./ sum(P, dims = 1)
    P_j = P ./ sum(P', dims = 1)
    P = 0.5*(P_i+P_j)

    D2_Y = data2dist(Y).^2
    Q = inv.(1.0 .+ D2_Y) - I
    Q = Q ./ sum(Q, dims = 1)

    return kl_divergence(P, Q)
end

#Auxilary function for tsne to obtain optimal beta for given perplexity
function opt_beta(D::AbstractMatrix{<:Real}, perp::Real = 15.0; max_tries::Integer = 100, tol::Real = 0.0001)

    #Init
    n1, n2 = size(D)
    P = zeros(n1,n1)
    beta = ones(n1)
    log_perp = log(perp)

    #Run bisection for optimal beta
    for i in 1:n1
        beta_min = -Inf
        beta_max = Inf

        P_cur, H = Hbeta(D[i, setdiff(1:n1, i)], beta[i])
        Hdiff = H - log_perp

        tries = 0
        while abs(Hdiff) > tol && tries <= max_tries
            if Hdiff > 0.0
                beta_min = beta[i]
                if isinf(beta_max)
                    beta[i] *= 2
                else
                    beta[i] = 0.5*(beta[i] + beta_max)
                end
            else
                beta_max = beta[i]
                if isinf(beta_min)
                    beta[i] /= 2
                else
                    beta[i] = 0.5*(beta[i] + beta_min)
                end
            end

            P_cur, H = Hbeta(D[i, setdiff(1:n1, i)], beta[i])
            Hdiff = H - log_perp
            tries += 1
        end

        P[i, setdiff(1:n1, i)] = P_cur

    end

    return P, beta

end

#Auxilary function for tsne to obtain point perplexities for given precision
function Hbeta(D::AbstractVector{<:Real}, beta::Real)
    P = exp.(-D*beta)
    P_sum = sum(P)
    H = log(P_sum) + beta * sum(D .* P) / P_sum
    P = P / P_sum
    return P, H
end

#Shorthand function for Kullback-leibler Divergence
function kl_div(P::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real})
    return kl_divergence(P,Q)
end

#Shorthand function for symmetric Kullback-leibler Divergence
function sym_kl_div(P::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real})
    return 0.5*(kl_divergence(P,Q), kl_divergence(P,Q))
end

#Shorthand function for generalized Kullback-leibler Divergence
function gen_kl_div(P::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real})
    return gkl_divergence(P,Q)
end

#Shorthand function for Renyi Divergence
function renyi_div(P::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real}, k::Real = 0.5)
    return renyi_divergence(P,Q,k)
end

#Shorthand function for Jensen-Shannon Divergence
function js_div(P::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real})
    return js_divergence(P,Q)
end
