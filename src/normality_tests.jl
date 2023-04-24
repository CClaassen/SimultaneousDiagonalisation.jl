#Univariate Normality tests [13 variants]
 
#Function to perform a normality test via comparing p-values
function normal_test(x::AbstractVector{<:Real}, test_function::Function = jarque_bera, α::Float64 = 0.05; bool_standard::Bool = true, bool_rob::Bool = false)
    #TODO: maybe add function verification check?
    #@assert test_function in

    if bool_standard*bool_rob
        x = rob_score(x)
    elseif bool_standard
        x = z_score(x)
    else
        @warn("Implemented tests expect standardization")
    end

    return round.(α .< test_function(x, α)[1], digits = 4)
end


#Moment-based tests (Skewness and Kurtosis) / Goodness-of-fit [5 variants]

#1
function jarque_bera(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    n = length(x)
    skew, ex_kurt = standard_moment(x, 3), standard_moment(x, 4) - 3.0
    null_dist = Chisq(2)

    #Calculate test statistic, critical value and p-value
    test_stat = n*(skew^2 + 0.25*(ex_kurt)^2)/6.0
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat)

    return p_val, test_stat, crit_val
end

#2
function bonnett_seier(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    n = length(x)
    null_dist = Normal()

    geary = geary_kurt(x)
    ω = 13.29*(log(inv(geary)))
    z = sqrt(n+2)*(ω-3)/3.54

    #Calculate test statistic, critical value and p-value
    test_stat = z
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat)

    return p_val, test_stat, crit_val
end

#3
function agostino_pearson(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    n = length(x)
    skew = standard_moment(x, 3)
    null_dist = Chisq(1)

    #Retrieve the first four moments of skewness
    μ1_s, μ2_s, γ1_s, γ2_s = skewness_moments(n)

    #Calculate the transformed skewness 'z_1'
    w_2 = sqrt(2.0 * γ2_s + 4.0) - 1.0
    δ = 1.0 / sqrt(log(sqrt(w_2)))
    α_2 = 2.0 / (w_2 - 1.0)
    z_1 = δ * asinh(skew / sqrt(α_2 * μ2_s))

    #Calculate test statistic, critical value and p-value
    test_stat = z_1^2
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat)

    return p_val, test_stat, crit_val
end

#4
function ansecombe_glynn(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    n = length(x)
    ex_kurt = standard_moment(x, 4) - 3.0
    null_dist = Chisq(1)

    #Retrieve the first four moments of skewness and kurtosis
    μ1_k, μ2_k, γ1_k, γ2_k = kurtosis_moments(n)

    #Calculate the transformed kurtosis 'z_2'
    a =  6.0 + (8.0 / γ1_k) * ((2.0 / γ1_k) + sqrt(1.0 + 4.0 / γ1_k^2.0))
    z_2 = sqrt(4.5a) * (1.0 - (2.0 / 9.0a) - ((1.0 - 2.0 / a) / (1.0 + (ex_kurt - μ1_k) / sqrt(μ2_k) * sqrt(2.0 / (a - 4.0))))^(1.0/3.0))

    #Calculate test statistic, critical value and p-value
    test_stat = z_2^2.0
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat)

    return p_val, test_stat, crit_val
end

#5
function omnibus_K2(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    n = length(x)
    skew, ex_kurt = standard_moment(x, 3), standard_moment(x, 4) - 3.0
    null_dist = Chisq(2)

    #Retrieve the first four moments of skewness and kurtosis
    μ1_s, μ2_s, γ1_s, γ2_s = skewness_moments(n)
    μ1_k, μ2_k, γ1_k, γ2_k = kurtosis_moments(n)

    #Calculate the transformed skewness 'z_1'
    w_2 = sqrt(2.0 * γ2_s + 4.0) - 1.0
    δ = 1.0 / sqrt(log(sqrt(w_2)))
    α_2 = 2.0 / (w_2 - 1.0)
    z_1 = δ * asinh(skew / sqrt(α_2 * μ2_s))

    #Calculate the transformed kurtosis 'z_2'
    a =  6.0 + (8.0 / γ1_k) * ((2.0 / γ1_k) + sqrt(1.0 + 4.0 / γ1_k^2.0))
    z_2 = sqrt(4.5a) * (1.0 - (2.0 / 9.0a) - ((1.0 - 2.0 / a) / (1.0 + (ex_kurt - μ1_k) / sqrt(μ2_k) * sqrt(2.0 / (a - 4.0))))^(1.0/3.0))

    #Calculate test statistic, critical value and p-value
    test_stat = z_1^2.0 + z_2^2.0
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat)

    #TODO: adjust critical value for smaller samples (z_1 and z_2 not independent)
    return p_val, test_stat, crit_val
end

#ECDF-based tests [5 variants]
#TODO: add critical values from tables or via simulation for 4 ecdf based tests

#1
function anderson_darling(x::Vector{<:Real}, α::Float64 = 0.05; null_dist::UnivariateDistribution = Normal())
    #Extract info
    n = length(x)

    x_sorted = sort(x)

    s = 0.0
    f_evals = cdf.(null_dist, x_sorted)
    for i in 1:n
        s += (2.0i - 1.0) * (log(f_evals[i]) + log(1.0 - f_evals[n+1-i]))
    end
    n = Float64(n)
    s = s / n

    #Calculate test statistic, critical value and p-value
    test_stat = -n - s

    #Apply small smaple correction
    if null_dist isa Normal{Float64}
        #test_stat = test_stat*(1.0 + 4.0 / n - 25.0 / n^2.0) #[Shorack & Wellner (1986, p239)]
        test_stat = test_stat*(1.0 + 0.75 / n - 2.25 / n^2.0) #[D'Agostino (1986)]
    else
        @warn "Given null distribution is non-Normal: be careful of small samples."
    end

    crit_val = quantile(null_dist, 1.0 - α /2)
    p_val = 2.0 * minimum((cdf(null_dist, test_stat), 1.0 - cdf(null_dist, test_stat)))
    #p_val = 1.0 - cdf(null_dist, test_stat)

    return p_val, test_stat, crit_val
end

#2
function kolmogorov_smirnov(x::Vector{<:Real}, α::Float64 = 0.05; null_dist::UnivariateDistribution = Normal())
    #Extract info
    x_ecdf = ecdf(x)
    z_emp = x_ecdf.(x)
    z_null = cdf.(null_dist, x)

    #Calculate test statistic, critical value and p-value
    test_stat = maximum(z_emp - z_null)
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat) #TODO: Obtain p-values of Kolmogorov distribution, see note above

     return p_val, test_stat, crit_val
end

#3
function lilliefors(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    x_ecdf = ecdf(x)
    z_emp = x_ecdf.(x)
    z_null = cdf.(null_dist, x)

    #Calculate test statistic, critical value and p-value
    test_stat = maximum(z_emp - z_null)
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat) #TODO: Obtain p-values of Lilliefors distribution, see note above

     return p_val, test_stat, crit_val
end

#4
function cramer_mises(x::Vector{<:Real}, α::Float64 = 0.05; null_dist::UnivariateDistribution = Normal())
    #Extract info
    n = length(x)
    z = cdf.(null_dist, sort(x))

    t = 1.0 / 12.0n
    for i in 1:n
        t += ((2i -1) / (2n) - z[i])^2.0
    end

    #Calculate test statistic, critical value and p-value
    test_stat = t
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat) #TODO: Obtain p-values of tabulated value, see note above

     return p_val, test_stat, crit_val
end

#5
function watson(x::Vector{<:Real}, α::Float64 = 0.05; null_dist::UnivariateDistribution = Normal())
    #Extract info
    n = length(x)
    z = cdf.(null_dist, sort(x))

    t = 1.0 / 12.0n
    for i in 1:n
        t += ((2i -1) / (2n) - z[i])^2.0
    end
    u_2 = t - n*(raw_moment(z, 1)-0.5)^2.0

    #Calculate test statistic, critical value and p-value
    test_stat = u_2
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat) #TODO: Obtain p-values of tabulated value, see note above

     return p_val, test_stat, crit_val
end

#Other tests [3 variants]

#1
function shapiro_wilk(x::Vector{<:Real}, α::Float64 = 0.05; null_dist::UnivariateDistribution = Normal())
    #Extract info
    n = length(x)

    x_test = hcat(sort.(ColVecs(randn(n,n)))...)
    mu, S = mean(x_test, dims = 2), cov(x_test)
    a = mu'inv(S) / norm(mu'inv(S))
    W = dot(a, sort(x))^2 / ((n-1) * var(x))

    #Calculate test statistic, critical value and p-value
    test_stat = W #0 < W < 1
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat) #TODO: Maybe obtain p-values via simulation

    return p_val, test_stat, crit_val
end

#2
function shapiro_francia(x::Vector{<:Real}, α::Float64 = 0.05)
    #Extract info
    n = length(x)

    x_test = hcat(sort.(ColVecs(randn(n,n)))...)
    mu, S = mean(x_test, dims = 2), cov(x_test)
    W = cor(mu, sort(x))[1]

    #Calculate test statistic, critical value and p-value
    test_stat = log(1 - W)
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat) #TODO: Result asymptotically valid, maybe obtain p-values via simulation

    return p_val, test_stat, crit_val
end

#3
function pearson_chi2(x::Vector{<:Real}, α::Float64 = 0.05; bounds = [-Inf, -1, -0.5, 0, 0.5, 1, Inf])
    #Extract info
    n = length(x)
    bounds = sort(bounds)
    null_dist = Chisq(length(bounds) - 1 - 3)

    actual_counts = count_sample_regions(x, bounds)
    cdf_prop = cdf.(Normal(), bounds)
    theoretical_counts = round_retain_sum(n*(cdf_prop[2:end] - cdf_prop[1:end-1]))
    chi2 = sum((actual_counts - theoretical_counts) .^2 ./ theoretical_counts)

    #Calculate test statistic, critical value and p-value
    test_stat = chi2
    crit_val = quantile(null_dist, 1.0 - α)
    p_val = 1.0 - cdf(null_dist, test_stat)

    return p_val, test_stat, crit_val
end

#Basic standardization function
function z_score(x::AbstractArray)
    μ, σ = mean(x), std(x)
    return (x .- μ) / σ
end

#Robust standardization function
function rob_score(x::AbstractArray)
    μ, σ = median(x), mad(x, normalize = true)
    return (x .- μ) / σ
end

#Alias for default MAD
function mad_n(x::AbstractArray)
    return mad(x, normalize = true)
end

#Auxilary function for moment based tests
function skewness_moments(n::Integer)
    n = Float64(n)

    #Exact finite mean, variance, skewness and kurtosis of 'skew' if sample is indeed normal
    μ1_s = 0.0
    μ2_s = (6.0 * (n - 2.0)) / ((n + 1.0) * (n + 3.0))
    γ1_s = 0.0
    γ2_s = (36.0 * (n - 7.0) * (n^2.0 + 2.0n - 5.0)) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))

    return μ1_s, μ2_s, γ1_s, γ2_s
end

#Auxilary function for moment based tests
function kurtosis_moments(n::Integer)
    n = Float64(n)

    #Exact finite mean, variance, skewness and kurtosis of 'kurt' if sample is indeed normal
    μ1_k = -6.0 / (n + 1.0)
    μ2_k = (24.0n * (n - 2.0) * (n - 3.0)) / ((n + 1.0)^2.0 * (n + 3.0) * (n + 5.0))
    γ1_k = ((6.0 * (n^2.0 - 5.0n + 2.0)) / ((n + 7.0) * (n + 9.0))) * sqrt((6.0 * (n + 3.0) * (n + 5.0)) / (n * (n - 2.0) * (n - 3.0)))
    γ2_k = (36.0 * (15.0n^6.0 - 36.0n^5.0 - 628.0n^4 + 982.0n^3 + 5777.0n^2.0 - 6402.0n + 900.0)) / (n * (n - 3.0) * (n - 2.0) * (n + 7.0) * (n + 9.0) * (n + 11.0) * (n + 13.0))

    return μ1_k, μ2_k, γ1_k, γ2_k
end

#Auxilary function for pearson χ2 test
function round_retain_sum(x::AbstractArray)
    n = length(x)

    x_r = Integer.(round.(x))
    x_f = x - x_r
    ind = sortperm(x_f)
    sum_rem = Integer(round(sum(x)) - sum(x_r))

    if sum_rem > 0
        x_r[ind[1:sum_rem]] .+= 1
    elseif sum_rem < 0
        ind = reverse(ind)
        x_r[ind[1:sum_rem]] .-= 1
    end

    return x_r
end

#Auxilary function for pearson χ2 test
function count_sample_regions(x::AbstractArray, bounds::AbstractArray; sorted::Bool = false)
    n_b = length(bounds)
    res = zeros(Integer, n_b)
    bounds = sorted ? bounds : sort(bounds)

    for i in 1:n_b
        res[i] = sum(x .<= bounds[i])
    end

    return res[2:end] - res[1:end-1]
end

#Auxilary function for bonnett-seier test
function geary_kurt(x::AbstractArray{<:Real})
    #Note: under normality this should return about inv(sqrt(pi/2)) = 0.7979
    return mean(abs.(x .- mean(x))) / std(x)
end
