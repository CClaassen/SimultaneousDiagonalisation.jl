#Functions for performing joint matrix decomposition

#R standardizes kurtosis values to prod() = 1
 
#This function performs 'ics': the simultaneous diagonalization of two kernels or scatters
function ics(X::AbstractMatrix{<:Real}, S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real},
    comp_select::Function = retain_all, comp_select_args::Tuple = ())

    n,p = size(X)

    #Perform decomposition
    if size(S1) == size(S2) == (p,p)
        V, λ = custom_svd(S1,S2)
        Z = X*V
    elseif size(S1) == size(S2) == (n,n)
        U, λ = reduced_svd(S1,S2)
        λ = sqrt.(λ) #Scale the eigenvalues to normalize in feature space
        Z = U * Diagonal(λ)
    else
        return @assert(false) #add dimension checks
    end

    retain_ind, perm = component_selection(X, Z, λ, comp_select, comp_select_args)

    return Z[:,retain_ind[perm]], Diagonal(λ[retain_ind[perm]])
end

function ics(X::AbstractMatrix{<:Real}, S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real}, loc::AbstractVecOrMat{<:Real} = zeros(size(X)[2]),
    comp_select::Function = retain_all, comp_select_args::Tuple = ())
    loc = vec(loc)
    @assert length(loc) == size(X)[2]

    return ics(X .- loc', S1, S2, comp_select, comp_select_args)
end

#This function performs 'gpca': the diagonalization of a single kernel or scatter
#TODO: maybe add seperate gpca for using svd on X directly
function gpca(X::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real} = cov(X), comp_select::Function = retain_all, comp_select_args::Tuple = ())
    n,p = size(X)

    #Perform decomposition
    if size(S) == (p,p)
        V, λ = EIG(S)
        Z = X*V
    elseif size(S) == (n,n)
        U, λ = SVD(S)
        λ = sqrt.(λ) #Scale the eigenvalues to normalize in feature space
        Z = U * Diagonal(λ)
    end

    retain_ind, perm = component_selection(X, Z, λ, comp_select, comp_select_args)

    return Z[:,retain_ind[perm]], Diagonal(λ[retain_ind[perm]])
end

function gpca(X::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real} = cov(X), loc::AbstractVecOrMat{<:Real} = zeros(size(X)[2]),
    comp_select::Function = retain_all, comp_select_args::Tuple = ())
    loc = vec(loc)
    @assert length(loc) == size(X)[2]

    return gpca(X .- loc', S, comp_select, comp_select_args)
end

#Standard generalized eigen decomposition
function REG_EIG(S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real})
    n1, p1 = size(S1)
    n2, p2 = size(S2)
    @assert n1 == p1 == n2 == p2

    #Obtain eigenvalues and eigenvectors
    λ, V = eigen(S1, S2)

    #Fix the signs of the eigenvectors
    fix_signs!(V)

    #Sort eigenvalues and eigenvectors
    perm = sortperm(λ, rev = true)
    λ, V = λ[perm], V[:,perm]

    return V, λ
end

#Symmetric version of generalized eigen decomposition
function SYM_EIG(S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real})
    @assert size(S1) == size(S2)

    return REG_EIG(Symmetric(S1), Symmetric(S2))
end

#Standard generalized singular value decomposition
function REG_SVD(S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real})
    @assert size(S1) == size(S2)

    #Obtain the (generalized) singular values and vectors
    F = svd(S1, S2)
    U, V = F.U, F.V
    k, l = F.k, F.l
    D1, D2 = F.a, F.b
    D = Diagonal(D1 ./ D2)

    #Perform eigen decomposition on the result
    L, P = eigen(V*D*U')
    ord = sortperm(L, rev = true)
    L, P = Diagonal(L[ord]), P[:,ord]

    #Retrieve and apply the 'stretch' for for orthonormal matrices
    T = dot.(RowVecs(P'), ColVecs(S2*P))
    P *= Diagonal(inv.(sqrt.(T)))

    return fix_signs!(P), diag(L)
end

#'Symmetric' version of generalized svd, uses svd to avoid numerical issues
function SYM_SVD(S1::AbstractMatrix{<:Real}, S2::AbstractMatrix{<:Real})
    @assert size(S1) == size(S2)

    return REG_SVD(Symmetric(S1), Symmetric(S2))
end

#Standard eigen decomposition
function EIG(S::AbstractMatrix{<:Real})
    #Obtain eigenvalues and eigenvectors
    λ, V = eigen(S)

    #Fix the signs of the eigenvectors
    fix_signs!(V)

    #Sort eigenvalues and eigenvectors
    perm = sortperm(λ, rev = true)
    λ, V = λ[perm], V[:,perm]

    return V, λ
end

#Standard singular value decomposition
function SVD(S::AbstractMatrix{<:Real})
    #Obtain eigenvalues and eigenvectors
    U, κ, V = svd(S)

    #Fix the signs of the eigenvectors
    fix_signs!(V)

    return V, κ
end


function custom_eig(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    D_B, V_B = eigen(B)
    T = V_B/Diagonal(sqrt.(D_B))
    D_C, V_C = eigen(T'A*T)
    V = T*V_C
    fix_signs!(V)
    ord = sortperm(D_C, rev = true)
    return V[:, ord], D_C[ord]
end

function custom_svd(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    _, S_B, V_B = svd(B)
    T = V_B/Diagonal(sqrt.(S_B))
    _, S_C, V_C = svd(T'A*T)
    V = T*V_C
    fix_signs!(V)

    return V, S_C
end

function reduced_svd(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    r = rank(B)
    _, S_B, V_B = svd(B)
    T = V_B[:,1:r] / Diagonal(sqrt.(S_B[1:r]))
    _, S_C, V_C = svd(T'A*T)
    V = T*V_C
    fix_signs!(V)

    return V, S_C
end

function custom_gsvd(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    F = svd(A, B)
    U, V = F.U, F.V
    k, l = F.k, F.l
    #r = size(F.R)[1]
    ind = k+1:1:k+l
    D1, D2 = F.a[ind], F.b[ind]
    D = Diagonal(D1 ./ D2)

    L, P = eigen(V[:,ind]*D*U[:,ind]')
    ord = sortperm(L, rev = true)
    L, P = Diagonal(L[ord]), P[:,ord]

    T = dot.(RowVecs(P'), ColVecs(B*P))
    #T = dot.(eachrow(P), eachcol(L/P))
    P *= Diagonal(inv.(sqrt.(T)))

    return fix_signs!(P), diag(L)
end



function constrained_svd(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    #n1, p1 = size(A)
    #n2, p2 = size(B)

    r_A, r_B = rank(A), rank(B)

    _, S_B, V_B = svd(B)
    T = V_B[:,1:r_B]/Diagonal(sqrt.(S_B[1:r_B]))
    _, S_C, V_C = svd(T'A*T)
    V = T*V_C
    fix_signs!(V)

    return V, S_C
end

function reduce_mat_svd(X::AbstractMatrix{<:Real}, red_dims::Tuple{Bool, Bool} = (false, true); r::Integer = rank(X))
    n,p = size(X)

    u, s, v = svd(X)
    X2 = u[:,1:r]*Diagonal(s[1:r])*v[:,1:r]'

    rng1 = red_dims[1] ? (1:r) : (1:n)
    rng2 = red_dims[2] ? (1:r) : (1:p)

    return X2[rng1,rng2]
end

function reduce_mat_qr(X::AbstractMatrix{<:Real}, red_dims::Tuple{Bool, Bool} = (false, true); r::Integer = rank(X))
    n,p = size(X)

    Q, R, p = qr(X, ColumnNorm())
    X2 = Q*R

    rng1 = red_dims[1] ? (1:r) : (1:n)
    rng2 = red_dims[2] ? (1:r) : (1:p)

    return X2[rng1,rng2]
end
