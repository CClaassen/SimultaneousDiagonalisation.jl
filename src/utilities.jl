#Various minor auxiliary functions for other files
 
#Shorthand for collecting a dense identity matrix
function eye(p::Integer)
    return Matrix(1.0I, p, p)
end

#Shorthand for collecting a diagonal identity matrix
function diag_eye(p::Integer)
    return Diagonal(1.0I, p)
end

#Shorthand for collecting a dense 'missing' identity matrix
function eye_nan(p::Integer)
    return fill(NaN, p, p)
end

#Shorthand for collecting a diagonal 'missing' identity matrix
function diag_eye_nan(p::Integer)
    return Diagonal(fill(NaN, p, p))
end

#Shorthand for the standard inner product
function self_inner(X::Union{Vector{<:Real}, Matrix{<:Real}})
    return X'X
end

#Shorthand for the standard outer product
function self_outer(X::Union{Vector{<:Real}, Matrix{<:Real}})
    return X*X'
end

#Shorthand for symmetrizing a square matrix
function symmetrize(X::Matrix{<:Real})
    return 0.5X+0.5X'
end

function text_subscript(str::String, i::Integer)
    return str * prod(x -> Char(0x2080 + x), reverse(digits(i)))
end

ts(str,i) = text_subscript(str, i)

duplicates(v) = [k for (k, v) in countmap(v) if v > 1]

function get_seed(genr::Xoshiro)
       return (genr.s0, genr.s1, genr.s2, genr.s3)
end

function next_seed(genr::Xoshiro)
    rand(genr, 1)
    return get_seed(genr)
end
