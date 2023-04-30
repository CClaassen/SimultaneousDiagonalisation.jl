#Various function for loading the data used in the thesis

##Set the correct directory
cd(@__DIR__)

using DelimitedFiles, Embeddings, MLDatasets, OutlierDetectionData

#Load the Iris dataset from MLDatasets
function iris_data()
    iris = Iris(as_df = false)
    X, y = permutedims(iris.features), String.(vec(iris.targets))
    y = replace.(y, "Iris-setosa" => "Setosa", "Iris-versicolor" => "Versicolor", "Iris-virginica" => "Virginica")
    z = vec_cat_encode(y)

    return X, y, z
end

#Loads Word2Vec embedding data from a file
function word2vec_data()
    #Read the data we want to use:
    W = readdlm("thesis_words.csv", ',', String)
    labels, words = W[:,1], W[:,2]
    W_emb = readdlm("word2vec_words_embedded.csv", ',', Float64)

    ord = sortperm(labels)

    return W_emb[ord,:], labels[ord], vec_cat_encode(labels[ord])
end

#Loads GloVe embedding data from a file
function glove_data()
    #Read the data we want to use:
    W = readdlm("thesis_words.csv", ',', String)
    labels, words = W[:,1], W[:,2]
    W_emb = readdlm("glove_words_embedded.csv", ',', Float64)

    ord = sortperm(labels)

    return W_emb[ord,:], labels[ord], vec_cat_encode(labels[ord])
end

#Loads FastText embedding data from a file
function fasttext_data()
    #Read the data we want to use:
    W = readdlm("thesis_words.csv", ',', String)
    labels, words = W[:,1], W[:,2]
    W_emb = readdlm("fasttext_words_embedded.csv", ',', Float64)

    ord = sortperm(labels)

    return W_emb[ord,:], labels[ord], vec_cat_encode(labels[ord])
end

function ODDS_data(str::String; replace_0::Pair{String, String} = "normal" => "Normal", replace_1::Pair{String, String} = "outlier" => "Outlier")
    str = lowercase(str)
    if str in ODDS.list()
        X, y = ODDS.load(str)
        X, y = Matrix(X), String.(y)
        ord = sortperm(y)
        X, y = X[ord,:], y[ord]
        y = replace.(y, replace_0,replace_1)
        z = vec_cat_encode(y) .- 1

        return X, y, z
    else
        throw(ArgumentError("Invalid dataset identifier string."))
    end
end

#Shorthands for datasets from ODDS via the above function:
breastw_data() = ODDS_data("breastw")       #683x9  ( 444,239)
glass_data() = ODDS_data("glass")           #214x9  ( 205,  9)
thyroid_data() = ODDS_data("thyroid")       #3772x6 (3772, 93)
vertebral_data() = ODDS_data("vertebral")   #240x6  ( 210, 30)
wbc_data() = ODDS_data("wbc",
replace_0 = "normal" => "Benign",
replace_1 = "outlier" => "Malignant")       #378x30 ( 357, 21)
wine_data() = ODDS_data("wine")             #129x13 ( 119, 30)

ionosphere_data() = ODDS_data("ionosphere") #351x33 ( 225,126)
lympho_data() = ODDS_data("lympho")         #148x18 ( 142,  6)


function mnist_data(n_samples::Integer = 1000, split::Symbol = :train, labels_sorted::Bool = false; seed::Integer = -1)
    seed = seed < 0 ? abs(rand(Int32)) : seed
    rng = Random.seed!(seed)

    dat = MNIST(split=split)
    X, y = dat.features, dat.targets
    n = length(y)

    ind = sample(rng, 1:n, n_samples, replace = false)
    if labels_sorted
        ord = sortperm(y[ind])
        ind = ind[ord]
    end

    return flatten1d(X, ind), y[ind], vec_cat_encode(y[ind])
end

function flatten2d(X::Array{Float32, 3}, ind::Vector{<:Integer})
    return reduce(vcat, [X[:,:,i]' for i in ind])
end

function flatten1d(X::Array{Float32, 3}, ind::Vector{<:Integer})
    return reduce(vcat, [vec(X[:,:,i]')' for i in ind])
end

function mnist_mean(X::Matrix{Float32}, label::Vector{<:Integer})
    res = zeros(Float32, 28, 28, 10)
    res_n = zeros(Integer, 10)
    n = length(label)

    for i in 1:n
        cur_ind = label[i]+1
        res_n[cur_ind] += 1
        ind1, ind2 = 28*i-27, 28*i
        #res[:, :, cur_ind] += X[ind1:ind2, :] # in case of 2d
        res[:, :, cur_ind] += reshape(X[i,:], (28,28))
    end

    for j in 1:10
        res[:,:,j] /= maximum([1,res_n[j]])
    end

    #display(sum(res_n))
    return reduce(hcat, [res[:,:,k] for k in 1:10])
end


#Loads the Word2Vec embeddings from scratch
function word2vec_data_scratch()
    #Read the data we want to use:
    W = readdlm("thesis_words.csv", ',', String)
    labels, words = W[:,1], W[:,2]

    #Load relevant Word2Vec data from Embeddings.jl example:
    embtable = load_embeddings(Word2Vec, keep_words = Set(words)) #Can take a while on first use
    word_indices = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

    #Saves/refreshes the obtained embeddings to a file:
    writedlm("word2vec_words_embedded.csv", get_embeddings(words, embtable, word_indices), ',')

    return get_embeddings(words, embtable, word_indices), labels, vec_cat_encode(labels)
end

#Loads the GloVe embeddings from scratch
function glove_data_scratch()
    #Read the data we want to use:
    W = readdlm("thesis_words.csv", ',', String)
    labels, words = W[:,1], W[:,2]

    #embtable = load_embeddings(Fast_Text{:en}, 1, keep_words = Set(words))

    #Load relevant GloVe data from Embeddings.jl example:
    embtable = load_embeddings(GloVe{:en}, 6, keep_words = Set(words)) #Can take a while on first use
    word_indices = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

    #Saves/refreshes the obtained embeddings to a file:
    writedlm("glove_words_embedded.csv", get_embeddings(words, embtable, word_indices), ',')

    return get_embeddings(words, embtable, word_indices), labels, vec_cat_encode(labels)
end

#Loads the FastText embeddings from scratch
function fasttext_data_scratch()
    #Read the data we want to use:
    W = readdlm("thesis_words.csv", ',', String)
    labels, words = W[:,1], W[:,2]

    #Load relevant FastText data via Embeddings.jl example:
    embtable = load_embeddings(FastText_Text{:en}, 1, keep_words = Set(words)) #Can take a while on first use
    word_indices = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

    #Saves/refreshes the obtained embeddings to a file:
    writedlm("fasttext_words_embedded.csv", get_embeddings(words, embtable, word_indices), ',')

    return get_embeddings(words, embtable, word_indices), labels, vec_cat_encode(labels)
end


#Auxillary functions for getting word embeddings

#Returns embeddings of a single word
function get_embedding(word::AbstractString, embtable::Embeddings.EmbeddingTable, word_indices::Dict{String, Int64})
    ind = word_indices[word]
    emb = embtable.embeddings[:,ind]
    return emb
end

#Returns embeddings of multiple words
function get_embeddings(words::AbstractArray, embtable::Embeddings.EmbeddingTable, word_indices::Dict{String, Int64}; features::Integer = 300)
    dim = size(words)[1]
    emb = zeros(dim, features)

    for i=1:dim
        index = word_indices[words[i]]
        emb[i,:] = embtable.embeddings[:,index]
    end

    return emb
end
