#This file contains function to obtain all figures and tables from the thesis

#Function to run all experiments from the thesis
function all_experiments()
    iris_res1, iris_res2 = iris_experiments()
    wine_experiments()
    wbc_res1, wbc_res2 = wbc_experiments()
    word2vec_res = word2vec_experiments()
    glove_res = glove_experiments()
    fasttext_res = fasttext_experiments()
    code_experiment()
    weighting_kernel_graph()

    return (iris_res1, iris_res2), (wbc_res1, wbc_res2), (word2vec_res, glove_res, fasttext_res)
end

function iris_experiments()
    X, labels, classes = iris_data()
    n,p = size(X)

    Z1,z1 = gpca(X, cov(X), mean1(X))
    Z2,z2 = ics(X, cov(X), cov4(X), mean3(X))

    prop_var = 100*round.(z1 / sum(z1), digits = 4)

    plt1 = scatter(Z1[:,1], Z1[:,2], group = labels, xlabel = "Principal Component 1 ($(prop_var[1,1])% of variance)", ylabel = "Principal Component 2 ($(prop_var[2,2])% of variance)", legend = :bottomright, title = "Iris Data Visualisation by PCA")
    plt2 = scatter(Z2[:,1], Z2[:,2], group = labels, xlabel = "Component 1", ylabel = "Component 2",legend = :bottomright, title = "Iris Data Visualisation by ICS")

    plt3 = scatter(ColVecs(tsne2(X, 2, 15))..., group = labels, xlabel = "Dimension 1", ylabel = "Dimension 2", legend = :topright, title = "Iris Data Visualisation by t-SNE")
    Random.seed!(987)
    plt4 = scatter(ColVecs(umap2(X, 2, 25))..., group = labels, xlabel = "Dimension 1", ylabel = "Dimension 2", legend = :topleft, title = "Iris Data Visualisation by UMAP")

    plt5 = contour_plot(Z1, labels, (1,2), contour_func = svm2, title = "Linear SVM on [PC1,PC2] of Iris Data", hide_labels = false)
    plt6 = contour_plot(Z1, labels, (1,2), contour_func = logit2, title = "Logistic Classifier on [PC1,PC2] of Iris Data", hide_labels = false)

    rng = 1:145
    CM_svm = stratified_k_fold(transform_z(Z1[rng,:]), classes[rng], 5, svm2, seed = 9)[2]
    CM_logit = stratified_k_fold(transform_z(Z1[rng,:]), classes[rng], 5, logit2, seed = 9)[2]


    plt7 = heatmap_plot(CM_svm,labels[rng], title = "Iris Data: 5-fold Out-of-Sample Results (Linear SVM)")
    plt8 = heatmap_plot(CM_logit,labels[rng], title = "Iris Data: 5-fold Out-of-Sample Results (Logistic)")

    #Save/refresh all figures
    savefig(plt1, "Figures/iris_1_plot.svg")
    savefig(plt2, "Figures/iris_2_plot.svg")
    savefig(plt3, "Figures/iris_3_plot.svg")
    savefig(plt4, "Figures/iris_4_plot.svg")
    savefig(plt5, "Figures/iris_5_plot.svg")
    savefig(plt6, "Figures/iris_6_plot.svg")
    savefig(plt7, "Figures/iris_7_plot.svg")
    savefig(plt8, "Figures/iris_8_plot.svg")

    res0 = diagnostics(CM_svm, method = :none)
    res1 = diagnostics(CM_svm, method = :macro)
    res2 = diagnostics(CM_svm, method = :weighted)
    res3 = diagnostics(CM_svm, method = :micro)


    coll = [res1,res2,res3]
    crit = [:TP,:TN,:FP,:FN, :T, :ACC, :F_1,:MCC]
    res4 = round.(get_all_eval(res0,crit), digits = 4)
    res5 = round.(get_all_eval(coll,crit), digits = 4)

    display.([plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8])
    return res4, res5
end

function wine_experiments()
    #Linearily seperable on entire data with supervision
    X, labels, classes = wine_data()

    n,p = size(X)
    contamination = 100*round(10/(10+119), digits = 4) #7.75 % outliers

    mu1 = mean1(X)
    mu2 = mean3(X)

    S0,mu0 = fastMCD2(X, 0.25, seed = 7)
    S1,S2 = cov(X), cov4(X)
    S3,mu3 = fastMCD2(X, 0.50, seed = 5)
    S4,mu4 = fastMCD2(X, 0.9, seed = 6)

    S5,S6 = alcov(X, gaussian_kernel_w, (1, 1.0), scale = 0.05), alcov(X, gaussian_kernel_w, (116, 1.0), scale = 1.0)
    S7,S8 = alcov(X, epanechnikov_kernel_w, (1, 1.0), scale = 0.05), alcov(X, epanechnikov_kernel_w, (123, 1.0), scale = 1.0)



    Z1,z1 = gpca(X,S1, mu1)
    Z2,z2 = ics(X,S2,S1, mu1)
    Z3,z3 = ics(X,S1,S3, mu1)
    Z4,z4 = ics(X,S4,S0, mu4)

    Z5,z5 = ics(X,S5,S2, mu1)
    Z6,z6 = ics(X,S6,S2, mu1)
    Z7,z7 = ics(X,S7,S2, mu1)
    Z8,z8 = ics(X,S8,S2, mu1)

    z1[1,1] = round(z1[1,1])
    z1[2,2] = round(z1[2,2], digits = 1)

    ind = (1,2,p)
    plt1 = contour_plot_ind(Z1,Z2,labels,labels,ind,z1,z2, title = ("{$(ts("Cov",2)),I}","{$(ts("Cov",4)),$(ts("Cov",2))}"))
    plt2 = contour_plot_ind(Z3,Z4,labels,labels,ind,z3,z4, title = ("{$(ts("Cov",2)),$(ts("MCD",50))}","{$(ts("MCD",90)),$(ts("MCD",25))}"))
    plt3 = contour_plot_ind(Z5,Z6,labels,labels,ind,z5,z6, title = ("{LCov,$(ts("Cov",4))}","{$(ts("ALCov",90)),$(ts("Cov",4))}"))
    plt4 = contour_plot_ind(Z7,Z8,labels,labels,ind,z7,z8, title = ("{LCov,$(ts("Cov",4))}","{$(ts("ALCov",95)),$(ts("Cov",4))}"))

    plt5 = contour_plot(Z1[:,1:2], labels, (1,2), contour_func = logit2, hide_labels = false, title = "Logistic Classifier on Result of {$(ts("Cov",2)),I} Pair")
    plt6 = contour_plot(Z3[:,1:2], labels, (1,2), contour_func = logit2, hide_labels = false, title = "   Logistic Classifier on Result of {$(ts("Cov",2)),$(ts("MCD",50))} Pair")
    plt7 = contour_plot(Z8[:,1:2], labels, (1,2), contour_func = logit2, hide_labels = false, title = "      Logistic Classifier on Result of {$(ts("ALCov",95)),$(ts("Cov",4))} Pair")
    plt8 = contour_plot(gpca(Z8[:,1:2])[1], labels, (1,2), contour_func = logit2, hide_labels = false, title = "      Logistic Classifier on Result of {$(ts("ALCov",95)),$(ts("Cov",4))} Pair")
    #plt10 = component_plot(gpca(Z10[:,1:2])[1], 2, labels, mix = true)

    plt0 = scatter(ColVecs(tsne2(X, 2, 30))..., group = labels, xlabel = "Dimension 1", ylabel = "Dimension 2", legend = :bottomright, title = "Wine Data Visualisation by t-SNE \n (Contamination: $contamination%)")
    plt9 = scatter(ColVecs(tsne2(Z8[:,[1,2,p-1,p]], 2, 15))..., group = labels, xlabel = "Dimension 1", ylabel = "Dimension 2", legend = :bottomright, title = "Wine Data Visualisation by t-SNE \n (On 4 Components of {$(ts("ALCov",95)),$(ts("Cov",4))} Pair)")

    #K1, K2, K3 = gaussian_kernel(X1, median_trick(X)), linear_kernel(X1, median_trick(X)), rational_kernel(X1, median_trick(X));
    #Z = ics(X,K3,K2*K3)[1]

    #Save/refresh all figures
    savefig(plt1, "Figures/wine_1_plot.svg")
    savefig(plt2, "Figures/wine_2_plot.svg")
    savefig(plt3, "Figures/wine_3_plot.svg")
    savefig(plt4, "Figures/wine_4_plot.svg")
    savefig(plt5, "Figures/wine_5_plot.svg")
    savefig(plt6, "Figures/wine_6_plot.svg")
    savefig(plt7, "Figures/wine_7_plot.svg")
    savefig(plt8, "Figures/wine_8_plot.svg")
    savefig(plt9, "Figures/wine_9_plot.svg")
    savefig(plt0, "Figures/wine_0_plot.svg")

    display.([plt0, plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9])

    return
end

function wbc_experiments()
    X, labels, classes = wbc_data()

    n,p = size(X)
    contamination = 100*round(21/(21+357), digits = 4) #5.56 % outliers

    mu1 = mean1(X)
    mu2 = mean3(X)

    S1,S2 = cov(X), cov4(X)
    S4,mu4 = fastMCD2(X, 0.90, seed = 40)
    S5,mu5 = fastMCD2(X, 0.95, seed = 60)
    S6,mu6 = fastMCD2(X, 0.50, seed = 50)

    S7,S8 = alcov(X, gaussian_kernel_w, (1, 1.0), scale = 0.1), alcov(X, gaussian_kernel_w, (38, 1.0), scale = 1.0)
    S9,S10 = alcov(X, epanechnikov_kernel_w, (1, 1.0), scale = 0.1), alcov(X, epanechnikov_kernel_w, (38, 1.0), scale = 1.0)

    S11,S12 = alcov(X, gaussian_kernel_w, (340, 1.0), scale = 1.0), alcov(X, epanechnikov_kernel_w, (340, 1.0), scale = 1.0)


    Z1,z1 = gpca(X,S1, mu1)
    Z2,z2 = gpca(X,S4, mu4)
    Z3,z3 = ics(X,S2,S1, mu1)
    Z4,z4 = ics(X,S1,S4, mu1)
    Z5,z5 = ics(X,S1,S6, mu1)
    Z6,z6 = ics(X,S6,S4, mu4)
    Z7,z7 = ics(X,S7,S1, mu1)
    Z8,z8 = ics(X,S8,S1, mu1)
    Z9,z9 = ics(X,S9,S1, mu1)
    Z10,z10 = ics(X,S10,S1, mu1)
    Z11,z11 = ics(X,S11,S1, mu1)
    Z12,z12 = ics(X,S12,S1, mu1)

    Random.seed!(2804)
    plt0 = scatter(ColVecs(umap2(X, 2, 20))..., group = labels, xlabel = "Dimension 1", ylabel = "Dimension 2", legend = :bottomleft, title = "WBC data visualisation by UMAP \n (Contamination: $contamination%)")

    #plt1 = contour_plot(Z1[:,1:2], labels, (1,2), contour_func = logit2, hide_labels = false, title = "Logistic Classifier on Result of {$(ts("Cov",2)),I} Pair")
    plt2 = contour_plot(gpca(Z10[:,1:2])[1], labels, (1,2), contour_func = logit2, hide_labels = false, title = "Logistic Classifier on Result of {$(ts("E_ALCov",5)),$(ts("Cov",2))} Pair")


    display.([plt0, plt2])

    ind1 = (1,2,3,p-1,p)
    res1 = [stratified_k_fold(transform_z(i[:,collect(ind1)]), labels, seed  = 1234)[2] for  i in [Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12]]
    res2 = diagnostics.(res1, method = :binary, pos_class =2)
    res3 = 100*round.(get_all_eval(res2, [:ACC, :PPV, :TPR, :F_1,:MCC]), digits = 4)

    ind2 = 1:20
    res4 = [stratified_k_fold(transform_z(i[:,collect(ind2)]), labels, seed  = 1234)[2] for  i in [Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12]]
    res5 = diagnostics.(res4, method = :binary, pos_class =2)
    res6 = 100*round.(get_all_eval(res5, [:ACC, :PPV, :TPR, :F_1,:MCC]), digits = 4)


    plt3 = heatmap_plot(res1[10],labels, title = "Linear SVM: Stratified 10-fold \n {$(ts("E_ALCov",10)),$(ts("Cov",2))} \n 5 Components")
    plt4 = heatmap_plot(res4[10],labels, title = "Linear SVM: Stratified 10-fold \n {$(ts("E_ALCov",10)),$(ts("Cov",2))} \n 20 Components")
    plt5 = heatmap_plot(stratified_k_fold(transform_z(X), labels, seed  = 1234)[2], labels, title = "Linear SVM: Stratified 10-fold \n Original Data \n 30 Components")

    display.([plt3, plt4, plt5])

    savefig(plt0, "Figures/wbc_umap.svg")
    savefig(plt2, "Figures/wbc_2d.svg")
    savefig(plt3, "Figures/wbc_cm1.svg")
    savefig(plt4, "Figures/wbc_cm2.svg")
    savefig(plt5, "Figures/wbc_cm3.svg")
    return res3, res6
end

function word2vec_experiments()
    X1, labels, classes = word2vec_data()
    #X2, _, _ = glove_data()
    #X3, _, _ = fasttext_data()

    n,p = size(X1)
    K0 = linear_kernel(X1)
    K1 = gaussian_kernel(X1, 5)
    K1_1 = gaussian_kernel(X1, 1/5)
    K2 = rational_kernel(X1)
    #K3 = neural_network_kernel(X1)

    Z1, z1 = gpca(X1, cov(X1))
    Z2, z2 = gpca(X1, K1_1)
    Z3, z3 = gpca(X1, K2)
    Z4, z4 = ics(X1, K1, K0)
    Z5, z5 = ics(X1, K2, K0)
    Z6, z6 = ics(X1, K2, K1)
    Z7, z7 = ics(X1, K2, K0.*K1)

    Z4 = reverse(Z4, dims = 2)
    Z5 = reverse(Z5, dims = 2)

    r0 = rank(X1)
    r1 = rank(Z1)
    r2 = rank(Z2)
    r3 = rank(Z3)
    r4 = rank(Z4)
    r5 = rank(Z5)
    r6 = rank(Z6)
    r7 = rank(Z7)
    r = [r0, r1, r2, r3, r4, r5, r6,r7]

    #grid for Acc,10,25,50,100,300, 601
    sd = 10
    grid = [5,10,15,25,50,100]#,300,601]
    res = zeros(length(r),length(grid)+2)
    Z = [X1,Z1,Z2,Z3,Z4,Z5,Z6,Z7]
    for i in 1:length(Z)
        Y = transform_z(Z[i])
        for j in 1:length(grid)
            #ind2 = minimum([r[i], grid[6]])
            #ind2 = minimum([r[i] - 15, grid[5]])
            res[i,j] =  mcc(stratified_k_fold(Y[:,1:grid[j]], labels, 10, seed = sd)[2])
        end
        res[i,end-1] = mcc(stratified_k_fold(Y[:,1:r[i]], labels, 10, seed = sd)[2])
        res[i,end] = maximum([mcc(stratified_k_fold(Y[:,1:k], labels, 10, seed = sd)[2]) for k in 3:20])
    end
    res  = 100*round.(res, digits = 4)

    clr = get_default_colour()

    plt0 = scatter(ColVecs(tsne2(transform_z(X1), 2, 30))..., legend = :outerright,
            color = clr[classes], group = labels,
            xlabel = "Dimension 1", ylabel = "Dimension 2",
            title = "Word2Vec data visualisation by t-SNE \n (N = 601, P = 300, Classes: 11)")

    savefig(plt0, "Figures/word2vec_tsne.svg")
    return res
end

function glove_experiments()
    X1, labels, classes = glove_data()

    n,p = size(X1)
    K0 = linear_kernel(X1)
    K1 = gaussian_kernel(X1, 3)
    K1_1 = gaussian_kernel(X1, 1/3)
    K2 = rational_kernel(X1)
    #K3 = neural_network_kernel(X1)

    Z1, z1 = gpca(X1, cov(X1))
    Z2, z2 = gpca(X1, K1_1)
    Z3, z3 = gpca(X1, K2)
    Z4, z4 = ics(X1, K1, K0)
    Z5, z5 = ics(X1, K2, K0)
    Z6, z6 = ics(X1, K2, K1)
    Z7, z7 = ics(X1, K2, K0.*K1)

    Z4 = reverse(Z4, dims = 2)
    Z5 = reverse(Z5, dims = 2)

    r0 = rank(X1)
    r1 = rank(Z1)
    r2 = rank(Z2)
    r3 = rank(Z3)
    r4 = rank(Z4)
    r5 = rank(Z5)
    r6 = rank(Z6)
    r7 = rank(Z7)
    r = [r0, r1, r2, r3, r4, r5, r6,r7]

    #grid for Acc,10,25,50,100,300, 601
    sd = 10
    grid = [5,10,15,25,50,100]#,300,601]
    res = zeros(length(r),length(grid)+2)
    Z = [X1,Z1,Z2,Z3,Z4,Z5,Z6,Z7]
    for i in 1:length(Z)
        Y = transform_z(Z[i])
        for j in 1:length(grid)
            #ind2 = minimum([r[i], grid[6]])
            #ind2 = minimum([r[i] - 15, grid[5]])
            res[i,j] =  mcc(stratified_k_fold(Y[:,1:grid[j]], labels, 10, seed = sd)[2])
        end
        res[i,end-1] = mcc(stratified_k_fold(Y[:,1:r[i]], labels, 10, seed = sd)[2])
        res[i,end] = maximum([mcc(stratified_k_fold(Y[:,1:k], labels, 10, seed = sd)[2]) for k in 3:20])
    end
    res  = 100*round.(res, digits = 4)

    return res
end

function fasttext_experiments()
    X1, labels, classes = fasttext_data()

    n,p = size(X1)
    K0 = linear_kernel(X1)
    K1 = gaussian_kernel(X1, 1)
    K1_1 = gaussian_kernel(X1, median_trick(X1))
    K2 = rational_kernel(X1)
    #K3 = neural_network_kernel(X1)

    Z1, z1 = gpca(X1, cov(X1))
    Z2, z2 = gpca(X1, K1_1)
    Z3, z3 = gpca(X1, K2)
    Z4, z4 = ics(X1, K1, K0)
    Z5, z5 = ics(X1, K2, K0)
    Z6, z6 = ics(X1, K2, K1)
    Z7, z7 = ics(X1, K2, K0.*K1)

    Z4 = reverse(Z4, dims = 2)
    Z5 = reverse(Z5, dims = 2)

    r0 = rank(X1)
    r1 = rank(Z1)
    r2 = rank(Z2)
    r3 = rank(Z3)
    r4 = rank(Z4)
    r5 = rank(Z5)
    r6 = rank(Z6)
    r7 = rank(Z7)
    r = [r0, r1, r2, r3, r4, r5, r6,r7]

    #grid for Acc,10,25,50,100,300, 601
    sd = 10
    grid = [5,10,15,25,50,100]#,300,601]
    res = zeros(length(r),length(grid)+2)
    Z = [X1,Z1,Z2,Z3,Z4,Z5,Z6,Z7]
    for i in 1:length(Z)
        Y = transform_z(Z[i])
        for j in 1:length(grid)
            #ind2 = minimum([r[i], grid[6]])
            #ind2 = minimum([r[i] - 15, grid[5]])
            res[i,j] =  mcc(stratified_k_fold(Y[:,1:grid[j]], labels, 10, seed = sd)[2])
        end
        res[i,end-1] = mcc(stratified_k_fold(Y[:,1:r[i]], labels, 10, seed = sd)[2])
        res[i,end] = maximum([mcc(stratified_k_fold(Y[:,1:k], labels, 10, seed = sd)[2]) for k in 3:20])
    end
    res  = 100*round.(res, digits = 4)

    Y = transform_z(Z4)
    plt = heatmap_plot(stratified_k_fold(Y[:,1:16], labels, 10, seed = sd)[2], labels,
    title = "FastText: Stratified 10-fold Cross-Validation Classification Results of {Gaussian,Linear} Pair [16 Components]")
    savefig(plt, "Figures/word2vec_cm.svg")

    clr = get_default_colour()

    plt0 = scatter(ColVecs(tsne2(transform_z(X1), 2, 30))..., legend = :outerright,
            color = clr[classes], group = labels,
            xlabel = "Dimension 1", ylabel = "Dimension 2",
            title = "FastText data visualisation by t-SNE \n (Using default PCA{$(ts("Cov",2))} initialisation)")

    savefig(plt0, "Figures/fasttext_tsne.svg")

    plt1 = scatter(ColVecs(tsne2(transform_z(Y[:,1:16]), 2, 30))..., legend = :outerright,
            color = clr[classes], group = labels,
            xlabel = "Dimension 1", ylabel = "Dimension 2",
            title = "FastText data visualisation by t-SNE \n (Using ICS{Gaussian,Linear} initialisation)")

    savefig(plt1, "Figures/fasttext_tsne2.svg")
    return res
end

function code_experiment()

    iris, species, labels = iris_data()                                     #1
    X, κ = ics(iris, linear_kernel(iris), eye(length(species)))             #2
    Y, λ = gpca(iris, rational_kernel(iris, median_trick(iris)))            #3
    X, Y = transform_z(X), transform_z(Y)                                   #4
    plt = contour_plot_ind(X, Y, species, species, (1,2,4), (1,2,3),        #5
            κ, λ, contour_func = svm2, title = ("Linear","Rational"))       #6
    _, acc_iris = svm2(iris, labels) #-> 0.9933                             #7
    _, acc_Y = svm2(Y, labels) #-> 1.0                                      #8
    mcc(stratified_k_fold(Y[:,1:6], labels, seed = 52)[2]) #-> 0.9706       #9
    savefig(plt, "Figures/iris_example.svg")                                #10

end

function weighting_kernel_graph()
    x = collect(-0.0:0.01:3.0)

    kernel_func0(x::Real) = abs(x) <= 1.0 ? 0.5 : 0.0
    kernel_func1(x::Real) = abs(x) <= 1.0 ? 1.0-abs(x) : 0.0
    kernel_func2(x::Real) = abs(x) <= 1.0 ? 0.75*(1.0-x^2) : 0.0
    kernel_func3(x::Real) = abs(x) <= 1.0 ? 0.9375*(1.0-x^2)^2 : 0.0
    kernel_func4(x::Real) = abs(x) <= 1.0 ? 1.09375*(1.0-x^2)^3 : 0.0
    kernel_func5(x::Real) = abs(x) <= 1.0 ? 70.0/81.0*(1.0-abs(x)^3)^3 : 0.0
    kernel_func6(x::Real) = abs(x) <= 1.0 ? 0.25*pi*cos(0.5*pi*x) : 0.0
    kernel_func7(x::Real) = inv(sqrt(2*pi))*exp(-0.5*x^2)
    kernel_func8(x::Real) = inv(2+exp(x)+exp(-x))
    kernel_func9(x::Real) = 2.0/pi*inv(exp(x)+exp(-x))
    kernel_func10(x::Real) = 0.5*exp(-abs(x)/sqrt(2))*sin(abs(x)/sqrt(2)+0.25*pi)

    clr = distinguishable_colors(14)
    #clr = get_different_colour(11)
    plt = plot(x, kernel_func0.(x), label = "Uniform", legend = :outerright, color = clr[1],
    xlabel = "Distance", ylabel = "Relative Weight", title = "        Graphical Overview of Certain Weighting Kernels")
    plot!(x, kernel_func1.(x), label = "Triangular", color = clr[2])
    plot!(x, kernel_func2.(x), label = "Epanechnikov", color = clr[3])
    plot!(x, kernel_func3.(x), label = "Quartic", color = clr[4])
    plot!(x, kernel_func4.(x), label = "Triweight", color = clr[5])
    plot!(x, kernel_func5.(x), label = "Tricube", color = clr[6])
    plot!(x, kernel_func6.(x), label = "Cosine", color = clr[7])
    plot!(x, kernel_func7.(x), label = "Gaussian", color = clr[8])
    plot!(x, kernel_func8.(x), label = "Logistic", color = clr[9])
    plot!(x, kernel_func9.(x), label = "Sigmoid", color = clr[10])
    plot!(x, kernel_func10.(x), label = "Silverman", color = clr[11])

    savefig(plt, "Figures/kernel_plot.svg")
    return plt
end
