export infgamma, InferenceResults
export generateSwAR, generate_coefficients, generate_priors

using Distributions
using LinearAlgebra
using Random
import PolynomialRoots.roots

struct InferenceResults
    mzs
    mγs
    mθs
    mA
    mas
    mbs
    mms
    mws
    mfe
end

infgamma(T, x; ϵ = 1e-3) = GammaShapeRate{T}(x^2 / ϵ, x / ϵ)

function generate_coefficients(seed, order::Int)
    rng = MersenneTwister(seed)

    stable = false
    true_a = []
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while !stable
        true_a = randn(rng, order)
        coefs =  append!([1.0], -true_a)
        if false in ([abs(root) for root in roots(coefs)] .> 1)
            continue
        else
            stable = true
        end
    end
    return true_a
end

function generateSwAR(seed, n_samples, l_slice, n_states, α, coefs_set, prec_set)
    @assert length(coefs_set) == length(prec_set)
    ARorder = length(coefs_set[1])

    rng = MersenneTwister(seed)
    dirichlet = Dirichlet(α)
    A_        = [rand(rng, dirichlet) for _ in 1:n_states]
    A_matrix  = reduce(hcat, A_)
    s_0       = zeros(n_states); s_0[1] = 1.0
    x         = 0.1*ones(ARorder)

    index = rand(rng, 1:n_states)
    states  = [s_0]

    for i in 1:n_samples
        if mod(i, l_slice) == 0
            a = A_matrix*states[end]
            index = rand(rng, Categorical(a/sum(a)))
            push!(states, zeros(n_states)); states[end][index] = 1.0;
        end

        dist = Normal(dot(coefs_set[index], x[end:-1:end-ARorder+1]), sqrt(1/prec_set[index]))
        push!(x, rand(rng, dist))
    end

    inputs = x[1:n_samples+1]
    outputs = circshift(x, -1)

    inputs = inputs[1:end-1]
    outputs = outputs[2:end-1]

    inputs_ = [inputs[i+ARorder-1:-1:i] for i in 1:length(inputs)-ARorder]
    outputs_ = outputs[1:size(inputs_, 1)];
    A_matrix, states, (inputs_, outputs_)
end

function generate_priors(coefs_set, prec_set)

    n_states       = length(coefs_set)
    dimensionality = length(first(coefs_set))

    priors_as = map(γ -> infgamma(Float64, γ), prec_set)
    priors_bs = map(_ -> infgamma(Float64, 1.0), prec_set)

    priors_ms = map(θ -> MvGaussianMeanPrecision(θ[1], θ[2]), 
                    zip(coefs_set, [1e4*diageye(dimensionality) for _ in 1:length(coefs_set)]))
    priors_ws = map(_ -> (dimensionality, diageye(dimensionality)), coefs_set)
    
    prior_s = fill(1.0 / n_states, n_states)
    prior_A = ones(n_states, n_states)
    

    return priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A
end