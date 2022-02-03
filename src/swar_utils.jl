export infgamma
export generateSwAR, generate_coefficients

using Distributions
using LinearAlgebra
# import PolynomialRoots.roots


infgamma(T, x; ϵ = 1e-3) = GammaShapeRate{T}(x^2 / ϵ, x / ϵ)

function generate_coefficients(order::Int)
    stable = false
    true_a = []
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while !stable
        true_a = randn(order)
        coefs =  append!([1.0], -true_a)
        if false in ([abs(root) for root in roots(coefs)] .> 1)
            continue
        else
            stable = true
        end
    end
    return true_a
end

function generateSwAR(n_samples, l_slice, n_states, α, coefs_set, prec_set)
    @assert length(coefs_set) == length(prec_set)

    ARorder = length(coefs_set[1])
    states  = Vector{Vector{Float64}}(undef, n_samples)

    dirichlet = Dirichlet(α)
    A_        = [rand(dirichlet) for _ in 1:n_states]
    A_matrix  = reduce(hcat, A_)
    s_0       = zeros(n_states); s_0[1] = 1.0
    x         = 0.1*ones(ARorder)

    index = rand(1:n_states)
    s_prev = s_0
    for i in 1:n_samples
        if mod(i, l_slice) == 0
            a = A_matrix*s_prev
            index = rand(Categorical(a/sum(a)))
        end
        states[i] = zeros(n_states); states[i][index] = 1.0;
        s_prev = states[i]

        dist = Normal(dot(coefs_set[index], x[end:-1:end-ARorder+1]), sqrt(1/prec_set[index]))
        push!(x, rand(dist))
    end
    A_matrix, states, x
end

# γs = [1, 30]
# θs = [[1.25166, -0.423974], [1.04586, -0.198375]]

# A, states, obs = generateSwAR(1000, 100, 2, 0.5ones(2), θs, γs)
# plot(last.(findmax.(states)))