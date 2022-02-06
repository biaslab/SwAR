export SwARParameters, switching_ar, inference_swar

using GraphPPL
using ReactiveMP
using Rocket
using Distributions
using Parameters

import ProgressMeter

# create custom structure for model parameters for simplicity
struct SwARParameters
    n_states  # number of states
    priors_as   # tuple of priors for variable a
    priors_bs   # tuple of priors for variable b
    priors_ms   # tuple of priors for variable m
    priors_ws   # tuple of priors for variable W
    prior_s     # prior of variable s
    prior_A     # prior of variable A
end

@model [ default_factorisation = MeanField() ] function switching_ar(n_samples, n_buckets, parameters)
    
    n_states   = parameters.n_states
    priors_as  = parameters.priors_as
    priors_bs  = parameters.priors_bs
    priors_ms  = parameters.priors_ms
    priors_ws  = parameters.priors_ws
    prior_s    = parameters.prior_s
    prior_A    = parameters.prior_A

    ARorder    = length(prior_s)

    A ~ MatrixDirichlet(prior_A)
    
    z_0 ~ Categorical(prior_s)

    # allocate vectors of random variables
    as = randomvar(n_states, prod_constraint = ProdGeneric(), form_constraint = PointMassFormConstraint(starting_point=(args...)->ones(1)))
    bs = randomvar(n_states)
    ms = randomvar(n_states)
    ws = randomvar(n_states)

    for i in 1:n_states
        as[i] ~ GammaShapeRate(shape(priors_as[i]), rate(priors_as[i]))
        bs[i] ~ GammaShapeRate(shape(priors_bs[i]), rate(priors_bs[i]))
        ms[i] ~ MvNormalMeanCovariance(mean(priors_ms[i]), cov(priors_ms[i]))
        ws[i] ~ Wishart(priors_ws[i][1], priors_ws[i][2])
    end

    z  = randomvar(n_buckets)
    γ  = randomvar(n_buckets)
    θ  = randomvar(n_buckets)
    
    dp = randomvar(n_samples)
    x  = datavar(Vector{Float64}, n_samples)
    y  = datavar(Float64, n_samples)

    tas = tuple(as...)
    tbs = tuple(bs...)
    tms = tuple(ms...)
    tws = tuple(ws...)

    z_prev = z_0
    for i in 1:n_buckets
        z[i] ~ Transition(z_prev, A) where { q = q(out, in)q(a) }
        γ[i] ~ GammaMixture(z[i], tas, tbs)
        θ[i] ~ GaussianMixture(z[i], tms, tws) where { q = MeanField() }
        z_prev = z[i]
    end
    
    k = div(n_samples + ARorder, n_buckets)

    for i in 1:n_samples
        r     = clamp(div(i - 1, k) + 1, 1, n_buckets)
        dp[i] ~ dot(x[i], θ[r])
        y[i]  ~ NormalMeanPrecision(dp[i], γ[r]) 
    end

    return z, A, as, bs, ms, ws, θ, γ, y, x
end


function inference_swar(inputs, outputs, n_buckets, n_its, parameters; with_progress = true)
    
    n_samples = length(outputs)

    @unpack n_states, priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A = parameters
    
    ARorder = size(priors_ms[1])[1]

    model, (z, A, as, bs, ms, ws, θs, γs, y, x) = switching_ar(n_samples, n_buckets, parameters, options=(limit_stack_depth=100,));
    
    mzs     = keep(Vector{Marginal})
    mA      = keep(Marginal)
    mas     = keep(Vector{Marginal})
    mbs     = keep(Vector{Marginal})
    mms     = keep(Vector{Marginal})
    mws     = keep(Vector{Marginal})
    mθs     = keep(Vector{Marginal})
    mγs     = keep(Vector{Marginal})
    fe      = ScoreActor(Float64)

    subscribe!(getmarginal(A), mA)
    subscribe!(getmarginals(z), mzs)
    subscribe!(getmarginals(as), mas)
    subscribe!(getmarginals(bs), mbs)
    subscribe!(getmarginals(ms), mms)
    subscribe!(getmarginals(ws), mws)
    subscribe!(getmarginals(θs), mθs)
    subscribe!(getmarginals(γs), mγs)

    subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    setmarginal!(A, vague(MatrixDirichlet, (n_states, n_states)))

    for (i, (a, b, m, w)) in enumerate(zip(as, bs, ms, ws))
        setmarginal!(a, infgamma(Float64, 1.0, ϵ = 1.0))
        setmarginal!(b, infgamma(Float64, 1.0, ϵ = 1.0))
        setmarginal!(m, vague(MvNormalMeanCovariance, ARorder))
        setmarginal!(w, vague(Wishart, ARorder))
    end

    for (θ, γ) in zip(θs, γs)
        setmarginal!(θ, vague(MvNormalMeanCovariance, ARorder))
        setmarginal!(γ, vague(Gamma))
    end

    progress = ProgressMeter.Progress(n_its, 1)

    for _ in 1:n_its
        update!(x, inputs)
        update!(y, outputs)
        if with_progress
            ProgressMeter.next!(progress)
        end
    end

    return map(getvalues, (mzs, mγs, mθs, mA, mas, mbs, mms, mws, fe))
end