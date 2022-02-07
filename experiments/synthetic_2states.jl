using DrWatson
@quickactivate :SwARExperiments

using Distributions
using ReactiveMP
using Random
using Plots

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others
experiments = dict_list(Dict(
    "n_samples" => 5000,
    "iterations" => 20,
    "epsilon" => 100.0,
    "seed" => collect(1:10),
    "prod_constraint" => [
        @onlyif("constraint" == EM(), FoldRightProdStrategy()),
        @onlyif("constraint" == Marginalisation(), FoldLeftProdStrategy()),
    ],
    "meta" => [
        @onlyif("constraint" == EM(), nothing),
        @onlyif("constraint" == Marginalisation(), ImportanceSamplingApproximation(MersenneTwister(1234), 5000))
    ],
    "jitter" => 0.0,
    "shift" => 10.0
))

function run_experiment(params)
    # We unpack all provided parameters into separate variables
    @unpack n, iterations, seed, constraint, prod_strategy, meta, jitter, shift = params

    # For reproducibility
    rng = MersenneTwister(seed)

    mixtures  = [ Gamma(9.0, inv(27.0)), Gamma(90.0, inv(270.0)) ]
    nmixtures = length(mixtures)
    mixing    = rand(rng, nmixtures)
    mixing    = mixing ./ sum(mixing)
    mixture   = MixtureModel(mixtures, mixing)

    dataset = rand(rng, mixture, n)

    # Priors are mostly vague and use information from dataset only (random means and fixed variances)
    priors_as, priors_bs = generate_priors(dataset, nmixtures, seed = seed, ϵ = epsilon, jitter = jitter, shift = shift)

    parameters = GammaMixtureModelParameters(
        nmixtures        = nmixtures,
        priors_as        = priors_as,
        priors_bs        = priors_bs,
        prior_s          = Dirichlet(10000 * mixing),
        as_prod_strategy = prod_strategy,
        as_constraint    = constraint,
        meta             = meta
    )

    result = InferenceResults(gamma_mixture_inference(dataset, iterations, parameters)...)

    # Specify which information should be saved in JLD2 file
    return @strdict result parameters mixtures mixing mixture dataset params
end