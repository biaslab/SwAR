using DrWatson
@quickactivate :SwARExperiments

using Distributions
using ReactiveMP
using Random
using Plots
using LinearAlgebra
using StatsBase

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others
experiments = dict_list(Dict(
    "n_samples" => 5000,
    "l_slice"   => 100,
    "n_states"   => collect(2:2),
    "ar_order"   => collect(2:2),
    "iterations" => 20,
    "seed" => collect(2:2)
))

function run_experiment(params)

    # We unpack all provided parameters into separate variables
    @unpack n_samples, l_slice, n_states, ar_order, iterations, seed = params

    # For reproducibility
    coefs_set = [generate_coefficients(seed*i,  ar_order) for i in 1:n_states]
    prec_set  = sample(MersenneTwister(seed), 0.1:1.0:100, n_states, replace=false)
    
    gen_A, gen_states, observations = generateSwAR(seed, n_samples, l_slice, n_states, ones(n_states), coefs_set, prec_set)
    inputs, outputs = observations[1], observations[2];
    
    priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A = generate_priors(coefs_set, prec_set)
    
    parameters = SwARParameters(n_states, priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A)
    
    result = InferenceResults(inference_swar(inputs, outputs, iterations, div(n_samples, l_slice), parameters)...)

    # Specify which information should be saved in JLD2 file
    return @strdict result parameters gen_A gen_states observations params
end

function generate_plots(input)
    try
        @unpack result, parameters, gen_A, gen_states, observations, params = input

        save_types  = (String, Real)

        if with_pgf
            pgf_path = projectdir("results", "synthetic", savename("fig", params, "tikz", allowedtypes = save_types)) 
            pgf_densities(string(params["constraint"]), mixture, result, dataset, pgf_path)
        end

        fig_path = projectdir("results", "synthetic", "2mixtures", "with_known_shape_rate", savename("fig", params, "png", allowedtypes = save_types)) 

        savefig(compare(mixture, result), fig_path)
    catch error
        @warn "Could not save plots: $error"
    end
end

results = map(experiments) do experiment
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "synthetic")
    # Types which should be used for cache file name
    save_types  = (String, Real)
    try
        result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types) do params
            run_experiment(params)
        end

        generate_plots(result)

        return result
    catch error
        return error
    end
end;