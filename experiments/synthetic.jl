using DrWatson
@quickactivate :SwARExperiments

using Distributions
using ReactiveMP
using Random
using Plots
using LinearAlgebra
using StatsBase

pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others
experiments = dict_list(Dict(
    "n_samples" => 5000,
    "l_slice"   => 100,
    "n_states"   => 2,
    "ar_order"   => 2,
    "iterations" => 20,
    "seed" => collect(1:100)
))

function run_experiment(params)

    # We unpack all provided parameters into separate variables
    @unpack n_samples, l_slice, n_states, ar_order, iterations, seed = params

    # For reproducibility
    coefs_set = [generate_coefficients(seed*i,  ar_order) for i in 1:n_states]
    prec_set  = sample(MersenneTwister(seed), [0.01, 0.1, 1.0, 10.0, 100.0], n_states, replace=false)
    
    gen_A, gen_states, observations = generateSwAR(seed, n_samples, l_slice, n_states, ones(n_states), coefs_set, prec_set)
    inputs, outputs = observations[1], observations[2];
    
    # We use weakly informative priors for AR processes' AR coefficients and precisions.
    # The rest of the priors are vague
    priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A = generate_priors(coefs_set, prec_set)
    
    parameters = SwARParameters(n_states, priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A)
    
    result = InferenceResults(inference_swar(inputs, outputs, div(n_samples, l_slice), iterations, parameters)...)

    # Specify which information should be saved in JLD2 file
    return @strdict result parameters gen_A gen_states observations params coefs_set prec_set
end

function generate_plots(input, format="svg")
    try
        @unpack result, parameters, gen_A, gen_states, observations, params, coefs_set, prec_set = input

        save_types  = (String, Real)

        fig_path = projectdir("results", "synthetic", savename("generated_swar", params, format, allowedtypes = save_types))
        plot_generated(observations[2], params["l_slice"], params["n_samples"], gen_states, fig_path)

        fig_path = projectdir("results", "synthetic", savename("inferred_gamma", params, format, allowedtypes = save_types))
        plot_gamma(div(params["n_samples"], params["l_slice"]), gen_states, result, prec_set, fig_path)

        for index in 1:length(first(coefs_set))
            fig_path = projectdir("results", "synthetic", savename("inferred_theta_$(index)", params, format, allowedtypes = save_types))
            plot_theta(div(params["n_samples"], params["l_slice"]), gen_states, result, coefs_set, fig_path, index)
        end

        fig_path = projectdir("results", "synthetic", savename("inferred_states", params, format, allowedtypes = save_types))
        plot_states(gen_states, result, fig_path)

    catch error
        @warn "Could not save plots: $error"
    end
end

results = map(experiments) do experiment
    FE = []
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "synthetic")
    # Types which should be used for cache file name
    save_types  = (String, Real)
    try
        result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types) do params
            run_experiment(params)
        end
        generate_plots(result, "tikz")
        generate_plots(result, "svg")

        return result
    catch error
        @warn error
    end
end;

begin
    FE = []
    for res in results
        try
            push!(FE, res["result"].mfe)
        catch error
            @warn error
        end
    end
    FE = sum(FE)/length(FE)
    fig_path = "results/synthetic/FE.tikz"
    vmp_its = 20
    plot_fe(fig_path, FE, vmp_its)
end

begin
    cat_accs = []
    for res in results
        try
            estimated = round.(mean.(res["result"].mzs[end][1:end]))
            real = last.(findmax.(res["gen_states"]))[1:end-1]
            push!(cat_accs, cat_accuracy(estimated, real))
        catch error
            @warn error
        end
    end
    cat_acc_score = mean(cat_accs)
    println(cat_acc_score)
end