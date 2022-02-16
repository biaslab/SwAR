using DrWatson
@quickactivate :SwARExperiments

using Rocket
using ReactiveMP
using GraphPPL
using Distributions
using Random
using Parameters
using LinearAlgebra
using WAV
using Plots
using PGFPlotsX
using Colors
using LaTeXStrings
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");
using Base.Iterators


function ar_ssm(series, order)
    inputs = [reverse!(series[1:order])]
    outputs = [series[order + 1]]
    for x in series[order+2:end]
        push!(inputs, vcat(outputs[end], inputs[end])[1:end-1])
        push!(outputs, x)
    end
    return inputs, outputs
end

signal, fs = WAV.wavread("data/btb.wav")
# signal, fs = signal[1:2:end], fs/2 # downsample signal for faster processing

# priors for train and bar sounds were extracted by means of bayesian AR
# see https://github.com/biaslab/AIDA/blob/master/src/environment/ar.jl
coefs_set = [[1.2963022142609468, -0.4492847417092366], [1.2177989080828675, -0.3255546763338871]]
prec_set  = [8323.087940830268, 5524.193364033609]
l_slice   = 15000
n_buckets = div(length(signal), l_slice)
n_states  = 2

priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A = generate_priors(coefs_set, prec_set)

parameters = SwARParameters(n_states, priors_as, priors_bs, priors_ms, priors_ws, prior_s, prior_A)

inputs, outputs = ar_ssm(signal, 2)
result = InferenceResults(inference_swar(inputs, outputs, n_buckets, 20, parameters)...)

states = collect(flatten(map(e -> repeated(e, l_slice), round.(mean.(result.mzs[end][1:end])))))


plt_acoustic = @pgf Axis({
    yticklabel_style={
    "/pgf/number format/fixed,
    /pgf/number format/precision=3"
    },
    legend_pos="north west",
    grid="major",
    yminorgrids=true,
    xmin=0.0,
    xmax=8.0,
    tick_align="outside",
    each_nth_point=10,
    scaled_y_ticks = false,
    ytick_distance=0.1, grid = "major", style={"ultra thin"},
    width="8cm", height="5cm",
    xlabel="sec", ylabel="amplitude",
},
Plot(
    {no_marks,color="cyan",fill_opacity=0.0, mark_size=4.0, mark="*"},
    Coordinates(collect(0:1/fs:length(signal)/fs)[1:end], vec(signal))
    ), LegendEntry("signal"),
Plot({only_marks, scatter, scatter_src = "explicit"},
    Table(
        {x = "x", y = "y", meta = "col"},
        x = collect(0:1/fs:length(signal)/fs)[1:length(states)], y = -0.15*ones(length(states)), col = states
        ),
     ),
)

pgfsave("results/real/inferred_acoustics.svg", plt_acoustic)
pgfsave("results/real/inferred_acoustics.tikz", plt_acoustic)
