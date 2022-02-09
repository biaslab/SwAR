export plot_generated, plot_gamma, plot_theta, plot_states

using Plots
using PGFPlotsX
using Parameters
using LaTeXStrings
using ColorSchemes
# pgfplotsx()
# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");

function plot_generated(outputs, path)
        plt_swar = @pgf Axis({
        title="Generated SWAR process",
        yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
        grid="major",
        xmin=0.0, xmax=5000,
        yminorgrids=true,
        tick_align="outside",
        scaled_y_ticks = false,
        xlabel=L"t", ylabel="value"
    },
    Plot({no_marks,color="orange"}, Coordinates(collect(1:length(outputs)), outputs)))

    pgfsave(path, plt_swar)
end

function plot_gamma(n_buckets, gen_states, result, prec_set, path)
    mγs = result.mγs
    real_states = last.(findmax.(gen_states))

    plt_gamma = @pgf Axis(
    {   xlabel="frame "*L" \sharp",
        xmin=0.0,
        legend_pos = "north east",
        legend_cell_align="{left}",
        grid = "major",
        ylabel=L"\gamma",
        legend_style = "{nodes={scale=0.5, transform shape}}",
    },
    Plot({no_marks,color="blue!70"}, Coordinates(collect(1:n_buckets), [prec_set[state] for state in real_states[1:end-1]])), LegendEntry("generated"),
    Plot({no_marks,color="black", style ="{dashed}"}, Coordinates(collect(1:n_buckets), mean.(mγs[end]))),
    Plot({"name path=f", no_marks,color="black",opacity=0.2 }, Coordinates(collect(1:n_buckets), mean.(mγs[end]) .+  std.(mγs[end]))),
    Plot({"name path=g", no_marks,color="black",opacity=0.2 }, Coordinates(collect(1:n_buckets), mean.(mγs[end]) .-  std.(mγs[end]))),
    Plot({ thick, color = "blue", fill = "black", opacity = 0.2 },
            raw"fill between [of=f and g]"), LegendEntry("inferred")
)
    pgfsave(path, plt_gamma)

end

function plot_theta(n_buckets, gen_states, result, coefs_set, path, index)
    dimension(n) = (x) -> map(i -> i[n], x)
    real_states = last.(findmax.(gen_states))
    mθs = result.mθs
    plt_theta = @pgf Axis(
        {   xlabel="frame "*L" \sharp",
            xmin=0.0,
            legend_pos = "south west",
            legend_cell_align="{left}",
            grid = "major",
            ylabel=L"\theta_{%$(index)}",
            legend_style = "{nodes={scale=0.5, transform shape}}",
        },
        Plot({no_marks,color="blue!70"}, Coordinates(collect(1:n_buckets), [coefs_set[state][index] for state in real_states[1:end-1]])), LegendEntry("generated"),
        Plot({no_marks,color="black", style ="{dashed}"}, Coordinates(collect(1:n_buckets), mean.(mθs[end]) |> dimension(index))),
        Plot({"name path=f", no_marks,color="black",opacity=0.2 }, Coordinates(collect(1:n_buckets), (mean.(mθs[end]) |> dimension(index)) .+  (sqrt.(var.(mθs[end]) |> dimension(index))) )),
        Plot({"name path=g", no_marks, color="black",opacity=0.2}, Coordinates(collect(1:n_buckets), (mean.(mθs[end]) |> dimension(index)) .-  (sqrt.(var.(mθs[end]) |> dimension(index))) )),
        Plot({ thick, color = "blue", fill = "black", opacity = 0.2 },
                    raw"fill between [of=f and g]"), LegendEntry("inferred")
        )

    pgfsave(path, plt_theta)
end

function plot_states(gen_states, result, path)
    mzs = result.mzs
    real_states = last.(findmax.(gen_states))
    plt_states = @pgf Axis({
        yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
        legend_style="{at={(0.01,0.3)},anchor=south west}",
        grid="major",
        yminorgrids=true,
        xmin=0.0,
        tick_align="outside",
        scaled_y_ticks = false,
        ytick_distance=1, grid = "major", style={"ultra thin"},
        width="20cm", height="6cm",
        xlabel="frame "*L" \sharp", ylabel="state "*L" \sharp",
    },
    Plot(
        {only_marks,color="black",fill_opacity=0.0,line_width="1pt", mark_size=4.0, mark="*"},
            Table(
                {x = "x", y = "y"},
                x = collect(1:length(real_states)), y = real_states
            )
        ), LegendEntry("active state"),
    Plot(
        {fill="black", only_marks,color="red", line_width="1pt", mark_size=3.5, opacity=1.0, mark="x"},
            Table(
                {x = "x", y = "y"},
                x = collect(1:length(mzs[end])), y = round.(mean.(mzs[end][1:end]))
            ),
        ), LegendEntry("inferred state"),
    )
    pgfsave(path, plt_states)

end