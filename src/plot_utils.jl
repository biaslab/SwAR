export plot_generated, plot_gamma, plot_theta, plot_states, plot_fe

using Plots
using PGFPlotsX
using Parameters
using LaTeXStrings
using ColorSchemes
using Colors
# pgfplotsx()
# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");

function plot_generated(outputs, l_slice, n_samples, states, path)
    n_buckets = div(n_samples, l_slice)
    from, to = 1, n_buckets - 1
    real_states = last.(findmax.(states))
    cols = distinguishable_colors(2, [RGB(0,0.0,0), RGB(1.0,1.0,0.9)], dropseed=true)
    colors = collect(Iterators.flatten(map(e -> Iterators.repeated(e, l_slice), real_states[1:end-1])))
    real_colors = map(x -> cols[x], colors)
    plt_swar = @pgf Axis({xlabel=L"t",
        title="Generated SWAR process",
            yticklabel_style={
            "/pgf/number format/fixed,
            /pgf/number format/precision=3"
            },
            ylabel="value", scaled_x_ticks="base 10:0",
            legend_pos = "outer north east",
            xmin=0.0, xmax=length(outputs),
            legend_cell_align="{left}",
            grid = "major", style={"thin"},
            width="20cm", height="10cm",
        },
        Plot({no_marks, style={"ultra thick"}, color=cols[1]}, Coordinates(zeros(1), zeros(1))), LegendEntry("AR-1"),
        Plot({no_marks, style={"ultra thick"}, color=cols[2]}, Coordinates(zeros(1), zeros(1))), LegendEntry("AR-2"),
        Iterators.flatten([
                        [Plot(
                                {no_marks, color=real_colors[from+i*l_slice-1]},
                                Coordinates(
                                    collect(from+(i-1)*l_slice:from+i*l_slice), 
                                    outputs[from+(i-1)*l_slice:from+i*l_slice]
                                ),
                            ) for i in from:to]])...
    )

    pgfsave(path, plt_swar)
end

function plot_gamma(n_buckets, gen_states, result, prec_set, path)
    mγs = result.mγs
    real_states = last.(findmax.(gen_states))

    plt_gamma = @pgf Axis(
    {   xlabel="frame "*L" \sharp",
        yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
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


function plot_fe(path, FE, vmp_its, start=3, )
    axis4 = @pgf Axis({xlabel="iteration",
                    ylabel="Bethe free energy [nats]",
                    legend_pos = "north east",
                    legend_cell_align="{left}",
                    scale = 1.0,
                    grid = "major",
                    width="20cm", height="12cm"
        },
        Plot({mark = "o", "red", mark_size=1}, Coordinates(collect(start:vmp_its), FE[start:end])), LegendEntry("BFE"))
    pgfsave(path, axis4)
    
end