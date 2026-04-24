#=
Generate convergence-study plots from CSV artifacts written by `generate_data.jl`.

All plots use CairoMakie and are written to `docs/src/assets/convergence/plots/`.

Usage:
    julia --project=docs docs/src/assets/convergence/generate_plots.jl [targets...]

Targets (omit to run all): h hseps p k eps wp 3d
=#

using CairoMakie
using DelimitedFiles
using Printf
using Statistics

const DATA_DIR = joinpath(@__DIR__, "data")
const PLOTS_DIR = joinpath(@__DIR__, "plots")
mkpath(PLOTS_DIR)

CairoMakie.activate!(type = "png")

# ----------------------------------------------------------------------
# Unusable (basis, poly_deg, operator) combinations
# ----------------------------------------------------------------------
# Empirically non-converging or numerically pathological — excluded from plots.
# See `scalar-operators.md` and `vector-operators.md` for prose notes.
const UNUSABLE = Set{Tuple{String, String, Int, Int}}(
    [
        # Single 2nd-derivative operators: PHS1/p=1 numerically unstable; p below
        # matched gives no convergence; shape-parameter bases at p=0 don't converge.
        ("laplacian", "PHS", 1, 1),
        ("laplacian", "PHS", 3, 1),
        ("laplacian", "IMQ", 0, 0),
        ("laplacian", "Gaussian", 0, 0),

        ("partial2", "PHS", 1, 1),
        ("partial2", "PHS", 3, 1),
        ("partial2", "IMQ", 0, 0),
        ("partial2", "Gaussian", 0, 0),

        # Mixed partials: PHS3/p=2 and PHS5/p=2 plateau; IMQ/Gaussian p=2 also diverge.
        ("mixed_partial", "PHS", 1, 1),
        ("mixed_partial", "PHS", 3, 1),
        ("mixed_partial", "PHS", 3, 2),
        ("mixed_partial", "PHS", 5, 2),
        ("mixed_partial", "IMQ", 0, 0),
        ("mixed_partial", "IMQ", 0, 2),
        ("mixed_partial", "Gaussian", 0, 0),
        ("mixed_partial", "Gaussian", 0, 2),

        # Hessian inherits the mixed-partial restriction.
        ("hessian", "PHS", 1, 1),
        ("hessian", "PHS", 3, 1),
        ("hessian", "PHS", 3, 2),
        ("hessian", "PHS", 5, 2),
        ("hessian", "IMQ", 0, 0),
        ("hessian", "IMQ", 0, 2),
        ("hessian", "Gaussian", 0, 0),
        ("hessian", "Gaussian", 0, 2),
    ]
)

is_unusable(op, fam, ord, pd) = (String(op), String(fam), Int(ord), Int(pd)) in UNUSABLE

# ----------------------------------------------------------------------
# Small CSV helper: read into Vector-of-NamedTuples
# ----------------------------------------------------------------------

function load_csv(path)
    data, hdr = readdlm(path, ','; header = true)
    cols = vec(hdr)
    rows = Vector{NamedTuple}(undef, size(data, 1))
    for i in 1:size(data, 1)
        vals = Tuple(data[i, j] for j in 1:length(cols))
        rows[i] = NamedTuple{Tuple(Symbol.(cols))}(vals)
    end
    return rows
end

function filter_rows(rows, pred)
    return [r for r in rows if pred(r)]
end

function sorted_by_N(rows)
    return sort(rows; by = r -> r.N)
end

# ----------------------------------------------------------------------
# Reference convergence slope (on N-axis, given h ∼ 1/√N in 2D / 1/∛N in 3D)
# ----------------------------------------------------------------------

function ref_slope!(ax, Ns, rate; anchor_N, anchor_err, label, dim = 2)
    # error ∝ h^rate; h ∼ N^(-1/dim)
    y = anchor_err .* (Ns ./ anchor_N) .^ (-rate / dim)
    return lines!(
        ax, Ns, y; linestyle = :dash, color = (:gray, 0.6),
        linewidth = 1.2, label = label
    )
end

# ----------------------------------------------------------------------
# Operator → plot metadata (title, ylabel, expected slopes for matched PHS)
# ----------------------------------------------------------------------

struct OpMeta
    name::String      # key in CSV operator column
    title::String     # plot title base
    ylabel::String
    rate_at_p3::Int   # expected convergence rate at poly_deg=3
    rate_label::String  # human-readable reference slope label
end

# At poly_deg = 3 the expected rate is `poly_deg + 1 - derivative_order`:
# interpolation=4, 1st-deriv=3, 2nd-deriv=2.
const OPS = [
    OpMeta("interpolation", "Interpolation", "NRMSE of u", 4, "O(h⁴)"),
    OpMeta("laplacian", "Laplacian (∇²)", "NRMSE of ∇²u", 2, "O(h²)"),
    OpMeta("gradient", "Gradient (∇)", "NRMSE of ∇u", 3, "O(h³)"),
    OpMeta("partial1", "First partial (∂/∂x₁)", "NRMSE of ∂u/∂x", 3, "O(h³)"),
    OpMeta("partial2", "Second partial (∂²/∂x₁²)", "NRMSE of ∂²u/∂x²", 2, "O(h²)"),
    OpMeta("mixed_partial", "Mixed partial (∂²/∂x₁∂x₂)", "NRMSE of ∂²u/∂x∂y", 2, "O(h²)"),
    OpMeta("hessian", "Hessian (Hu)", "Frobenius NRMSE", 2, "O(h²)"),
    OpMeta("jacobian", "Jacobian (Ju)", "Frobenius NRMSE", 3, "O(h³)"),
    OpMeta("divergence", "Divergence (∇·u)", "NRMSE of ∇·u", 3, "O(h³)"),
    OpMeta("curl2d", "Curl (∇×u, 2D)", "NRMSE of ∇×u", 3, "O(h³)"),
]

# ----------------------------------------------------------------------
# h-refinement plots — three per operator
# ----------------------------------------------------------------------

function plot_h_phs_matched!(ax, rows_for_op, rate, rate_label, op_name)
    # Show PHS3, PHS5, PHS7 all at poly_deg=3 so differences between curves isolate
    # RBF smoothness from polynomial augmentation.
    POLY_DEG = 3
    orders = [3, 5, 7]
    cmap = Makie.wong_colors()
    plotted_Ns = Int[]
    leftmost_errs = Float64[]
    for (i, ord) in enumerate(orders)
        is_unusable(op_name, "PHS", ord, POLY_DEG) && continue
        rs = sorted_by_N(
            filter_rows(
                rows_for_op,
                r -> r.family == "PHS" && r.phs_order == ord && r.poly_deg == POLY_DEG
            )
        )
        isempty(rs) && continue
        Ns = [r.N for r in rs]
        errs = [r.nrmse for r in rs]
        scatterlines!(
            ax, Ns, errs; color = cmap[i + 1],
            label = "PHS$ord, p=$POLY_DEG", markersize = 8, linewidth = 1.8
        )
        plotted_Ns = Ns
        push!(leftmost_errs, first(errs))
    end
    # Reference slope anchored at the geometric mean of curves at leftmost N
    # so the dashed line sits among the data rather than below it.
    return if !isempty(leftmost_errs)
        Ns_line = range(minimum(plotted_Ns), maximum(plotted_Ns); length = 50) |> collect
        anchor_err = exp(mean(log.(leftmost_errs)))
        ref_slope!(
            ax, Ns_line, rate;
            anchor_N = first(plotted_Ns), anchor_err = anchor_err,
            label = rate_label
        )
    end
end

function plot_h_phs_polydeg_panel!(ax, rows_for_op, phs_order, poly_degs, title, op_name)
    cmap = Makie.wong_colors()
    for (i, p) in enumerate(poly_degs)
        is_unusable(op_name, "PHS", phs_order, p) && continue
        rs = sorted_by_N(
            filter_rows(
                rows_for_op,
                r -> r.family == "PHS" && r.phs_order == phs_order && r.poly_deg == p
            )
        )
        isempty(rs) && continue
        Ns = [r.N for r in rs]
        errs = [r.nrmse for r in rs]
        scatterlines!(
            ax, Ns, errs; color = cmap[i],
            label = "p=$p", markersize = 7, linewidth = 1.5
        )
    end
    return ax.title = title
end

function plot_h_shape_panel!(ax, rows_for_op, family, poly_degs, title, op_name)
    cmap = Makie.wong_colors()
    for (i, p) in enumerate(poly_degs)
        is_unusable(op_name, family, 0, p) && continue
        rs = sorted_by_N(
            filter_rows(
                rows_for_op,
                r -> r.family == family && r.poly_deg == p
            )
        )
        isempty(rs) && continue
        Ns = [r.N for r in rs]
        errs = [r.nrmse for r in rs]
        scatterlines!(
            ax, Ns, errs; color = cmap[i],
            label = "p=$p", markersize = 7, linewidth = 1.5
        )
    end
    return ax.title = title
end

function make_h_plots(h_rows)
    for op in OPS
        rows_op = filter(r -> r.operator == op.name, h_rows)
        isempty(rows_op) && continue

        # A — PHS3, PHS5, PHS7 at fixed poly_deg=3 (isolates PHS-order effect)
        fig = Figure(size = (640, 460))
        ax = Axis(
            fig[1, 1]; xscale = log10, yscale = log10,
            xlabel = "N", ylabel = op.ylabel,
            title = "$(op.title): PHS orders at poly_deg=3"
        )
        plot_h_phs_matched!(ax, rows_op, op.rate_at_p3, op.rate_label, op.name)
        axislegend(ax; position = :lb, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "$(op.name)_phs_matched.png"), fig)

        # B — PHS poly_deg sweeps (3-panel)
        fig = Figure(size = (1200, 420))
        for (j, (ord, ps)) in enumerate([(3, 1:4), (5, 2:5), (7, 3:6)])
            ax = Axis(
                fig[1, j]; xscale = log10, yscale = log10,
                xlabel = "N", ylabel = j == 1 ? op.ylabel : ""
            )
            plot_h_phs_polydeg_panel!(
                ax, rows_op, ord, collect(ps),
                "PHS$ord, poly_deg sweep", op.name
            )
            axislegend(ax; position = :lb, framevisible = false, labelsize = 9)
        end
        Label(
            fig[0, :], "$(op.title): effect of polynomial degree for each PHS order",
            fontsize = 14
        )
        save(joinpath(PLOTS_DIR, "$(op.name)_phs_polydeg_sweep.png"), fig)

        # C — IMQ / Gaussian poly_deg sweep (2-panel)
        fig = Figure(size = (950, 420))
        for (j, family) in enumerate(("IMQ", "Gaussian"))
            ax = Axis(
                fig[1, j]; xscale = log10, yscale = log10,
                xlabel = "N", ylabel = j == 1 ? op.ylabel : ""
            )
            plot_h_shape_panel!(
                ax, rows_op, family, 0:4,
                "$family (ε=1), poly_deg sweep", op.name
            )
            axislegend(ax; position = :lb, framevisible = false, labelsize = 9)
        end
        Label(
            fig[0, :], "$(op.title): shape-parameter basis poly_deg sweep",
            fontsize = 14
        )
        save(joinpath(PLOTS_DIR, "$(op.name)_imq_gaussian_polydeg.png"), fig)

        println("  h-plots: $(op.name) ✓")
    end
    return
end

# ----------------------------------------------------------------------
# Fixed-ε vs scaled-ε comparison (shape-parameter bases page)
# ----------------------------------------------------------------------
#
# Each figure is two panels (shared y-axis):
#   Left:  IMQ / Gaussian at fixed ε=1 (from h_refinement.csv) — shows
#          divergence as N grows due to the RBF uncertainty principle.
#   Right: Same bases at scaled ε = c/h (from h_refinement_scaled_eps.csv)
#          — shows clean convergence once ε·h is held near its sweet spot.
# A PHS5/p=3 reference curve anchors both panels.

const EPS_COMPARE_OPS = [
    ("interpolation", "Interpolation", "NRMSE of u"),
    ("laplacian", "Laplacian (∇²)", "NRMSE of ∇²u"),
    ("mixed_partial", "Mixed partial (∂²/∂x₁∂x₂)", "NRMSE of ∂²u/∂x∂y"),
]

function _plot_eps_compare_panel_fixed!(
        ax, h_rows, op_name, poly_degs, style_by_family,
    )
    cmap = Makie.wong_colors()
    for family in ("IMQ", "Gaussian")
        for (i, p) in enumerate(poly_degs)
            is_unusable(op_name, family, 0, p) && continue
            rs = sorted_by_N(
                filter_rows(
                    h_rows,
                    r -> r.operator == op_name && r.family == family &&
                        r.phs_order == 0 && r.poly_deg == p,
                ),
            )
            isempty(rs) && continue
            Ns = [r.N for r in rs]
            errs = [r.nrmse for r in rs]
            scatterlines!(
                ax, Ns, errs;
                color = cmap[i + 2], linestyle = style_by_family[family],
                marker = family === "IMQ" ? :circle : :diamond,
                markersize = 8, linewidth = 1.5,
                label = "$family p=$p",
            )
        end
    end
    return
end

function _plot_eps_compare_panel_scaled!(
        ax, scaled_rows, op_name, poly_degs, style_by_family,
    )
    cmap = Makie.wong_colors()
    for family in ("IMQ", "Gaussian")
        for (i, p) in enumerate(poly_degs)
            is_unusable(op_name, family, 0, p) && continue
            rs = sorted_by_N(
                filter_rows(
                    scaled_rows,
                    r -> r.operator == op_name && r.family == family &&
                        r.poly_deg == p,
                ),
            )
            isempty(rs) && continue
            Ns = [r.N for r in rs]
            errs = [r.nrmse for r in rs]
            scatterlines!(
                ax, Ns, errs;
                color = cmap[i + 2], linestyle = style_by_family[family],
                marker = family === "IMQ" ? :circle : :diamond,
                markersize = 8, linewidth = 1.5,
                label = "$family p=$p",
            )
        end
    end
    return
end

function _plot_phs_reference!(ax, h_rows, op_name)
    # PHS5/p=3 as a stable reference across both panels.
    rs = sorted_by_N(
        filter_rows(
            h_rows,
            r -> r.operator == op_name && r.family == "PHS" &&
                r.phs_order == 5 && r.poly_deg == 3,
        ),
    )
    isempty(rs) && return
    Ns = [r.N for r in rs]
    errs = [r.nrmse for r in rs]
    scatterlines!(
        ax, Ns, errs;
        color = (:black, 0.7), marker = :utriangle, markersize = 7,
        linewidth = 1.5, linestyle = :solid, label = "PHS5 p=3 (ref)",
    )
    return
end

function make_eps_compare_plots(h_rows, scaled_rows)
    style_by_family = Dict("IMQ" => :solid, "Gaussian" => :dash)
    for (op_name, op_title, ylabel) in EPS_COMPARE_OPS
        # Pick poly_degs to show: p=2 and p=3 where both are usable. For
        # mixed_partial, p=2 is in UNUSABLE for shape-parameter bases, so
        # only p=3 shows up there (both panels).
        poly_degs = [2, 3]
        c = get(
            Dict(
                "interpolation" => 0.15,
                "laplacian" => 0.03,
                "mixed_partial" => 0.03,
            ), op_name, 0.03
        )

        fig = Figure(size = (1100, 460))
        ax_left = Axis(
            fig[1, 1]; xscale = log10, yscale = log10,
            xlabel = "N", ylabel = ylabel,
            title = "Fixed ε = 1",
        )
        ax_right = Axis(
            fig[1, 2]; xscale = log10, yscale = log10,
            xlabel = "N", ylabel = "",
            title = "Scaled ε = $c / h",
        )

        _plot_phs_reference!(ax_left, h_rows, op_name)
        _plot_eps_compare_panel_fixed!(
            ax_left, h_rows, op_name, poly_degs, style_by_family,
        )

        _plot_phs_reference!(ax_right, h_rows, op_name)
        _plot_eps_compare_panel_scaled!(
            ax_right, scaled_rows, op_name, poly_degs, style_by_family,
        )

        linkyaxes!(ax_left, ax_right)

        axislegend(ax_left; position = :lb, framevisible = false, labelsize = 9)
        axislegend(ax_right; position = :lb, framevisible = false, labelsize = 9)

        Label(
            fig[0, :], "$(op_title): fixed-ε divergence vs scaled-ε convergence",
            fontsize = 14,
        )
        save(joinpath(PLOTS_DIR, "$(op_name)_imq_gaussian_eps_compare.png"), fig)
        println("  eps-compare: $(op_name) ✓")
    end
    return
end

# ----------------------------------------------------------------------
# p-refinement (§12)
# ----------------------------------------------------------------------

function plot_p_refinement(p_rows)
    operators = ["interpolation", "laplacian", "gradient"]
    titles = Dict(
        "interpolation" => "Interpolation",
        "laplacian" => "Laplacian (∇²)",
        "gradient" => "Gradient (∇)"
    )
    cmap = Makie.wong_colors()
    families = [
        ("PHS", 3, "PHS3"), ("PHS", 5, "PHS5"), ("PHS", 7, "PHS7"),
        ("IMQ", 0, "IMQ ε=1"), ("Gaussian", 0, "Gaussian ε=1"),
    ]

    for op in operators
        fig = Figure(size = (640, 460))
        ax = Axis(
            fig[1, 1]; yscale = log10,
            xlabel = "polynomial degree", ylabel = "NRMSE",
            title = "$(titles[op]): p-refinement at N=2500"
        )
        rows_op = filter(r -> r.operator == op, p_rows)
        for (i, (family, ord, lab)) in enumerate(families)
            rs = sort(
                filter(r -> r.family == family && r.phs_order == ord, rows_op);
                by = r -> r.poly_deg
            )
            isempty(rs) && continue
            pd = [r.poly_deg for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(
                ax, pd, er; color = cmap[i], label = lab,
                markersize = 8, linewidth = 1.8
            )
        end
        axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "p_refinement_$(op).png"), fig)
    end

    # Cost plot: build_time_s vs poly_deg for laplacian
    fig = Figure(size = (640, 460))
    ax = Axis(
        fig[1, 1]; yscale = log10,
        xlabel = "polynomial degree", ylabel = "build time (s)",
        title = "Weight-build cost vs poly_deg (laplacian, N=2500)"
    )
    rows_op = filter(r -> r.operator == "laplacian", p_rows)
    for (i, (family, ord, lab)) in enumerate(families)
        rs = sort(
            filter(r -> r.family == family && r.phs_order == ord, rows_op);
            by = r -> r.poly_deg
        )
        isempty(rs) && continue
        pd = [r.poly_deg for r in rs]
        bt = [r.build_time_s for r in rs]
        scatterlines!(
            ax, pd, bt; color = Makie.wong_colors()[i], label = lab,
            markersize = 8, linewidth = 1.8
        )
    end
    axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
    save(joinpath(PLOTS_DIR, "p_refinement_cost.png"), fig)
    return println("  p-refinement plots ✓")
end

# ----------------------------------------------------------------------
# k-refinement (§13)
# ----------------------------------------------------------------------

function plot_k_refinement(k_rows)
    operators = [
        ("interpolation", "Interpolation"),
        ("laplacian", "Laplacian (∇²)"),
        ("gradient", "Gradient (∇)"),
    ]
    cmap = Makie.wong_colors()
    configs = [(3, 3, "PHS3, p=3"), (5, 3, "PHS5, p=3"), (7, 3, "PHS7, p=3")]

    for (op_key, op_title) in operators
        fig = Figure(size = (640, 460))
        ax = Axis(
            fig[1, 1]; yscale = log10,
            xlabel = "stencil size k", ylabel = "NRMSE",
            title = "$op_title: k-refinement at N=4900"
        )
        rows_op = filter(r -> r.operator == op_key, k_rows)
        for (i, (ord, pdeg, lab)) in enumerate(configs)
            rs = sort(
                filter(r -> r.phs_order == ord && r.poly_deg == pdeg, rows_op);
                by = r -> r.k
            )
            isempty(rs) && continue
            ks = [r.k for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(
                ax, ks, er; color = cmap[i], label = lab,
                markersize = 8, linewidth = 1.8
            )
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "k_refinement_$(op_key).png"), fig)
    end
    return println("  k-refinement plots ✓")
end

# ----------------------------------------------------------------------
# ε-refinement (§14)
# ----------------------------------------------------------------------

function plot_eps_refinement(eps_rows)
    cmap = Makie.wong_colors()

    # 14.1, 14.2: IMQ / Gaussian, three operators, poly_deg=2
    for family in ("IMQ", "Gaussian")
        fig = Figure(size = (640, 460))
        ax = Axis(
            fig[1, 1]; xscale = log10, yscale = log10,
            xlabel = "ε (shape parameter)", ylabel = "NRMSE",
            title = "$family shape-parameter sensitivity (poly_deg=2, N=900)"
        )
        for (i, op) in enumerate(("interpolation", "laplacian", "gradient"))
            rs = sort(
                filter(r -> r.operator == op && r.family == family && r.poly_deg == 2, eps_rows);
                by = r -> r.eps
            )
            isempty(rs) && continue
            eps = [r.eps for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(
                ax, eps, er; color = cmap[i], label = op,
                markersize = 8, linewidth = 1.8
            )
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "eps_refinement_$(lowercase(family)).png"), fig)
    end

    # 14.3, 14.4: Effect of poly_deg on ε sensitivity for laplacian
    for family in ("IMQ", "Gaussian")
        fig = Figure(size = (640, 460))
        ax = Axis(
            fig[1, 1]; xscale = log10, yscale = log10,
            xlabel = "ε", ylabel = "NRMSE",
            title = "$family Laplacian: poly_deg × ε (N=900)"
        )
        for (i, p) in enumerate(0:4)
            rs = sort(
                filter(r -> r.operator == "laplacian_polysweep" && r.family == family && r.poly_deg == p, eps_rows);
                by = r -> r.eps
            )
            isempty(rs) && continue
            eps = [r.eps for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(
                ax, eps, er; color = cmap[i], label = "p=$p",
                markersize = 8, linewidth = 1.8
            )
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "eps_polydeg_sweep_$(lowercase(family)).png"), fig)
    end
    return println("  ε-refinement plots ✓")
end

# ----------------------------------------------------------------------
# Work-precision (§15)
# ----------------------------------------------------------------------

# Curated (family, phs_order, poly_deg, label, color, linestyle, marker) for WP plots.
# PHS-only: shape-parameter bases need scaled ε to give a fair cost comparison — see
# the Shape-Parameter Bases page. Within-family: solid = matched poly_deg; dashed =
# higher-poly_deg overshoot showing the extra-accuracy-for-extra-cost tradeoff.
function _wp_configs()
    wong = Makie.wong_colors()
    c_phs3, c_phs5, c_phs7 = wong[1], wong[2], wong[3]
    return [
        ("PHS", 3, 2, "PHS3, p=2", c_phs3, :solid, :circle),
        ("PHS", 3, 4, "PHS3, p=4", c_phs3, :dash, :circle),
        ("PHS", 5, 3, "PHS5, p=3", c_phs5, :solid, :utriangle),
        ("PHS", 5, 5, "PHS5, p=5", c_phs5, :dash, :utriangle),
        ("PHS", 7, 4, "PHS7, p=4", c_phs7, :solid, :rect),
        ("PHS", 7, 6, "PHS7, p=6", c_phs7, :dash, :rect),
    ]
end

function _plot_wp_curves!(ax, rows_op, op_name, xfield)
    for (fam, ord, pdeg, label, color, ls, mk) in _wp_configs()
        is_unusable(op_name, fam, ord, pdeg) && continue
        rs = sort(
            filter(
                r -> r.family == fam && r.phs_order == ord && r.poly_deg == pdeg,
                rows_op,
            );
            by = r -> r.N,
        )
        isempty(rs) && continue
        x = getproperty.(rs, xfield)
        e = [r.nrmse for r in rs]
        scatterlines!(
            ax, x, e;
            color = color, linestyle = ls, marker = mk,
            markersize = 11, linewidth = 2.2,
            label = label,
        )
    end
    return
end

function plot_work_precision(wp_rows)
    for op in ("laplacian", "interpolation")
        rows_op = filter(r -> r.operator == op, wp_rows)
        for (time_col, suffix, title) in [
                (:build_time_s, "build", "weight-build time"),
                (:apply_time_s, "apply", "operator-apply time"),
            ]
            fig = Figure(size = (950, 520))
            ax = Axis(
                fig[1, 1]; xscale = log10, yscale = log10,
                xlabel = "$title (s)", ylabel = "NRMSE",
                title = "Work-precision ($op): $title",
            )
            _plot_wp_curves!(ax, rows_op, op, time_col)
            Legend(
                fig[1, 2], ax;
                framevisible = true, labelsize = 12,
                patchsize = (24, 16), rowgap = 4,
            )
            save(joinpath(PLOTS_DIR, "work_precision_$(op)_$(suffix).png"), fig)
        end
    end

    # Memory footprint vs NRMSE for laplacian (same style, just different x-field)
    fig = Figure(size = (950, 520))
    ax = Axis(
        fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "build memory (bytes)", ylabel = "NRMSE",
        title = "Work-precision: memory footprint (laplacian)",
    )
    rows_op = filter(r -> r.operator == "laplacian", wp_rows)
    _plot_wp_curves!(ax, rows_op, "laplacian", :build_bytes)
    Legend(
        fig[1, 2], ax;
        framevisible = true, labelsize = 12,
        patchsize = (24, 16), rowgap = 4,
    )
    save(joinpath(PLOTS_DIR, "work_precision_memory.png"), fig)

    return println("  work-precision plots ✓")
end

# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

function main(targets)
    targets = isempty(targets) ? ["h", "hseps", "p", "k", "eps", "wp"] : targets
    for t in targets
        if t == "h"
            rows = load_csv(joinpath(DATA_DIR, "h_refinement.csv"))
            make_h_plots(rows)
        elseif t == "hseps"
            h_rows = load_csv(joinpath(DATA_DIR, "h_refinement.csv"))
            scaled_rows = load_csv(
                joinpath(DATA_DIR, "h_refinement_scaled_eps.csv"),
            )
            make_eps_compare_plots(h_rows, scaled_rows)
        elseif t == "p"
            rows = load_csv(joinpath(DATA_DIR, "p_refinement.csv"))
            plot_p_refinement(rows)
        elseif t == "k"
            rows = load_csv(joinpath(DATA_DIR, "k_refinement.csv"))
            plot_k_refinement(rows)
        elseif t == "eps"
            rows = load_csv(joinpath(DATA_DIR, "eps_refinement.csv"))
            plot_eps_refinement(rows)
        elseif t == "wp"
            rows = load_csv(joinpath(DATA_DIR, "work_precision.csv"))
            plot_work_precision(rows)
        else
            error("unknown target: $t")
        end
    end
    return
end

main(ARGS)
