#=
Generate convergence-study plots from CSV artifacts written by `generate_data.jl`.

All plots use CairoMakie and are written to `docs/src/assets/convergence/plots/`.

Usage:
    julia --project=docs docs/src/assets/convergence/generate_plots.jl [targets...]

Targets (omit to run all): h p k eps wp 3d
=#

using CairoMakie
using DelimitedFiles
using Printf

const DATA_DIR = joinpath(@__DIR__, "data")
const PLOTS_DIR = joinpath(@__DIR__, "plots")
mkpath(PLOTS_DIR)

CairoMakie.activate!(type = "png")

# ----------------------------------------------------------------------
# Unusable (basis, poly_deg, operator) combinations
# ----------------------------------------------------------------------
# Empirically non-converging or numerically pathological — excluded from plots.
# See `scalar-operators.md` and `vector-operators.md` for prose notes.
const UNUSABLE = Set{Tuple{String, String, Int, Int}}([
    # Single 2nd-derivative operators: PHS1/p=1 numerically unstable; p below
    # matched gives no convergence; shape-parameter bases at p=0 don't converge.
    ("laplacian",     "PHS",      1, 1),
    ("laplacian",     "PHS",      3, 1),
    ("laplacian",     "IMQ",      0, 0),
    ("laplacian",     "Gaussian", 0, 0),

    ("partial2",      "PHS",      1, 1),
    ("partial2",      "PHS",      3, 1),
    ("partial2",      "IMQ",      0, 0),
    ("partial2",      "Gaussian", 0, 0),

    # Mixed partials: PHS3/p=2 and PHS5/p=2 plateau; IMQ/Gaussian p=2 also diverge.
    ("mixed_partial", "PHS",      1, 1),
    ("mixed_partial", "PHS",      3, 1),
    ("mixed_partial", "PHS",      3, 2),
    ("mixed_partial", "PHS",      5, 2),
    ("mixed_partial", "IMQ",      0, 0),
    ("mixed_partial", "IMQ",      0, 2),
    ("mixed_partial", "Gaussian", 0, 0),
    ("mixed_partial", "Gaussian", 0, 2),

    # Hessian inherits the mixed-partial restriction.
    ("hessian",       "PHS",      1, 1),
    ("hessian",       "PHS",      3, 1),
    ("hessian",       "PHS",      3, 2),
    ("hessian",       "PHS",      5, 2),
    ("hessian",       "IMQ",      0, 0),
    ("hessian",       "IMQ",      0, 2),
    ("hessian",       "Gaussian", 0, 0),
    ("hessian",       "Gaussian", 0, 2),

    # 3D Laplacian: PHS1 unusable for 2nd derivatives in 3D just as in 2D.
    ("laplacian3d",   "PHS",      1, 1),
])

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
    return lines!(ax, Ns, y; linestyle = :dash, color = (:gray, 0.6),
                  linewidth = 1.2, label = label)
end

# ----------------------------------------------------------------------
# Operator → plot metadata (title, ylabel, expected slopes for matched PHS)
# ----------------------------------------------------------------------

struct OpMeta
    name::String      # key in CSV operator column
    title::String     # plot title base
    ylabel::String
    rates_matched::NTuple{4, Int}  # expected rates for PHS1/3/5/7 at matched poly_deg
end

const OPS = [
    OpMeta("interpolation", "Interpolation",                     "NRMSE of u",          (2, 4, 6, 8)),
    OpMeta("laplacian",     "Laplacian (∇²)",                    "NRMSE of ∇²u",        (0, 2, 4, 6)),
    OpMeta("gradient",      "Gradient (∇)",                      "NRMSE of ∇u",         (1, 3, 5, 7)),
    OpMeta("partial1",      "First partial (∂/∂x₁)",             "NRMSE of ∂u/∂x",      (1, 3, 5, 7)),
    OpMeta("partial2",      "Second partial (∂²/∂x₁²)",          "NRMSE of ∂²u/∂x²",    (0, 2, 4, 6)),
    OpMeta("mixed_partial", "Mixed partial (∂²/∂x₁∂x₂)",         "NRMSE of ∂²u/∂x∂y",   (0, 2, 4, 6)),
    OpMeta("hessian",       "Hessian (Hu)",                      "Frobenius NRMSE",     (0, 2, 4, 6)),
    OpMeta("jacobian",      "Jacobian (Ju)",                     "Frobenius NRMSE",     (1, 3, 5, 7)),
    OpMeta("divergence",    "Divergence (∇·u)",                  "NRMSE of ∇·u",        (1, 3, 5, 7)),
    OpMeta("curl2d",        "Curl (∇×u, 2D)",                    "NRMSE of ∇×u",        (1, 3, 5, 7)),
]

# ----------------------------------------------------------------------
# h-refinement plots — three per operator
# ----------------------------------------------------------------------

function plot_h_phs_matched!(ax, rows_for_op, rates, op_name)
    configs = [(1, 1), (3, 2), (5, 3), (7, 4)]
    cmap = Makie.wong_colors()
    for (i, (ord, pdeg)) in enumerate(configs)
        is_unusable(op_name, "PHS", ord, pdeg) && continue
        rs = sorted_by_N(filter_rows(rows_for_op,
            r -> r.family == "PHS" && r.phs_order == ord && r.poly_deg == pdeg))
        isempty(rs) && continue
        Ns = [r.N for r in rs]
        errs = [r.nrmse for r in rs]
        scatterlines!(ax, Ns, errs; color = cmap[i],
                      label = "PHS$ord, p=$pdeg", markersize = 8, linewidth = 1.8)
    end
    # Reference slopes anchored near the cleanest (highest-rate) non-PHS1 curve
    ref_row = filter(r -> r.family == "PHS" && r.phs_order == 5 && r.poly_deg == 3, rows_for_op)
    if !isempty(ref_row)
        ref_row = sort(ref_row; by = r -> r.N)
        Ns_ref = [r.N for r in ref_row]
        Ns_line = range(minimum(Ns_ref), maximum(Ns_ref); length = 50) |> collect
        for (rate, lab) in zip(rates, ("O(h)", "O(h²)", "O(h⁴)", "O(h⁶)"))
            rate == 0 && continue
            ref_slope!(ax, Ns_line, rate;
                       anchor_N = Ns_ref[1], anchor_err = 10.0 ^ (-1 - rate/4),
                       label = "O(h^$rate)")
        end
    end
end

function plot_h_phs_polydeg_panel!(ax, rows_for_op, phs_order, poly_degs, title, op_name)
    cmap = Makie.wong_colors()
    for (i, p) in enumerate(poly_degs)
        is_unusable(op_name, "PHS", phs_order, p) && continue
        rs = sorted_by_N(filter_rows(rows_for_op,
            r -> r.family == "PHS" && r.phs_order == phs_order && r.poly_deg == p))
        isempty(rs) && continue
        Ns = [r.N for r in rs]
        errs = [r.nrmse for r in rs]
        scatterlines!(ax, Ns, errs; color = cmap[i],
                      label = "p=$p", markersize = 7, linewidth = 1.5)
    end
    ax.title = title
end

function plot_h_shape_panel!(ax, rows_for_op, family, poly_degs, title, op_name)
    cmap = Makie.wong_colors()
    for (i, p) in enumerate(poly_degs)
        is_unusable(op_name, family, 0, p) && continue
        rs = sorted_by_N(filter_rows(rows_for_op,
            r -> r.family == family && r.poly_deg == p))
        isempty(rs) && continue
        Ns = [r.N for r in rs]
        errs = [r.nrmse for r in rs]
        scatterlines!(ax, Ns, errs; color = cmap[i],
                      label = "p=$p", markersize = 7, linewidth = 1.5)
    end
    ax.title = title
end

function make_h_plots(h_rows)
    for op in OPS
        rows_op = filter(r -> r.operator == op.name, h_rows)
        isempty(rows_op) && continue

        # A — all PHS at matched poly_deg
        fig = Figure(size = (640, 460))
        ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "N", ylabel = op.ylabel,
                  title = "$(op.title): PHS orders at matched poly_deg")
        plot_h_phs_matched!(ax, rows_op, op.rates_matched, op.name)
        axislegend(ax; position = :lb, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "$(op.name)_phs_matched.png"), fig)

        # B — PHS poly_deg sweeps (3-panel)
        fig = Figure(size = (1200, 420))
        for (j, (ord, ps)) in enumerate([(3, 1:4), (5, 2:5), (7, 3:6)])
            ax = Axis(fig[1, j]; xscale = log10, yscale = log10,
                      xlabel = "N", ylabel = j == 1 ? op.ylabel : "")
            plot_h_phs_polydeg_panel!(ax, rows_op, ord, collect(ps),
                                      "PHS$ord, poly_deg sweep", op.name)
            axislegend(ax; position = :lb, framevisible = false, labelsize = 9)
        end
        Label(fig[0, :], "$(op.title): effect of polynomial degree for each PHS order",
              fontsize = 14)
        save(joinpath(PLOTS_DIR, "$(op.name)_phs_polydeg_sweep.png"), fig)

        # C — IMQ / Gaussian poly_deg sweep (2-panel)
        fig = Figure(size = (950, 420))
        for (j, family) in enumerate(("IMQ", "Gaussian"))
            ax = Axis(fig[1, j]; xscale = log10, yscale = log10,
                      xlabel = "N", ylabel = j == 1 ? op.ylabel : "")
            plot_h_shape_panel!(ax, rows_op, family, 0:4,
                                "$family (ε=1), poly_deg sweep", op.name)
            axislegend(ax; position = :lb, framevisible = false, labelsize = 9)
        end
        Label(fig[0, :], "$(op.title): shape-parameter basis poly_deg sweep",
              fontsize = 14)
        save(joinpath(PLOTS_DIR, "$(op.name)_imq_gaussian_polydeg.png"), fig)

        println("  h-plots: $(op.name) ✓")
    end
end

# ----------------------------------------------------------------------
# p-refinement (§12)
# ----------------------------------------------------------------------

function plot_p_refinement(p_rows)
    operators = ["interpolation", "laplacian", "gradient"]
    titles = Dict("interpolation" => "Interpolation",
                  "laplacian"     => "Laplacian (∇²)",
                  "gradient"      => "Gradient (∇)")
    cmap = Makie.wong_colors()
    families = [("PHS", 3, "PHS3"), ("PHS", 5, "PHS5"), ("PHS", 7, "PHS7"),
                ("IMQ", 0, "IMQ ε=1"), ("Gaussian", 0, "Gaussian ε=1")]

    for op in operators
        fig = Figure(size = (640, 460))
        ax = Axis(fig[1, 1]; yscale = log10,
                  xlabel = "polynomial degree", ylabel = "NRMSE",
                  title = "$(titles[op]): p-refinement at N=2500")
        rows_op = filter(r -> r.operator == op, p_rows)
        for (i, (family, ord, lab)) in enumerate(families)
            rs = sort(filter(r -> r.family == family && r.phs_order == ord, rows_op);
                      by = r -> r.poly_deg)
            isempty(rs) && continue
            pd = [r.poly_deg for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(ax, pd, er; color = cmap[i], label = lab,
                          markersize = 8, linewidth = 1.8)
        end
        axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "p_refinement_$(op).png"), fig)
    end

    # Cost plot: build_time_s vs poly_deg for laplacian
    fig = Figure(size = (640, 460))
    ax = Axis(fig[1, 1]; yscale = log10,
              xlabel = "polynomial degree", ylabel = "build time (s)",
              title = "Weight-build cost vs poly_deg (laplacian, N=2500)")
    rows_op = filter(r -> r.operator == "laplacian", p_rows)
    for (i, (family, ord, lab)) in enumerate(families)
        rs = sort(filter(r -> r.family == family && r.phs_order == ord, rows_op);
                  by = r -> r.poly_deg)
        isempty(rs) && continue
        pd = [r.poly_deg for r in rs]
        bt = [r.build_time_s for r in rs]
        scatterlines!(ax, pd, bt; color = Makie.wong_colors()[i], label = lab,
                      markersize = 8, linewidth = 1.8)
    end
    axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
    save(joinpath(PLOTS_DIR, "p_refinement_cost.png"), fig)
    println("  p-refinement plots ✓")
end

# ----------------------------------------------------------------------
# k-refinement (§13)
# ----------------------------------------------------------------------

function plot_k_refinement(k_rows)
    operators = [("interpolation", "Interpolation"),
                 ("laplacian", "Laplacian (∇²)"),
                 ("gradient",  "Gradient (∇)")]
    cmap = Makie.wong_colors()
    configs = [(3, 2, "PHS3, p=2"), (5, 3, "PHS5, p=3"), (7, 4, "PHS7, p=4")]

    for (op_key, op_title) in operators
        fig = Figure(size = (640, 460))
        ax = Axis(fig[1, 1]; yscale = log10,
                  xlabel = "stencil size k", ylabel = "NRMSE",
                  title = "$op_title: k-refinement at N=4900")
        rows_op = filter(r -> r.operator == op_key, k_rows)
        for (i, (ord, pdeg, lab)) in enumerate(configs)
            rs = sort(filter(r -> r.phs_order == ord && r.poly_deg == pdeg, rows_op);
                      by = r -> r.k)
            isempty(rs) && continue
            ks = [r.k for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(ax, ks, er; color = cmap[i], label = lab,
                          markersize = 8, linewidth = 1.8)
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "k_refinement_$(op_key).png"), fig)
    end
    println("  k-refinement plots ✓")
end

# ----------------------------------------------------------------------
# ε-refinement (§14)
# ----------------------------------------------------------------------

function plot_eps_refinement(eps_rows)
    cmap = Makie.wong_colors()

    # 14.1, 14.2: IMQ / Gaussian, three operators, poly_deg=2
    for family in ("IMQ", "Gaussian")
        fig = Figure(size = (640, 460))
        ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "ε (shape parameter)", ylabel = "NRMSE",
                  title = "$family shape-parameter sensitivity (poly_deg=2, N=900)")
        for (i, op) in enumerate(("interpolation", "laplacian", "gradient"))
            rs = sort(filter(r -> r.operator == op && r.family == family && r.poly_deg == 2, eps_rows);
                      by = r -> r.eps)
            isempty(rs) && continue
            eps = [r.eps for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(ax, eps, er; color = cmap[i], label = op,
                          markersize = 8, linewidth = 1.8)
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "eps_refinement_$(lowercase(family)).png"), fig)
    end

    # 14.3, 14.4: Effect of poly_deg on ε sensitivity for laplacian
    for family in ("IMQ", "Gaussian")
        fig = Figure(size = (640, 460))
        ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "ε", ylabel = "NRMSE",
                  title = "$family Laplacian: poly_deg × ε (N=900)")
        for (i, p) in enumerate(0:4)
            rs = sort(filter(r -> r.operator == "laplacian_polysweep" && r.family == family && r.poly_deg == p, eps_rows);
                      by = r -> r.eps)
            isempty(rs) && continue
            eps = [r.eps for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(ax, eps, er; color = cmap[i], label = "p=$p",
                          markersize = 8, linewidth = 1.8)
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "eps_polydeg_sweep_$(lowercase(family)).png"), fig)
    end
    println("  ε-refinement plots ✓")
end

# ----------------------------------------------------------------------
# Work-precision (§15)
# ----------------------------------------------------------------------

function plot_work_precision(wp_rows)
    # Build markers per (family, poly_deg). Within a family, poly_deg is a color;
    # across families, use different marker shapes.
    family_markers = Dict("PHS" => :circle, "IMQ" => :rect, "Gaussian" => :diamond)

    for op in ("laplacian", "interpolation")
        rows_op = filter(r -> r.operator == op, wp_rows)
        for (time_col, suffix, title) in [(:build_time_s, "build", "weight-build time"),
                                          (:apply_time_s, "apply", "operator-apply time")]
            fig = Figure(size = (700, 500))
            ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                      xlabel = "$title (s)", ylabel = "NRMSE",
                      title = "Work-precision ($op): $title")
            colors_phs = cgrad(:viridis, 10; categorical = true)
            for (i, (ord, pdeg)) in enumerate([(1,1),(3,2),(3,3),(3,4),(5,3),(5,4),(5,5),(7,4),(7,5),(7,6)])
                is_unusable(op, "PHS", ord, pdeg) && continue
                rs = filter(r -> r.family == "PHS" && r.phs_order == ord && r.poly_deg == pdeg, rows_op)
                isempty(rs) && continue
                rs = sort(rs; by = r -> r.N)
                t = getproperty.(rs, time_col)
                e = [r.nrmse for r in rs]
                scatterlines!(ax, t, e; color = colors_phs[i], marker = :circle,
                              markersize = 7, linewidth = 1.2,
                              label = "PHS$ord/p=$pdeg")
            end
            for family in ("IMQ", "Gaussian")
                for (i, p) in enumerate([1, 2, 3])
                    is_unusable(op, family, 0, p) && continue
                    rs = filter(r -> r.family == family && r.poly_deg == p, rows_op)
                    isempty(rs) && continue
                    rs = sort(rs; by = r -> r.N)
                    t = getproperty.(rs, time_col)
                    e = [r.nrmse for r in rs]
                    scatterlines!(ax, t, e; color = cgrad(:plasma, 4)[i + 1],
                                  marker = family_markers[family], markersize = 7,
                                  linewidth = 1.2, label = "$family/p=$p")
                end
            end
            axislegend(ax; position = :lt, framevisible = true, labelsize = 8,
                      nbanks = 2)
            save(joinpath(PLOTS_DIR, "work_precision_$(op)_$(suffix).png"), fig)
        end
    end

    # Memory footprint vs NRMSE for laplacian
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
              xlabel = "build memory (bytes)", ylabel = "NRMSE",
              title = "Work-precision: memory footprint (laplacian)")
    rows_op = filter(r -> r.operator == "laplacian", wp_rows)
    colors_phs = cgrad(:viridis, 10; categorical = true)
    for (i, (ord, pdeg)) in enumerate([(1,1),(3,2),(3,3),(3,4),(5,3),(5,4),(5,5),(7,4),(7,5),(7,6)])
        is_unusable("laplacian", "PHS", ord, pdeg) && continue
        rs = filter(r -> r.family == "PHS" && r.phs_order == ord && r.poly_deg == pdeg, rows_op)
        isempty(rs) && continue
        rs = sort(rs; by = r -> r.N)
        b = [r.build_bytes for r in rs]
        e = [r.nrmse for r in rs]
        scatterlines!(ax, b, e; color = colors_phs[i],
                      markersize = 7, linewidth = 1.2, label = "PHS$ord/p=$pdeg")
    end
    axislegend(ax; position = :lt, framevisible = true, labelsize = 8)
    save(joinpath(PLOTS_DIR, "work_precision_memory.png"), fig)

    println("  work-precision plots ✓")
end

# ----------------------------------------------------------------------
# 3D (§16)
# ----------------------------------------------------------------------

function plot_3d(rows_3d)
    cmap = Makie.wong_colors()

    # 3D h-ref: laplacian3d + gradient3d, PHS1/3/5/7 at matched poly_deg
    for (op, title) in [("laplacian3d", "3D Laplacian"),
                        ("gradient3d",  "3D Gradient")]
        fig = Figure(size = (640, 460))
        ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "N", ylabel = "NRMSE",
                  title = "$title: PHS orders at matched poly_deg")
        for (i, (ord, pdeg)) in enumerate([(1,1),(3,2),(5,3),(7,4)])
            is_unusable(op, "PHS", ord, pdeg) && continue
            rs = sort(filter(r -> r.operator == op && r.phs_order == ord && r.poly_deg == pdeg,
                             rows_3d); by = r -> r.N)
            isempty(rs) && continue
            Ns = [r.N for r in rs]
            er = [r.nrmse for r in rs]
            scatterlines!(ax, Ns, er; color = cmap[i], label = "PHS$ord, p=$pdeg",
                          markersize = 8, linewidth = 1.8)
        end
        axislegend(ax; position = :lb, framevisible = false, labelsize = 10)
        save(joinpath(PLOTS_DIR, "3d_$(op)_h_ref.png"), fig)
    end

    # 3D k-ref
    fig = Figure(size = (640, 460))
    ax = Axis(fig[1, 1]; yscale = log10,
              xlabel = "stencil size k", ylabel = "NRMSE",
              title = "3D Laplacian: k-refinement at N=4096")
    for (i, (ord, pdeg)) in enumerate([(3,2),(5,3),(7,4)])
        rs = sort(filter(r -> r.operator == "laplacian3d_kref" && r.phs_order == ord && r.poly_deg == pdeg,
                         rows_3d); by = r -> r.k)
        isempty(rs) && continue
        ks = [r.k for r in rs]
        er = [r.nrmse for r in rs]
        scatterlines!(ax, ks, er; color = cmap[i], label = "PHS$ord, p=$pdeg",
                      markersize = 8, linewidth = 1.8)
    end
    axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
    save(joinpath(PLOTS_DIR, "3d_laplacian_k_ref.png"), fig)
    println("  3D plots ✓")
end

# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

function main(targets)
    targets = isempty(targets) ? ["h", "p", "k", "eps", "wp", "3d"] : targets
    for t in targets
        if t == "h"
            rows = load_csv(joinpath(DATA_DIR, "h_refinement.csv"))
            make_h_plots(rows)
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
        elseif t == "3d"
            rows = load_csv(joinpath(DATA_DIR, "3d_refinement.csv"))
            plot_3d(rows)
        else
            error("unknown target: $t")
        end
    end
end

main(ARGS)
