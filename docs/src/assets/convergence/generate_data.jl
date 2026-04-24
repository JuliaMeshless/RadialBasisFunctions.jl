#=
Generate convergence-study data for the Convergence & Parameter Selection guide.

Runs h/p/k/ε-refinement sweeps plus work-precision benchmarks across every
operator × basis × poly_deg combination, writing per-topic CSVs under
`docs/src/assets/convergence/data/`. Idempotent: existing rows are preserved and
skipped on rerun.

Usage:
    julia --project=docs docs/src/assets/convergence/generate_data.jl [targets...]

Targets (omit to run all): h hseps p k eps wp 3d
=#

using RadialBasisFunctions
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using Random
using Printf
using BenchmarkTools

const DATA_DIR = joinpath(@__DIR__, "data")
mkpath(DATA_DIR)

# ----------------------------------------------------------------------
# Test functions and helpers
# ----------------------------------------------------------------------

function scattered_points(n_side; seed = 42)
    Random.seed!(seed)
    h = 1.0 / n_side
    return [
        SVector(
                clamp(h * (i - 0.5) + 0.2h * randn(), 0.001, 0.999),
                clamp(h * (j - 0.5) + 0.2h * randn(), 0.001, 0.999),
            ) for i in 1:n_side for j in 1:n_side
    ]
end

franke(x) = 0.75 * exp(-(9x[1] - 2)^2 / 4 - (9x[2] - 2)^2 / 4) +
    0.75 * exp(-(9x[1] + 1)^2 / 49 - (9x[2] + 1) / 10) +
    0.5 * exp(-(9x[1] - 7)^2 / 4 - (9x[2] - 3)^2 / 4) -
    0.2 * exp(-(9x[1] - 4)^2 - (9x[2] - 7)^2)

g(x) = 1 + sin(4x[1]) + cos(3x[1]) + sin(2x[2]) + sin(x[1] + x[2])
∂g_∂x1(x) = 4cos(4x[1]) - 3sin(3x[1]) + cos(x[1] + x[2])
∂g_∂x2(x) = 2cos(2x[2]) + cos(x[1] + x[2])
∂²g_∂x1²(x) = -16sin(4x[1]) - 9cos(3x[1]) - sin(x[1] + x[2])
∂²g_∂x2²(x) = -4sin(2x[2]) - sin(x[1] + x[2])
∂²g_∂x1∂x2(x) = -sin(x[1] + x[2])
∇²g(x) = ∂²g_∂x1²(x) + ∂²g_∂x2²(x)

# Vector field with all derivatives nonzero on [0,1]² (so NRMSE well-defined)
u1(x) = sin(π * x[1]) + 0.5cos(2π * x[2])
u2(x) = exp(x[1] * x[2])
∂u1_∂x1(x) = π * cos(π * x[1])
∂u1_∂x2(x) = -π * sin(2π * x[2])
∂u2_∂x1(x) = x[2] * exp(x[1] * x[2])
∂u2_∂x2(x) = x[1] * exp(x[1] * x[2])
div_u(x) = ∂u1_∂x1(x) + ∂u2_∂x2(x)
curl_u(x) = ∂u2_∂x1(x) - ∂u1_∂x2(x)

nrmse(computed, exact) = sqrt(sum(abs2, computed .- exact) / sum(abs2, exact))

function frobenius_nrmse(computed::AbstractArray, exact::AbstractArray)
    return sqrt(sum(abs2, computed .- exact) / sum(abs2, exact))
end

# ----------------------------------------------------------------------
# Basis factory
# ----------------------------------------------------------------------

function make_basis(family::Symbol, phs_order::Int, poly_deg::Int, eps::Float64)
    if family === :PHS
        return PHS(phs_order; poly_deg = poly_deg)
    elseif family === :IMQ
        return IMQ(eps; poly_deg = poly_deg)
    elseif family === :Gaussian
        return Gaussian(eps; poly_deg = poly_deg)
    else
        error("unknown basis family: $family")
    end
end

basis_label(family, phs_order, poly_deg, eps) =
    family === :PHS ? "PHS$(phs_order), p=$(poly_deg)" :
    family === :IMQ ? "IMQ ε=$(eps), p=$(poly_deg)" :
    "Gaussian ε=$(eps), p=$(poly_deg)"

# ----------------------------------------------------------------------
# Per-operator error evaluation
# ----------------------------------------------------------------------

# Returns NRMSE of computed operator output against analytic truth.
# `op_kind` controls what's computed and what the exact answer is.
function compute_error(op_kind::Symbol, pts, basis; k = nothing)
    kw = k === nothing ? (; basis) : (; basis, k)

    if op_kind === :interpolation
        Random.seed!(99)
        eval_pts = [SVector{2}(rand(2) .* 0.98 .+ 0.01) for _ in 1:500]
        vals = franke.(pts)
        interp = Interpolator(pts, vals, basis)
        got = interp.(eval_pts)
        exact = franke.(eval_pts)
        return nrmse(got, exact)

    elseif op_kind === :laplacian
        op = laplacian(pts; kw...)
        got = op(g.(pts))
        return nrmse(got, ∇²g.(pts))

    elseif op_kind === :gradient
        op = gradient(pts; kw...)
        got = op(g.(pts))  # N × 2
        exact = hcat(∂g_∂x1.(pts), ∂g_∂x2.(pts))
        return frobenius_nrmse(got, exact)

    elseif op_kind === :partial1
        op = partial(pts, 1, 1; kw...)
        got = op(g.(pts))
        return nrmse(got, ∂g_∂x1.(pts))

    elseif op_kind === :partial2
        op = partial(pts, 2, 1; kw...)
        got = op(g.(pts))
        return nrmse(got, ∂²g_∂x1².(pts))

    elseif op_kind === :mixed_partial
        op = mixed_partial(pts, 1, 2; kw...)
        got = op(g.(pts))
        return nrmse(got, ∂²g_∂x1∂x2.(pts))

    elseif op_kind === :hessian
        op = hessian(pts; kw...)
        got = op(g.(pts))  # N × 2 × 2
        exact = similar(got)
        for (i, x) in enumerate(pts)
            exact[i, 1, 1] = ∂²g_∂x1²(x)
            exact[i, 1, 2] = ∂²g_∂x1∂x2(x)
            exact[i, 2, 1] = ∂²g_∂x1∂x2(x)
            exact[i, 2, 2] = ∂²g_∂x2²(x)
        end
        return frobenius_nrmse(got, exact)

    elseif op_kind === :jacobian
        op = jacobian(pts; kw...)
        u_mat = hcat(u1.(pts), u2.(pts))
        got = op(u_mat)  # N × 2 × 2
        exact = similar(got)
        for (i, x) in enumerate(pts)
            exact[i, 1, 1] = ∂u1_∂x1(x)
            exact[i, 1, 2] = ∂u1_∂x2(x)
            exact[i, 2, 1] = ∂u2_∂x1(x)
            exact[i, 2, 2] = ∂u2_∂x2(x)
        end
        return frobenius_nrmse(got, exact)

    elseif op_kind === :divergence
        op = divergence(pts; kw...)
        u_mat = hcat(u1.(pts), u2.(pts))
        got = op(u_mat)
        return nrmse(got, div_u.(pts))

    elseif op_kind === :curl2d
        op = curl(pts; kw...)
        u_mat = hcat(u1.(pts), u2.(pts))
        got = op(u_mat)
        return nrmse(got, curl_u.(pts))

    else
        error("unknown op_kind: $op_kind")
    end
end

# ----------------------------------------------------------------------
# CSV I/O with idempotency
# ----------------------------------------------------------------------

struct CSVStore
    path::String
    header::Vector{String}
    seen::Set{Tuple}
    key_cols::Vector{Int}  # which columns form the uniqueness key
end

function CSVStore(path, header, key_cols_names)
    key_cols = [findfirst(==(c), header) for c in key_cols_names]
    seen = Set{Tuple}()
    if isfile(path)
        data, hdr = readdlm(path, ','; header = true)
        @assert vec(hdr) == header "header mismatch in $path"
        for row in eachrow(data)
            push!(seen, Tuple(row[i] for i in key_cols))
        end
    else
        open(path, "w") do io
            println(io, join(header, ","))
        end
    end
    return CSVStore(path, header, seen, key_cols)
end

function has_row(store::CSVStore, key_values)
    return Tuple(key_values) in store.seen
end

function append_row!(store::CSVStore, values)
    open(store.path, "a") do io
        println(io, join(values, ","))
    end
    key = Tuple(values[i] for i in store.key_cols)
    push!(store.seen, key)
    return nothing
end

# ----------------------------------------------------------------------
# Config: sweep sets
# ----------------------------------------------------------------------

const N_SIDES = [10, 15, 20, 30, 45, 70]

# Configurations sweeping PHS orders at matched poly_deg
const PHS_MATCHED = [(1, 1), (3, 2), (5, 3), (7, 4)]

# Per-PHS-order poly_deg sweep (order, [poly_degs])
const PHS_POLYDEG_SWEEPS = Dict(
    3 => [1, 2, 3, 4],
    5 => [2, 3, 4, 5],
    7 => [3, 4, 5, 6],
)

# IMQ/Gaussian poly_deg sweep at fixed ε=1
const SHAPE_POLYDEGS = [0, 1, 2, 3, 4]
const DEFAULT_EPS = 1.0

# Operators to sweep for h-refinement (§2–§11)
const OPERATORS_2D = [
    :interpolation, :laplacian, :gradient,
    :partial1, :partial2, :mixed_partial,
    :hessian, :jacobian, :divergence, :curl2d,
]

# ----------------------------------------------------------------------
# Generate configurations (family, phs_order, poly_deg, eps) per operator
# ----------------------------------------------------------------------

function all_configs_2d()
    cfgs = Tuple{Symbol, Int, Int, Float64}[]
    for (ord, pdeg) in PHS_MATCHED
        push!(cfgs, (:PHS, ord, pdeg, 0.0))
    end
    for (ord, pdegs) in PHS_POLYDEG_SWEEPS
        for p in pdegs
            push!(cfgs, (:PHS, ord, p, 0.0))
        end
    end
    for p in SHAPE_POLYDEGS
        push!(cfgs, (:IMQ, 0, p, DEFAULT_EPS))
        push!(cfgs, (:Gaussian, 0, p, DEFAULT_EPS))
    end
    return unique(cfgs)
end

# ----------------------------------------------------------------------
# h-refinement sweep
# ----------------------------------------------------------------------

function sweep_h_refinement()
    header = ["operator", "family", "phs_order", "poly_deg", "eps", "N", "nrmse"]
    store = CSVStore(
        joinpath(DATA_DIR, "h_refinement.csv"), header,
        ["operator", "family", "phs_order", "poly_deg", "eps", "N"]
    )

    cfgs = all_configs_2d()
    total = length(OPERATORS_2D) * length(cfgs) * length(N_SIDES)
    done = 0

    for op in OPERATORS_2D, (family, ord, pdeg, eps) in cfgs, n_side in N_SIDES
        done += 1
        N = n_side^2
        key = (String(op), String(family), ord, pdeg, eps, N)
        if has_row(store, key)
            continue
        end
        basis = make_basis(family, ord, pdeg, eps)
        pts = scattered_points(n_side)
        err = try
            compute_error(op, pts, basis)
        catch e
            @warn "failed" op family ord pdeg eps N exception = (e, catch_backtrace())
            NaN
        end
        append_row!(store, (String(op), String(family), ord, pdeg, eps, N, err))
        if done % 20 == 0
            @printf("[h-ref] %d / %d\n", done, total)
        end
    end
    return println("[h-ref] done")
end

# ----------------------------------------------------------------------
# Scaled-ε h-refinement sweep (IMQ / Gaussian)
# ----------------------------------------------------------------------
#
# With fixed ε and shrinking h ~ 1/n_side, the non-dimensional ε·h collapses
# as N grows; shape-parameter bases go "flat" over each stencil and the
# collocation matrix becomes ill-conditioned (RBF uncertainty principle).
#
# Fix: hold ε·h constant across the sweep, i.e. ε = c/h = c · n_side.
# The constants `c` below are calibrated from the fixed-N ε-refinement data
# in `data/eps_refinement.csv` at N=900 (n_side=30), using the NRMSE minimum
# for each operator (averaged across IMQ and Gaussian; picked so the scaled
# ε lands in the sweet spot at representative N).
#
#   operator       ε_opt (N=900)   c = ε_opt · h = ε_opt / 30
#   interpolation  3 – 6           ≈ 0.10 – 0.21  → use 0.15
#   laplacian      0.5 – 1.0       ≈ 0.017 – 0.033 → use 0.03
#   mixed_partial  (no sweep; use laplacian's c — same operator order)

const EPS_SCALE_C = Dict(
    :interpolation => 0.15,
    :laplacian => 0.03,
    :mixed_partial => 0.03,
)

function sweep_h_refinement_scaled_eps()
    header = [
        "operator", "family", "poly_deg", "eps_scale_c", "N", "eps", "nrmse",
    ]
    store = CSVStore(
        joinpath(DATA_DIR, "h_refinement_scaled_eps.csv"), header,
        ["operator", "family", "poly_deg", "eps_scale_c", "N"],
    )

    operators = [:interpolation, :laplacian, :mixed_partial]
    families = [:IMQ, :Gaussian]
    poly_degs = [2, 3]

    total = length(operators) * length(families) * length(poly_degs) * length(N_SIDES)
    done = 0

    for op in operators, family in families, pdeg in poly_degs, n_side in N_SIDES
        done += 1
        N = n_side^2
        c = EPS_SCALE_C[op]
        eps = c * n_side  # ε = c / h, h = 1 / n_side
        key = (String(op), String(family), pdeg, c, N)
        has_row(store, key) && continue
        basis = make_basis(family, 0, pdeg, eps)
        pts = scattered_points(n_side)
        err = try
            compute_error(op, pts, basis)
        catch e
            @warn "failed" op family pdeg eps N exception = (e, catch_backtrace())
            NaN
        end
        append_row!(store, (String(op), String(family), pdeg, c, N, eps, err))
        if done % 10 == 0
            @printf("[h-ref scaled-ε] %d / %d\n", done, total)
        end
    end
    return println("[h-ref scaled-ε] done")
end

# ----------------------------------------------------------------------
# p-refinement sweep (fixed N, vary poly_deg)
# ----------------------------------------------------------------------

function sweep_p_refinement()
    header = ["operator", "family", "phs_order", "poly_deg", "eps", "N", "nrmse", "build_time_s"]
    store = CSVStore(
        joinpath(DATA_DIR, "p_refinement.csv"), header,
        ["operator", "family", "phs_order", "poly_deg", "eps", "N"]
    )

    N_fixed = 2500  # n_side = 50
    n_side = 50
    pts = scattered_points(n_side)
    operators = [:interpolation, :laplacian, :gradient]
    poly_degs = 0:6
    families = [(:PHS, 3), (:PHS, 5), (:PHS, 7), (:IMQ, 0), (:Gaussian, 0)]

    total = length(operators) * length(poly_degs) * length(families)
    done = 0
    for op in operators, p in poly_degs, (family, ord) in families
        done += 1
        # IMQ/Gaussian use ε=1
        eps = family === :PHS ? 0.0 : DEFAULT_EPS
        key = (String(op), String(family), ord, p, eps, N_fixed)
        has_row(store, key) && continue
        basis = make_basis(family, ord, p, eps)
        err = try
            compute_error(op, pts, basis)
        catch e
            @warn "failed" op family ord p exception = (e, catch_backtrace())
            NaN
        end
        bt = try
            t = @belapsed ($compute_error)($op, $pts, $basis) samples = 3 evals = 1
            t
        catch
            NaN
        end
        append_row!(store, (String(op), String(family), ord, p, eps, N_fixed, err, bt))
        done % 10 == 0 && @printf("[p-ref] %d / %d\n", done, total)
    end
    return println("[p-ref] done")
end

# ----------------------------------------------------------------------
# k-refinement sweep (fixed N, poly_deg; vary stencil size k)
# ----------------------------------------------------------------------

function sweep_k_refinement()
    header = ["operator", "family", "phs_order", "poly_deg", "eps", "N", "k", "nrmse"]
    store = CSVStore(
        joinpath(DATA_DIR, "k_refinement.csv"), header,
        ["operator", "family", "phs_order", "poly_deg", "eps", "N", "k"]
    )

    n_side = 70
    N_fixed = n_side^2
    pts = scattered_points(n_side)

    configs = [
        (:PHS, 3, 3, 0.0),
        (:PHS, 5, 3, 0.0),
        (:PHS, 7, 3, 0.0),
    ]
    operators = [:interpolation, :laplacian, :gradient]
    k_range = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

    total = length(operators) * length(configs) * length(k_range)
    done = 0
    for op in operators, (family, ord, pdeg, eps) in configs, k in k_range
        done += 1
        key = (String(op), String(family), ord, pdeg, eps, N_fixed, k)
        has_row(store, key) && continue
        basis = make_basis(family, ord, pdeg, eps)
        err = try
            if op === :interpolation
                # Interpolator has no stencil concept — skip
                NaN
            else
                compute_error(op, pts, basis; k = k)
            end
        catch e
            @warn "failed" op family ord pdeg k exception = (e, catch_backtrace())
            NaN
        end
        append_row!(store, (String(op), String(family), ord, pdeg, eps, N_fixed, k, err))
        done % 10 == 0 && @printf("[k-ref] %d / %d\n", done, total)
    end
    return println("[k-ref] done")
end

# ----------------------------------------------------------------------
# ε-refinement sweep (IMQ and Gaussian)
# ----------------------------------------------------------------------

function sweep_eps_refinement()
    header = ["operator", "family", "poly_deg", "eps", "N", "nrmse", "cond_est"]
    store = CSVStore(
        joinpath(DATA_DIR, "eps_refinement.csv"), header,
        ["operator", "family", "poly_deg", "eps"]
    )

    n_side = 30
    N_fixed = n_side^2
    pts = scattered_points(n_side)

    # ε range: log-uniform from 0.1 to 20
    eps_range = 10.0 .^ range(-1, stop = 1.3, length = 15)
    families = [:IMQ, :Gaussian]
    operators = [:interpolation, :laplacian, :gradient]
    poly_deg_single = 2  # 14.1, 14.2
    poly_degs_sweep = [0, 1, 2, 3, 4]  # 14.3, 14.4 (for laplacian)

    # 14.1, 14.2: poly_deg=2 fixed, sweep ε across operators
    for op in operators, family in families, eps in eps_range
        key = (String(op), String(family), poly_deg_single, eps)
        has_row(store, key) && continue
        basis = make_basis(family, 0, poly_deg_single, eps)
        err = try
            compute_error(op, pts, basis)
        catch
            NaN
        end
        append_row!(store, (String(op), String(family), poly_deg_single, eps, N_fixed, err, NaN))
    end

    # 14.3, 14.4: Laplacian, vary poly_deg and ε
    for family in families, p in poly_degs_sweep, eps in eps_range
        key = ("laplacian_polysweep", String(family), p, eps)
        has_row(store, key) && continue
        basis = make_basis(family, 0, p, eps)
        err = try
            compute_error(:laplacian, pts, basis)
        catch
            NaN
        end
        append_row!(store, ("laplacian_polysweep", String(family), p, eps, N_fixed, err, NaN))
    end
    return println("[ε-ref] done")
end

# ----------------------------------------------------------------------
# Work-precision benchmarks (§15)
# ----------------------------------------------------------------------

function sweep_work_precision()
    header = [
        "operator", "family", "phs_order", "poly_deg", "eps", "N",
        "nrmse", "build_time_s", "apply_time_s", "build_bytes",
    ]
    store = CSVStore(
        joinpath(DATA_DIR, "work_precision.csv"), header,
        ["operator", "family", "phs_order", "poly_deg", "eps", "N"]
    )

    cfgs = all_configs_2d()
    operators = [:laplacian, :interpolation]
    n_sides_wp = [15, 30, 45, 70]

    total = length(operators) * length(cfgs) * length(n_sides_wp)
    done = 0
    for op in operators, (family, ord, pdeg, eps) in cfgs, n_side in n_sides_wp
        done += 1
        N = n_side^2
        key = (String(op), String(family), ord, pdeg, eps, N)
        if has_row(store, key)
            continue
        end
        pts = scattered_points(n_side)
        basis = make_basis(family, ord, pdeg, eps)
        err = NaN
        bt = NaN
        at = NaN
        bbytes = NaN
        try
            if op === :laplacian
                # timed build
                b = @benchmarkable laplacian($pts; basis = $basis) samples = 3 evals = 1 seconds = 30
                tune!(b)
                res = run(b)
                bt = median(res.times) / 1.0e9
                bbytes = res.memory
                # build one instance for apply timing and error
                opx = laplacian(pts; basis = basis)
                vals = g.(pts)
                # force first application to warmup cache
                _ = opx(vals)
                a = @benchmarkable $opx($vals) samples = 5 evals = 1 seconds = 30
                tune!(a)
                ar = run(a)
                at = median(ar.times) / 1.0e9
                err = nrmse(opx(vals), ∇²g.(pts))
            elseif op === :interpolation
                vals = franke.(pts)
                b = @benchmarkable Interpolator($pts, $vals, $basis) samples = 3 evals = 1 seconds = 30
                tune!(b)
                res = run(b)
                bt = median(res.times) / 1.0e9
                bbytes = res.memory
                interp = Interpolator(pts, vals, basis)
                Random.seed!(99)
                eval_pts = [SVector{2}(rand(2) .* 0.98 .+ 0.01) for _ in 1:500]
                _ = interp.(eval_pts)
                a = @benchmarkable $interp.($eval_pts) samples = 5 evals = 1 seconds = 30
                tune!(a)
                ar = run(a)
                at = median(ar.times) / 1.0e9
                err = nrmse(interp.(eval_pts), franke.(eval_pts))
            end
        catch e
            @warn "work-precision failed" op family ord pdeg eps N exception = (e,)
        end
        append_row!(store, (String(op), String(family), ord, pdeg, eps, N, err, bt, at, bbytes))
        @printf("[wp] %d / %d\n", done, total)
    end
    return println("[wp] done")
end

# ----------------------------------------------------------------------
# Machine specs sidecar
# ----------------------------------------------------------------------

function write_machine_info()
    path = joinpath(DATA_DIR, "machine.txt")
    return open(path, "w") do io
        println(io, "Julia version: ", VERSION)
        println(io, "CPU: ", Sys.cpu_info()[1].model)
        println(io, "Threads: ", Threads.nthreads())
        println(io, "OS: ", Sys.KERNEL, " ", Sys.MACHINE)
        println(io, "RadialBasisFunctions: ", pkgversion(RadialBasisFunctions))
        println(io, "Generated: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    end
end

using Dates

# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

function main(targets)
    targets = isempty(targets) ? ["h", "hseps", "p", "k", "eps", "wp"] : targets
    for t in targets
        t == "h" ? sweep_h_refinement() :
            t == "hseps" ? sweep_h_refinement_scaled_eps() :
            t == "p" ? sweep_p_refinement() :
            t == "k" ? sweep_k_refinement() :
            t == "eps" ? sweep_eps_refinement() :
            t == "wp" ? sweep_work_precision() :
            error("unknown target: $t")
    end
    return write_machine_info()
end

main(ARGS)
