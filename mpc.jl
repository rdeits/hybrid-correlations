module MPC

using JuMP
using Gurobi: GurobiSolver
using DataStructures: OrderedDict
using AxisArrays

include("axisvars.jl")
using .AxisVars: @axis_variables

immutable State{T}
    q::T
    v::T
    qlimb::T
end

typealias VarVector AxisArray{Variable, 1, Vector{Variable}, Tuple{Axis{:time, LinSpace{Float64}}}}

immutable MPCModel
    m::Model
    q::VarVector
    v::VarVector
    qlimb::VarVector
    contact::AxisArray{Variable, 2, Matrix{Variable}, Tuple{Axis{:side, Vector{Symbol}}, Axis{:time, LinSpace{Float64}}}}
    slack::Vector{JuMP.Variable}
end

contactvars(m::MPCModel) = m.contact

type Result
    model::MPCModel
    status::Symbol
end

immutable CTLinearSytstem{T}
    A::Matrix{T}
    B::Matrix{T}
end

immutable DTLinearSystem{T}
    A::Matrix{T}
    B::Matrix{T}
    Δt::T
end

"""
Convert continuous-time linear system to discrete time, assuming
a zero-order hold on the inputs.

Based on Python code originally written by Tobia Marcucci.
"""
function discretize{T}(s::CTLinearSytstem{T}, Δt)
    nx = size(s.A, 1)
    nu = size(s.B, 2)
    c = zeros(nx + nu, nx + nu)
    c[1:nx, 1:nx] .= s.A
    c[1:nx, nx+1:end] .= s.B
    d = expm(c * Δt)
    A = d[1:nx, 1:nx]
    B = d[1:nx, nx+1:end]
    DTLinearSystem{T}(A, B, Δt)
end

function create_model(sys::DTLinearSystem, time, side)
    m = Model(solver=GurobiSolver(OutputFlag=0))
    N = length(time)
    Δt = time[2:end] - time[1:end-1]
    @assert all(isapprox.(Δt, sys.Δt))
    wall_pos = 1
    limb_length = 1.05
    force_max = 100
    v_max = 10
    vlimb_max = 10

    @axis_variables(m, -wall_pos <= q[time] <= wall_pos)
    @axis_variables(m, v[time])
    @axis_variables(m, -wall_pos <= qlimb[time] <= wall_pos)
    @axis_variables(m, -force_max <= f[side, time] <= force_max)

    @axis_variables(m, contact[side, time], category=:Bin)

    u = f[:left, :] .+ f[:right, :]

    @constraint(m, single_contact[i=1:N], sum(contact[Axis{:side}(:), Axis{:time}(i)]) <= 1)
    @constraint(m, limb_ub[i=1:N], qlimb[i] - q[i] <= limb_length)
    @constraint(m, limb_lb[i=1:N], -(qlimb[i] - q[i]) <= limb_length)
    @constraint(m, v_ub[i=1:N], v[i] <= v_max)
    @constraint(m, v_lb[i=1:N], -v[i] <= v_max)
    @constraint(m, vlimb_ub[i=1:N-1], qlimb[i + 1] - qlimb[i] <= vlimb_max * Δt[i])
    @constraint(m, vlimb_lb[i=1:N-1], -(qlimb[i + 1] - qlimb[i]) <= vlimb_max * Δt[i])
    @constraint(m, contact_at_distance_right[i=1:N-1], 1 - qlimb[i + 1] <= 2 * (1 - contact[:right, i]))
    @constraint(m, contact_at_distance_left[i=1:N-1], qlimb[i + 1] - (-1) <= 2 * (1 - contact[:left, i]))
    @constraint(m, normal_force_right[i=1:N], f[:right, i] <= 0)
    @constraint(m, force_without_contact_right[i=1:N], -f[:right, i] <= force_max * contact[:right, i])
    @constraint(m, normal_force_left[i=1:N], -f[:left, i] <= 0)
    @constraint(m, force_without_contact_left[i=1:N], f[:left, i] <= force_max * contact[:left, i])
    @constraint(m, dynamics_q[i=1:N-1], q[i + 1] == (sys.A[1, :]' * [q[i], v[i]] .+ sys.B[1, :]' * [u[i]])[1])
    @constraint(m, dynamics_v[i=1:N-1], v[i + 1] == (sys.A[2, :]' * [q[i], v[i]] + sys.B[2, :]' * [u[i]])[1])

    @objective m Min sum(f.^2) + 10 * sum(q.^2) + 100 * q[end]^2 + 10 * v[end]^2 + 1 * sum(diff(qlimb).^2)

    slack = relax!(m)

    MPCModel(m, q, v, qlimb, contact, slack)
end

function dummy_model(time, side)
    m = Model(solver=GurobiSolver(OutputFlag=0))
    N = length(time)

    @axis_variables(m, q[time])
    @axis_variables(m, v[time])
    @axis_variables(m, qlimb[time])
    @axis_variables(m, f[side, time])

    @axis_variables(m, contact[side, time], category=:Bin)

    rows = 14 * N
    vars = vcat(vec(q), vec(v), vec(qlimb), vec(f), vec(contact[:right, :]), vec(contact[:left, :]))
    nvars = length(vars)
    density = 2 / nvars
    for i in 1:rows
        a = zeros(nvars)
        for j in 1:rand(1:3)
            a[rand(1:nvars)] = randn()
        end
        @constraint(m, (a' * vars)[1] <= randn())
    end

    @objective m Min sum(f.^2) + 10 * sum(q.^2) + 100 * q[end]^2 + 10 * v[end]^2 + 1 * sum(diff(qlimb).^2)

    slack = relax!(m)

    MPCModel(m, OrderedDict{Symbol, Vector{JuMP.ConstraintRef}}(), q, v, qlimb, contact, slack)
end

function solve!(model::MPCModel, state::State; contact_sequence=nothing, relax=false)
    if contact_sequence !== nothing
        for I in eachindex(model.contact)
            JuMP.fix(model.contact[I], contact_sequence[I])
        end
    end
    JuMP.fix(model.q[1], state.q)
    JuMP.fix(model.v[1], state.v)
    JuMP.fix(model.qlimb[1], state.qlimb)

    if relax
        setlowerbound.(model.slack, 0)
        setupperbound.(model.slack, Inf)
    else
        setlowerbound.(model.slack, 0)
        setupperbound.(model.slack, 0)
    end

    status = solve(model.m; suppress_warnings=true)
    Result(model, status)
end

function relax!(m::Model)
    lb, ub = JuMP.constraintbounds(m)
    nconstr = length(lb)
    y = JuMP.Variable[]
    const M = typeof(m)
    const C = JuMP.GenericRangeConstraint{JuMP.GenericAffExpr{Float64, JuMP.Variable}}
    
    function addslack(i, coeff)
        push!(y, @variable(m, objective=0, inconstraints=[JuMP.ConstraintRef{M, C}(m, i)], coefficients=[coeff], basename="y"))
    end
    
    for i in 1:nconstr
        if lb[i] == -Inf
            addslack(i, -1.0)
        elseif ub[i] == Inf
            addslack(i, 1.0)
        end
    end
    @objective m Min getobjective(m) + sum(y.^2)
    y
end

function relax!(r::Result)
    relax!(r.model)
    r.status = solve(r.model; suppress_warnings=true)
end

end
