module MPC

using JuMP
using Gurobi: GurobiSolver
using DataStructures: OrderedDict

include("axisvars.jl")
using .AxisVars: @axis_variables

immutable State{T}
    q::T
    v::T
    qlimb::T
end

type Result
    model::Model
    constraints::OrderedDict{Symbol, Vector{JuMP.ConstraintRef}}
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

function run_opt(sys::DTLinearSystem, state::State, time, side, contact_sequence=nothing)
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

    if contact_sequence === nothing
        @axis_variables(m, contact[side, time], category=:Bin)
    else
        contact = contact_sequence
    end

    u = f[:left, :] .+ f[:right, :]

    constraints = OrderedDict(
        :limb_ub => @constraint(m, [i=1:N], qlimb[i] - q[i] <= limb_length),
        :limb_lb => @constraint(m, [i=1:N], -(qlimb[i] - q[i]) <= limb_length),
        :v_ub => @constraint(m, [i=1:N], v[i] <= v_max),
        :v_lb => @constraint(m, [i=1:N], -v[i] <= v_max),
        :vlimb_ub => @constraint(m, [i=1:N-1], qlimb[i + 1] - qlimb[i] <= vlimb_max * Δt[i]),
        :vlimb_lb => @constraint(m, [i=1:N-1], -(qlimb[i + 1] - qlimb[i]) <= vlimb_max * Δt[i]),
        :contact_at_distance_right => @constraint(m, [i=1:N-1], 1 - qlimb[i + 1] <= 2 * (1 - contact[:right, i])),
        :contact_at_distance_left => @constraint(m, [i=1:N-1], qlimb[i + 1] - (-1) <= 2 * (1 - contact[:left, i])),
        :normal_force_right => @constraint(m, [i=1:N], f[:right, i] <= 0),
        :force_without_contact_right => @constraint(m, [i=1:N], -f[:right, i] <= force_max * contact[:right, i]),
        :normal_force_left => @constraint(m, [i=1:N], -f[:left, i] <= 0),
        :force_without_contact_left => @constraint(m, [i=1:N], f[:left, i] <= force_max * contact[:left, i]),
        :dynamics_q => @constraint(m, [i=1:N-1], q[i + 1] == (sys.A[1, :]' * [q[i], v[i]] .+ sys.B[1, :]' * [u[i]])[1]),
        :dynamics_v => @constraint(m, [i=1:N-1], v[i + 1] == (sys.A[2, :]' * [q[i], v[i]] + sys.B[2, :]' * [u[i]])[1])
    )

    @constraints m begin
        q[1] == state.q
        v[1] == state.v
        qlimb[1] == state.qlimb
    end

    @objective m Min sum(f.^2) + 10 * sum(q.^2) + 100 * q[end]^2 + 10 * v[end]^2 + 1 * sum(diff(qlimb).^2)

    status = solve(m; suppress_warnings=true)
    Result(m, constraints, status)
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
    m
end

function relax!(r::Result)
    relax!(r.model)
    r.status = solve(r.model; suppress_warnings=true)
end

end
