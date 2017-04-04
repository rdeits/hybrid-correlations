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

State{T}(q::T, v::T, qlimb::T) = State{T}(q, v, qlimb)

immutable Result
    model::Model
    constraints::OrderedDict{Symbol, Vector{JuMP.ConstraintRef}}
    status::Symbol
    objective::JuMP.GenericQuadExpr{Float64, JuMP.Variable}
end

function run_opt(state::State, time, side, contact=nothing)
    m = Model(solver=GurobiSolver(OutputFlag=0))
    N = length(time)
    dt = time[2:end] - time[1:end-1]
    wall_pos = 1
    limb_length = 1.05
    force_max = 100
    v_max = 10
    vlimb_max = 10
    mass = 1

    @axis_variables(m, -wall_pos <= q[time] <= wall_pos)
    @axis_variables(m, v[time])
    @axis_variables(m, -wall_pos <= qlimb[time] <= wall_pos)
    @axis_variables(m, -force_max <= f[side, time] <= force_max)

    if contact === nothing
        @axis_variables(m, contact[side, time], category=:Bin)
    end

    a = (f[:left, :] .+ f[:right, :]) / mass

    constraints = OrderedDict(
        :limb_ub => @constraint(m, [i=1:N], qlimb[i] - q[i] <= limb_length),
        :limb_lb => @constraint(m, [i=1:N], -(qlimb[i] - q[i]) <= limb_length),
        :v_ub => @constraint(m, [i=1:N], v[i] <= v_max),
        :v_lb => @constraint(m, [i=1:N], -v[i] <= v_max),
        :vlimb_ub => @constraint(m, [i=1:N-1], qlimb[i + 1] - qlimb[i] <= vlimb_max * dt[i]),
        :vlimb_lb => @constraint(m, [i=1:N-1], -(qlimb[i + 1] - qlimb[i]) <= vlimb_max * dt[i]),
        :contact_at_distance_right => @constraint(m, [i=1:N-1], 1 - qlimb[i + 1] <= 2 * (1 - contact[:right, i])),
        :contact_at_distance_left => @constraint(m, [i=1:N-1], qlimb[i + 1] - (-1) <= 2 * (1 - contact[:left, i])),
        :normal_force_right => @constraint(m, [i=1:N], f[:right, i] <= 0),
        :force_without_contact_right => @constraint(m, [i=1:N], -f[:right, i] <= force_max * contact[:right, i]),
        :normal_force_left => @constraint(m, [i=1:N], -f[:left, i] <= 0),
        :force_without_contact_left => @constraint(m, [i=1:N], f[:left, i] <= force_max * contact[:left, i]),
        :dynamics_q => @constraint(m, [i=1:N-1], q[i + 1] == q[i] + dt[i] * v[i] + dt[i]^2/2 * a[i]),
        :dynamics_v => @constraint(m, [i=1:N-1], v[i + 1] == v[i] + dt[i] * a[i])
    )

    @constraints m begin
        q[1] == state.q
        v[1] == state.v
        qlimb[1] == state.qlimb
    end

    @expression m objective sum(f.^2) + 10 * sum(q.^2) + 100 * q[end]^2 + 10 * v[end]^2 + 1 * sum(diff(qlimb).^2)
    @objective m Min objective

    status = solve(m; suppress_warnings=true)
    Result(m, constraints, status, objective)
end

end
