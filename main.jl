module c

using AxisArrays
using JuMP
using DataStructures: OrderedDict

include("mpc.jl")

# immutable Record
#     state::MPC.State{Float64}
#     contact::AxisArray{Bool, 2, Array{Bool, 2}, Tuple{Axis{:side, Array{Symbol, 1}}, Axis{:time, LinSpace{Float64}}}}
#     duals::OrderedDict{Symbol, Vector{Float64}}
#     cost::Float64
#     new_costs::AxisArray{Float64, 2, Array{Float64, 2}, Tuple{Axis{:side, Array{Symbol, 1}}, Axis{:time, LinSpace{Float64}}}}
# end

immutable Sample
    dual::Float64
    Δcost::Float64
end

function collect_data(sys, numsamples)
    dt = sys.Δt
    N = 10
    time = Axis{:time}(linspace(0, (N - 1) * dt, N))
    side = Axis{:side}([:left, :right])
    # records = Record[]

    contact = AxisArray(zeros(Bool, 2, N), side, time)
    state = MPC.State(0., 0., 0.)
    result = MPC.run_opt(sys, state, time, side, contact)
    samples = AxisArray(Array{Sample}(numsamples, length(result.constraints), N - 1, 2, N),
                        Axis{:sample}(1:numsamples),
                        Axis{:constraint}(collect(keys(result.constraints))),
                        Axis{:constraint_t}(1:N-1),
                        Axis{:contact_side}([:left, :right]),
                        Axis{:contact_t}(1:N))

    sample_index = 1
    while sample_index <= numsamples
        contact = AxisArray(rand(Bool, 2, N), side, time)
        for j in 1:N
            if all(contact[:, j])
                contact[rand(1:2), j] = false
            end
        end
        q0 = 2 * rand() - 1
        v0 = 4 * rand() - 2
        qlimb0 = 2 * rand() - 1
        state = MPC.State(q0, v0, qlimb0)
        result = MPC.run_opt(sys, state, time, side, contact)
        if result.status == :Optimal
            # duals = OrderedDict((key, getdual(value)) for (key, value) in result.constraints)
            newcosts = AxisArray(fill(Inf, 2, N), side, time)
            for i in 1:2
                for j in 1:N
                    newcontact = copy(contact)
                    newcontact[i, j] = !newcontact[i, j]
                    newresult = MPC.run_opt(sys, state, time, side, newcontact)
                    if newresult.status == :Optimal
                        newcosts[i, j] = getvalue(newresult.objective)
                    end
                end
            end
            # push!(records, Record(state, contact, duals, getvalue(result.objective), newcosts))
            for (constraint, duals) in result.constraints
                for constraint_t in 1:N-1
                    for contact_side in [:left, :right]
                        for contact_t in 1:N
                            samples[Axis{:sample}(sample_index),
                                    Axis{:constraint}(constraint),
                                    Axis{:constraint_t}(constraint_t),
                                    Axis{:contact_side}(contact_side),
                                    Axis{:contact_t}(contact_t)] =
                                Sample(getdual(duals[constraint_t]),
                                       newcosts[Axis{:side}(contact_side),
                                                Axis{:time}(contact_t)] - getvalue(result.objective))
                        end
                    end
                end
            end
            sample_index += 1
        end
    end
    samples
end

function phi(samples)
    count = zeros(2, 2)
    for s in samples
        if abs(s.dual) > 1e-5
            i = 2
        else
            i = 1
        end
        if s.Δcost < -1e-5
            j = 2
        else
            j = 1
        end
        count[i, j] += 1
    end
    (count[2, 2] * count[1, 1] - count[2, 1] * count[1, 2]) / sqrt(sum(count[:, 1]) * sum(count[:, 2]) * sum(count[1, :]) * sum(count[2, :]))
end

function correlate(samples)
    correlations = similar(samples[Axis{:sample}(1)], Float64)
    for I in eachindex(correlations)
        correlations[I] = phi(samples[:, I])
    end
    correlations
end

end
