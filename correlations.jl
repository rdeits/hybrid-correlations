module c

using AxisArrays
using JuMP
using DataStructures: OrderedDict
using LIBSVM

include("mpc.jl")

immutable Record
    model::MPC.MPCModel
    state::MPC.State{Float64}
    contact::AxisArray{Bool, 2, Array{Bool, 2}, Tuple{Axis{:side, Array{Symbol, 1}}, Axis{:time, LinSpace{Float64}}}}
    duals::Vector{Float64}
    cost::Float64
    new_costs::AxisArray{Float64, 2, Array{Float64, 2}, Tuple{Axis{:side, Array{Symbol, 1}}, Axis{:time, LinSpace{Float64}}}}
end

immutable Sample
    dual::Float64
    Δcost::Float64
end


function Record(model::MPC.MPCModel)
    side, time = axes(MPC.contactvars(model))
    N = length(time)
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
    result = MPC.solve!(model, state; contact_sequence=contact)
    # result = MPC.run_opt(sys, state, time, side, contact)
    if result.status != :Optimal
        result = MPC.solve!(model, state; contact_sequence=contact, relax=true)
        # MPC.relax!(result)
    end
    @assert result.model.m === model.m
    @assert result.status == :Optimal
    oldobjective = getvalue(getobjective(result.model.m))

    duals = copy(model.m.linconstrDuals)
    newcosts = AxisArray(fill(Inf, 2, N), side, time)
    for i in 1:2
        for j in 1:N
            newcontact = copy(contact)
            newcontact[i, j] = !newcontact[i, j]
            newresult = MPC.solve!(model, state; contact_sequence=newcontact)
            # newresult = MPC.run_opt(sys, state, time, side, newcontact)
            if newresult.status != :Optimal
                newresult = MPC.solve!(model, state; contact_sequence=newcontact, relax=true)
                # MPC.relax!(newresult)
            end
            @assert newresult.status == :Optimal
            newcosts[i, j] = getvalue(getobjective(newresult.model.m))
            @assert newresult.model.m === model.m
        end
    end
    Record(model, state, contact, duals, oldobjective, newcosts)
end

function samples(record::Record)
    constraints = record.model.constraints
    side, time = axes(record.model.contact)
    N = length(time)
    s = AxisArray(Array{Sample}(length(constraints),
                                N - 1,
                                length(side),
                                N),
                  Axis{:constraint}(collect(keys(constraints))),
                  Axis{:constraint_t}(1:(N-1)),
                  Axis{:contact_side}([:left, :right]),
                  Axis{:contact_t}(1:N))
    for (constraintname, refs) in constraints
        for constraint_t in 1:N-1
            d = record.duals[refs[constraint_t].idx]
            for contact_side in [:left, :right]
                for contact_t in 1:N
                    s[Axis{:constraint}(constraintname),
                      Axis{:constraint_t}(constraint_t),
                      Axis{:contact_side}(contact_side),
                      Axis{:contact_t}(contact_t)] = 
                        Sample(d, record.new_costs[Axis{:side}(contact_side),
                                                   Axis{:time}(contact_t)] - record.cost)
                end
            end
        end
    end
    s
end


# function collect_data(model::MPC.MPCModel, numsamples::Integer)
#     # dt = sys.Δt
#     # N = 10
#     # time = Axis{:time}(linspace(0, (N - 1) * dt, N))
#     # side = Axis{:side}([:left, :right])
#     records = Record[]

#     # model = MPC.create_model(sys, time, side)
#     side, time = axes(MPC.contactvars(model))
#     N = length(time)

#     while length(records) <= numsamples
#         contact = AxisArray(rand(Bool, 2, N), side, time)
#         # for j in 1:N
#         #     if all(contact[:, j])
#         #         contact[rand(1:2), j] = false
#         #     end
#         # end
#         q0 = 2 * rand() - 1
#         v0 = 4 * rand() - 2
#         qlimb0 = 2 * rand() - 1
#         state = MPC.State(q0, v0, qlimb0)
#         result = MPC.solve!(model, state; contact_sequence=contact)
#         # result = MPC.run_opt(sys, state, time, side, contact)
#         if result.status != :Optimal
#             result = MPC.solve!(model, state; contact_sequence=contact, relax=true)
#             # MPC.relax!(result)
#         end
#         @assert result.model.m === model.m
#         @assert result.status == :Optimal
#         oldobjective = getvalue(getobjective(result.model.m))

#         duals = model.m.linconstrDuals
#         newcosts = AxisArray(fill(Inf, 2, N), side, time)
#         for i in 1:2
#             for j in 1:N
#                 newcontact = copy(contact)
#                 newcontact[i, j] = !newcontact[i, j]
#                 newresult = MPC.solve!(model, state; contact_sequence=newcontact)
#                 # newresult = MPC.run_opt(sys, state, time, side, newcontact)
#                 if newresult.status != :Optimal
#                     newresult = MPC.solve!(model, state; contact_sequence=newcontact, relax=true)
#                     # MPC.relax!(newresult)
#                 end
#                 @assert newresult.status == :Optimal
#                 newcosts[i, j] = getvalue(getobjective(newresult.model.m))
#                 @assert newresult.model.m === model.m
#             end
#         end
#         push!(records, Record(state, contact, duals, oldobjective, newcosts))
#         # for (constraint, duals) in result.constraints
#         #     for constraint_t in 1:N-1
#         #         for contact_side in [:left, :right]
#         #             for contact_t in 1:N
#         #                 if contact[contact_side, contact_t]
#         #                     polarity = Axis{:polarity}(:off)
#         #                 else
#         #                     polarity = Axis{:polarity}(:on)
#         #                 end
#         #                 samples[Axis{:sample}(sample_index),
#         #                         Axis{:constraint}(constraint),
#         #                         Axis{:constraint_t}(constraint_t),
#         #                         Axis{:contact_side}(contact_side),
#         #                         Axis{:contact_t}(contact_t),
#         #                         polarity] =
#         #                     Sample(getdual(duals[constraint_t]),
#         #                            newcosts[Axis{:side}(contact_side),
#         #                                     Axis{:time}(contact_t)] - oldobjective)
#         #             end
#         #         end
#         #     end
#         # end
#         # sample_index += 1
#     end
#     # samples
#     records
# end

immutable Example
    contact::BitArray{1}
    duals::BitArray{1}
    best_action::Int
end

function Base.convert(::Type{Example}, r::Record)
    contact = vec(r.contact)
    duals = abs.(r.duals) .> 1e-5
    best_action = indmin(vec(r.new_costs))
    Example(contact, duals, best_action)
end

function correlation_counts(samples)
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
    count
end

function phi(samples)
    count = correlation_counts(samples)
    (count[2, 2] * count[1, 1] - count[2, 1] * count[1, 2]) / sqrt(sum(count[:, 1]) * sum(count[:, 2]) * sum(count[1, :]) * sum(count[2, :]))
end

function correlate(samples)
    correlations = similar(samples[Axis{:sample}(1)], Float64)
    for I in eachindex(correlations)
        # correlations[I] = cor([abs(s.dual) > 1e-5 for s in @view(samples[:, I])],
        #                       [s.Δcost for s in @view(samples[:, I])])
        correlations[I] = phi(samples[:, I])
    end
    correlations
end


end
