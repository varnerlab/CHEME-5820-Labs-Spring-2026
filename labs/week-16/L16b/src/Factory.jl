"""
    build(modeltype::Type{MyDQNworldContextModel}, data::NamedTuple) -> MyDQNworldContextModel

Construct a `MyDQNworldContextModel` from a NamedTuple of parameters. All fields
of the type must be supplied by the caller; no defaults are filled in here.
"""
function build(modeltype::Type{MyDQNworldContextModel}, data::NamedTuple)::MyDQNworldContextModel

    model = modeltype();
    model.nrows = data.nrows;
    model.ncols = data.ncols;
    model.lava = data.lava;
    model.charger = data.charger;
    model.lava_radius = data.lava_radius;
    model.charger_radius = data.charger_radius;
    model.thrust = data.thrust;
    model.drag = data.drag;
    model.dt = data.dt;
    model.step_cost = data.step_cost;
    model.lava_reward = data.lava_reward;
    model.charger_reward = data.charger_reward;
    model.σ = data.σ;
    model.vmax = data.vmax;
    model.Z = data.Z;

    return model;
end

"""
    build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple) -> MyDQNLearningAgentModel

Construct a `MyDQNLearningAgentModel`. The action set is built here from a
discrete thrust dictionary so the learn loop only sees indices: action 1 is
"+x thrust", action 2 is "−x thrust", action 3 is "+y thrust", action 4 is
"−y thrust", and action 5 is "coast" (no thrust). The thrust magnitude is
`data.thrust`. State dimension is fixed at 4 (x, y, vx, vy).

### Required NamedTuple fields
    - mainnetwork::Chain
    - targetnetwork::Chain
    - thrust::Float32  (magnitude of the velocity change per directed thrust)
    - number_of_inputs::Int64  (state dimension; 4 for (x, y, vx, vy))
"""
function build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple)::MyDQNLearningAgentModel

    model = modeltype();

    Δv = data.thrust;
    actions = Dict{Int64, Vector{Float32}}(
        1 => Float32[ Δv,  0.0],
        2 => Float32[-Δv,  0.0],
        3 => Float32[ 0.0,  Δv],
        4 => Float32[ 0.0, -Δv],
        5 => Float32[ 0.0,  0.0],
    );

    model.mainnetwork = data.mainnetwork;
    model.targetnetwork = data.targetnetwork;
    model.actions = actions;
    model.number_of_actions = length(actions);
    model.number_of_inputs = data.number_of_inputs;

    return model;
end
