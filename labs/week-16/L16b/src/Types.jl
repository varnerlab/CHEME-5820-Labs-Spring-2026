abstract type AbstractNeuralNetwork end
abstract type AbstractLearningModel end
abstract type AbstractWorldContextModel end

"""
    struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork

A thin wrapper around a Flux `Chain` so that `Flux.@layer` can register the
trainable submodules by name. Used to build both the main and target Q-networks.

### Fields
    - chain::Chain: the underlying Flux model.
"""
struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork
    chain::Chain
end

"""
    mutable struct MyDQNLearningAgentModel <: AbstractLearningModel

DQN agent state shared across the training loop.

### Fields
    - actions::Dict{Int64, Vector{Float32}}: action index → 4-d thrust vector applied to (vx, vy)
    - mainnetwork::Chain: main Q-network (the one we update at every step)
    - targetnetwork::Chain: delayed target Q-network used to compute bootstrap targets
    - number_of_actions::Int64: |𝒜|, equals length(actions)
    - number_of_inputs::Int64: dimension of the state vector consumed by the networks
    - replaybuffer::CircularBuffer{...}: stored transitions (s, a, r, s′, done)
"""
mutable struct MyDQNLearningAgentModel <: AbstractLearningModel

    actions::Dict{Int64, Vector{Float32}}
    mainnetwork::Chain
    targetnetwork::Chain
    number_of_actions::Int64
    number_of_inputs::Int64
    replaybuffer::CircularBuffer{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}

    MyDQNLearningAgentModel() = new();
end

"""
    mutable struct MyDQNworldContextModel <: AbstractWorldContextModel

Continuous LavaWorld parameters consumed by the `world(...)` function. Holds
the board geometry, the lava and charger locations, and the dynamics knobs
(thrust gain, drag, step cost, terminal rewards, noise).

### Fields
    - nrows::Int: number of grid rows (board is x ∈ [0, ncols], y ∈ [0, nrows])
    - ncols::Int: number of grid columns
    - lava::Vector{Tuple{Float32,Float32}}: (x, y) centers of lava pits
    - charger::Tuple{Float32,Float32}: (x, y) of the goal/charger
    - lava_radius::Float32: radius of each lava pit (any visit inside terminates with lava penalty)
    - charger_radius::Float32: radius of the charger (visit inside terminates with charger reward)
    - thrust::Float32: thrust gain per action (Δv per step in chosen direction)
    - drag::Float32: linear-drag coefficient applied to velocity at every step (∈ [0, 1])
    - dt::Float32: integration step size
    - step_cost::Float32: per-step reward (negative; encourages reaching the goal quickly)
    - lava_reward::Float32: terminal reward for hitting any lava pit
    - charger_reward::Float32: terminal reward for hitting the charger
    - σ::Float32: actuator noise std added to the thrust each step
    - vmax::Float32: cap on the magnitude of vx, vy
    - Z::Normal{Float32}: standard Normal used to draw actuator noise
"""
mutable struct MyDQNworldContextModel <: AbstractWorldContextModel

    nrows::Int
    ncols::Int
    lava::Vector{Tuple{Float32, Float32}}
    charger::Tuple{Float32, Float32}
    lava_radius::Float32
    charger_radius::Float32
    thrust::Float32
    drag::Float32
    dt::Float32
    step_cost::Float32
    lava_reward::Float32
    charger_reward::Float32
    σ::Float32
    vmax::Float32
    Z::Normal{Float32}

    MyDQNworldContextModel() = new();
end
