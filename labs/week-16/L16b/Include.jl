# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

!isdir(_PATH_TO_FIGS) && mkpath(_PATH_TO_FIGS);

using Pkg;
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Statistics
using StatsBase
using LinearAlgebra
using Random
using DataFrames
using PrettyTables
using Plots
using Colors
using Distributions
using DataStructures
using Flux
using NNlib

# set the random seed for reproducibility -
Random.seed!(42);

# load local source files -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
