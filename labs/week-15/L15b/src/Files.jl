"""
    list_mnist_files(class::Int) -> Vector{String}

Return the absolute paths of every `.jpg` MNIST image under
`data/mnist/<class>/`, sorted by filename.
"""
function list_mnist_files(class::Int)::Vector{String}
    @assert 0 <= class <= 9 "class must lie in 0:9"
    dir = joinpath(_PATH_TO_MNIST, string(class))
    @assert isdir(dir) "directory $(dir) does not exist; did you copy the MNIST images?"
    files = filter(f -> endswith(f, ".jpg"), readdir(dir))
    return sort([joinpath(dir, f) for f in files])
end

"""
    load_mnist_image(class::Int, idx::Int = 1) -> Matrix{Float64}

Load the `idx`-th MNIST image of the given class as a `28 × 28` matrix of
grayscale values in `[0, 1]`. Defaults to the first image of that class.
"""
function load_mnist_image(class::Int, idx::Int = 1)::Matrix{Float64}
    files = list_mnist_files(class)
    @assert 1 <= idx <= length(files) "idx out of range; class $(class) has $(length(files)) images"
    img = Float64.(Gray.(load(files[idx])))
    return img
end
