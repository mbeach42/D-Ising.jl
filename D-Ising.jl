using ForwardDiff
using BenchmarkTools

# include("wolff.jl")

struct Ising{D, T}
    dims::Dims{D}
    h::Float64
    β::Float64
    spins::Array{Int, D}
    cluster::Array{Bool, D}
    sites::Array{CartesianIndex{D},D}
    neighbours_table::Array{CartesianIndex{D},T}
end
Base.size(ising::Ising) = ising.dims
Base.length(ising::Ising) = prod(ising.dims)

mutable struct Observables{T <: AbstractFloat}
    M::T
    E::T
    χ::T
    Cᵥ::T
end

Observables() = Observables(0.0, 0.0, 0.0, 0.0)

function unit_tuple(D::Int, i::Int, val::Int)
    tmp = zeros(Int, D)
    tmp[i] = val
    return Tuple(tmp)
end

function Ising(dims::Dims{D}; h::Real=0, β::Real=5) where D
    spins = rand(-1:2:1, dims)
    cluster = fill(false, dims)
    sites = CartesianIndices(dims)
    neighbours_table = [circshift(sites, unit_tuple(D, 1, i)) for i in -1:2:1]
    for j in 2:D
        append!(neighbours_table, [circshift(sites, unit_tuple(D, j, i)) for i in -1:2:1])
    end
    neighbours_table = cat(neighbours_table..., dims=D+1)
    Ising{D,D+1}(dims, h, β, spins, cluster, sites, neighbours_table)
end

function energy_singleflip(ising::Ising, site::CartesianIndex{D}) where D
    E = 0
    for n in 1:2D
        new_site = ising.neighbours_table[site, n]
        if ising.spins[site] ≡ ising.spins[new_site]
            E += 1
        else
            E -= 1
        end
    end
    E -= ising.h * ising.spins[site] 
    return 2E
end

function metropolis_step!(ising::Ising)
    rand_site = rand(ising.sites)
    dE = energy_singleflip(ising, rand_site)
    if dE < 0 || rand() < exp(-ising.β * dE)
        ising.spins[rand_site] *= -1
    end
end

magnetization(ising::Ising) = abs(sum(ising.spins))

function energy(ising::Ising)
    E = 0
    for site in ising.sites
        E += energy_singleflip(ising, site)
    end
    return E/2
end

function update_observables!(ising::Ising, observables::Observables)
    observables.E += energy(ising)
    observables.M += magnetization(ising)
end

function metropolis_sweep!(ising::Ising, observables::Observables)
    for site in eachindex(ising.spins)
        metropolis_step!(ising)
    end
end

function run!(ising::Ising, observables::Observables, N::Int)
    for i in 1:2N
        metropolis_sweep!(ising, observables)
        # wolff!(ising)
        if i > N
            update_observables!(ising, observables)
        end
    end
end


N = 5000
dims = (40,40)
ising = Ising(dims, h=0., β=1.0)
observables = Observables()
run!(ising, observables, 1)
@time run!(ising, observables, N)
println("E is ", observables.E/N/prod(dims))
println("M is ", observables.M/N/prod(dims))

N = 5000
dims = (10, 10, 10, 5)
ising = Ising(dims, h=0., β=1.0)
observables = Observables()
run!(ising, observables, 1)
@time run!(ising, observables, N)
println("E is ", observables.E/N/prod(dims))
println("M is ", observables.M/N/prod(dims))

# display(ising.spins)
# display(sum(ising.spins))
# @btime sum(ising.spins)

# g = x -> ForwardDiff.gradient(h -> energy(ising, h[]), x)
# @btime g([1])