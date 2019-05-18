using BenchmarkTools
using Random
Random.seed!(42)

struct Ising{D, T <: AbstractFloat}
    dims::Dims{D}
    h::T
    β::T
    spins::Array{Int, D}
    cluster::Array{Bool, D}
    sites::Array{CartesianIndex{D},D}
    neighbours_table::Vector{Array{CartesianIndex{D},D}}
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
    Ising{D, Float64}(dims, h, β, spins, cluster, sites, neighbours_table)
end

function energy_singleflip(ising::Ising, site::CartesianIndex)
    E = 0
    for nn_table in ising.neighbours_table 
        new_site = nn_table[site]
        if ising.spins[site] ≡ ising.spins[new_site]
            E += 1
        else
            E -= 1
        end
    end
    return 2E
end

function metropolis_step!(ising::Ising)
    rand_site= rand(ising.sites)
    dE = energy_singleflip(ising, rand_site)
    if dE < 0 
        ising.spins[rand_site] *= -1
    elseif rand() < exp(-dE)
        ising.spins[rand_site] *= -1
    end
end

magnetization(ising::Ising) = abs(sum(ising.spins))

function energy(ising::Ising)
    E = 0
    for site in ising.sites
        for nn_table in ising.neighbours_table 
            new_site = nn_table[site]
            E += ising.spins[site] * ising.spins[new_site]
        end
    end
    return E
end

function update_observables!(ising::Ising, observables::Observables)
    observables.E += energy(ising)
    observables.M += magnetization(ising)
end

function metropolis_sweep!(ising::Ising, observables::Observables)
    for site in eachindex(ising.spins)
        metropolis_step!(ising)
    end
    update_observables!(ising, observables)
end

function run!(ising::Ising, observables::Observables, N::Int)
    for i in 1:N
        metropolis_sweep!(ising, observables)
    end
end


ising = Ising((40, 40))
observables = Observables()
run!(ising, observables, 1)

@btime run!(ising, observables, 5000)