function createcluster!(ising::Ising, site::CartesianIndex, spin_val::Int)
    ising.cluster[site] = true
    ising.spins[site] = spin_val

    for nn_table in ising.neighbours_table 
        new_site = nn_table[site]
        if ising.spins[new_site] ≡ -ising.spins[site] && ising.cluster[new_site] ≡ false
            if rand() < 1 - exp(-2 * ising.β)
                createcluster!(ising, new_site, spin_val)
            end
        end
    end
end

function wolff!(ising::Ising)
    fill!(ising.cluster, false)
    rand_site = rand(ising.sites)
    spin_val = ising.spins[rand_site]
    createcluster!(ising, rand_site, spin_val)
end