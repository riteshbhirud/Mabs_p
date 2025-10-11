
# helper function for safe factorial calculation to avoid performance issues.
function _safe_factorial(n::Int)
    if n <= 20
        return factorial(n)
    else
        return factorial(big(n))
    end
end

"""
    create(site::ITensors.Index)

Create the bosonic creation operator (raising operator) for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag

Returns:
- ITensors.ITensor: Creation operator tensor
"""
function create(site::ITensors.Index)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 0:(max_occ-1)
        op[n+2, n+1] = sqrt(n+1)
    end
    return op
end

"""
    destroy(site::ITensors.Index)

Create the bosonic annihilation operator (lowering operator) for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag

Returns:
- ITensors.ITensor: Annihilation operator tensor
"""
function destroy(site::ITensors.Index)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 1:max_occ
        op[n, n+1] = sqrt(n)
    end
    return op
end

"""
    number(site::ITensors.Index)

Create the bosonic number operator for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag

Returns:
- ITensors.ITensor: Number operator tensor
"""
function number(site::ITensors.Index)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 0:max_occ
        op[n+1, n+1] = n
    end
    return op
end

"""
    displace(site::ITensors.Index, α::Number)

Create the displacement operator `D(α) = exp(α*a† - α*a)` for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- α::Number: Displacement amplitude (can be complex)

Returns:
- ITensors.ITensor: Displacement operator tensor
"""
function displace(site::ITensors.Index, α::Number)
    #  G = α*a† - α*a
    a_dag = create(site)
    a = destroy(site)
    generator = α * a_dag - conj(α) * a
    op = ITensors.exp(generator)
    return op
end

"""
    squeeze(site::ITensors.Index, ξ::Number)

Create the squeezing operator `S(ξ) = exp(0.5*(ξ*a†² - ξ*a²))` for a given site.
Uses direct matrix element construction for numerical stability.

Arguments:
- site::ITensors.Index: Site index with bosonic tag  
- ξ::Number: Squeezing parameter (can be complex)

Returns:
- ITensors.ITensor: Squeezing operator tensor
"""
function squeeze(site::ITensors.Index, ξ::Number)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    r = abs(ξ)
    φ = angle(ξ)
    @inbounds for n in 0:max_occ
        @inbounds for m in 0:max_occ
            if (n + m) % 2 == 0 
                k_max = min(n, m) ÷ 2
                element = 0.0 + 0.0im
                @inbounds for k in 0:k_max
                    coeff = sqrt(_safe_factorial(n) * _safe_factorial(m)) / 
                           (_safe_factorial(k) * _safe_factorial((n-2k)) * _safe_factorial((m-2k)))
                    coeff *= (-0.5 * tanh(r) * exp(2im*φ))^k / sqrt(cosh(r))
                    if n == m && k == 0
                        coeff /= sqrt(cosh(r))  
                    end
                    element += coeff
                end
                op[m+1, n+1] = convert(ComplexF64, element)
            end
        end
    end
    return op
end

"""
    kerr(site::ITensors.Index, χ::Real, t::Real)

Create the Kerr evolution operator `exp(-i*χ*t*n²)` for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- χ::Real: Kerr nonlinearity strength
- t::Real: Evolution time

Returns:
- ITensors.ITensor: Kerr evolution operator tensor
"""
function kerr(site::ITensors.Index, χ::Real, t::Real)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 0:max_occ
        phase = exp(-1im * χ * t * n^2)
        op[n+1, n+1] = phase
    end
    return op
end

"""
    harmonic_chain(sites::Vector{<:ITensors.Index}; ω::Real=1.0, J::Real=0.0)

Build MPO for a chain of harmonic oscillators with optional nearest-neighbor coupling.
Here the Hamiltonian is `H = Σᵢ ω*nᵢ + J*Σᵢ (aᵢ†aᵢ₊₁ + aᵢaᵢ₊₁†)`.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- ω::Real: Harmonic oscillator frequency (default: 1.0)
- J::Real: Nearest-neighbor hopping strength (default: 0.0)

Returns:
- BMPO: Matrix product operator for harmonic chain
"""
function harmonic_chain(sites::Vector{<:ITensors.Index}; ω::Real=1.0, J::Real=0.0)
    opsum = ITensors.OpSum()
    @inbounds for i in 1:length(sites)
        opsum += ω, "N", i
    end
    if J != 0.0
        @inbounds for i in 1:(length(sites)-1)
            opsum += J, "Adag", i, "A", i+1
            opsum += J, "A", i, "Adag", i+1
        end
    end
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, Truncated())
end

"""
    kerr(sites::Vector{<:ITensors.Index}; ω::Real=1.0, χ::Real=0.1)

Build MPO for a chain of Kerr oscillators.
Here the Hamiltonian is `H = Σᵢ (ω*nᵢ + χ*nᵢ²)`

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- ω::Real: Linear frequency (default: 1.0)
- χ::Real: Kerr nonlinearity strength (default: 0.1)

Returns:
- BMPO: Matrix product operator for Kerr chain
"""
function kerr(sites::Vector{<:ITensors.Index}; ω::Real=1.0, χ::Real=0.1)
    opsum = ITensors.OpSum()
    @inbounds for i in 1:length(sites)
        opsum += ω, "N", i
    end
    @inbounds for i in 1:length(sites)
        opsum += χ, "N", i, "N", i
    end
    mpo = ITensorMPS.MPO(opsum, sites)

    return BMPO(mpo, Truncated())
end

function create(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    return create_op_quantics(cluster_sites)
end

function destroy(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    return destroy_op_quantics(cluster_sites)
end

function number(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    return number_op_quantics(cluster_sites)
end

function displace(sites::Vector{<:ITensors.Index}, α::Number, alg::PseudoSite, mode::Int)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    return displace_op_quantics(cluster_sites, α)
end

function squeeze(sites::Vector{<:ITensors.Index}, ξ::Number, alg::PseudoSite, mode::Int)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    return squeeze_op_quantics(cluster_sites, ξ)
end

function kerr(sites::Vector{<:ITensors.Index}, χ::Real, t::Real, alg::PseudoSite, mode::Int)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    return kerr_op_quantics(cluster_sites, χ, t)
end

"""
    harmonic_chain(sites::Vector{<:ITensors.Index}, alg::PseudoSite; ω::Real=1.0, J::Real=0.0)

Build harmonic chain Hamiltonian in PseudoSite representation.
H = Σᵢ ω*nᵢ + J*Σᵢ (aᵢ†aᵢ₊₁ + h.c.)

"""
function harmonic_chain(sites::Vector{<:ITensors.Index}, alg::PseudoSite; ω::Real=1.0, J::Real=0.0)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    opsum = ITensors.OpSum()
    n_qubits = n_qubits_per_mode(alg)
    
    for mode in 1:alg.n_modes
        for i in 1:n_qubits
            weight = ω * 2^(i-1)
            global_idx = (mode - 1) * n_qubits + i
            opsum += weight, "N", global_idx
        end
    end
    
    H = ITensorMPS.MPO(opsum, sites)
    if J != 0.0
        for mode in 1:(alg.n_modes - 1)
            H_hop = build_hopping_mpo(sites, alg, mode, J)
            H = ITensorMPS.add(H, H_hop; cutoff=1e-15)
        end
    end
    return BMPO(H, alg)
end

"""
    kerr(sites::Vector{<:ITensors.Index}, alg::PseudoSite; ω::Real=1.0, χ::Real=0.1)

Build Kerr nonlinearity Hamiltonian in PseudoSite representation.
H = Σᵢ (ω*nᵢ + χ*nᵢ²)
"""
function kerr(sites::Vector{<:ITensors.Index}, alg::PseudoSite; ω::Real=1.0, χ::Real=0.1)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    opsum = ITensors.OpSum()
    n_qubits = n_qubits_per_mode(alg)
    
    for mode in 1:alg.n_modes
        for i in 1:n_qubits
            weight = ω * 2^(i-1)
            global_idx = (mode - 1) * n_qubits + i
            opsum += weight, "N", global_idx
        end
        for i in 1:n_qubits
            for j in 1:n_qubits
                weight = χ * 2^(i + j - 2)
                idx_i = (mode - 1) * n_qubits + i
                idx_j = (mode - 1) * n_qubits + j
                
                if i == j
                    opsum += weight, "N", idx_i
                else
                    opsum += weight, "N", idx_i, "N", idx_j
                end
            end
        end
    end
    
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, alg)
end

function ITensors.op(::ITensors.OpName"N", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[0.0 0.0; 0.0 1.0], s', s)
end

function ITensors.op(::ITensors.OpName"Id", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[1.0 0.0; 0.0 1.0], s', s)
end

function ITensors.op(::ITensors.OpName"X", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[0.0 1.0; 1.0 0.0], s', s)
end

function ITensors.op(::ITensors.OpName"Z", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[1.0 0.0; 0.0 -1.0], s', s)
end

function ITensors.op(::ITensors.OpName"Y", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[0.0 -im; im 0.0], s', s)
end