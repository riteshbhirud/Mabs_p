"""
    PseudoSite{S<:Vector{ITensors.Index}} <: MabsAlg

Algorithm for representing bosonic systems using quantics (binary) mapping.
Maps a bosonic Hilbert space of dimension 2^N to N qubits per mode.

Fields:
- sites::S: Vector of qubit site indices (N_qubits × N_modes total)
- n_qubits_per_mode::Int: Number of qubits per bosonic mode
- n_modes::Int: Number of bosonic modes
- original_max_occ::Int: Maximum occupation in original bosonic space (2^N - 1)

The quantics mapping represents occupation number states in binary:
|n⟩ → |b_{N-1}⟩⊗|b_{N-2}⟩⊗...⊗|b_0⟩ where n = Σᵢ bᵢ × 2^i
"""
struct PseudoSite{S<:Vector{ITensors.Index}} <: MabsAlg 
    sites::S
    n_qubits_per_mode::Int
    n_modes::Int
    original_max_occ::Int
    
    function PseudoSite(sites::Vector{ITensors.Index}, n_qubits_per_mode::Int, 
                        n_modes::Int, original_max_occ::Int)
        length(sites) == n_qubits_per_mode * n_modes || 
            throw(ArgumentError("Site count mismatch: expected $(n_qubits_per_mode * n_modes), got $(length(sites))"))
        original_max_occ == 2^n_qubits_per_mode - 1 || 
            throw(ArgumentError(PSEUDOSITE_ERROR))
        for site in sites
            ITensors.dim(site) == 2 || 
                throw(ArgumentError("All sites must be qubits (dimension 2)"))
        end
        return new{typeof(sites)}(sites, n_qubits_per_mode, n_modes, original_max_occ)
    end
end

"""
    PseudoSite(n_modes::Int, max_occ::Int)

Create PseudoSite algorithm specification for bosonic system.

Arguments:
- n_modes::Int: Number of bosonic modes
- max_occ::Int: Maximum occupation number (must be 2^N - 1 for some integer N)

Returns:
- PseudoSite: Algorithm specification with generated qubit sites
"""
function PseudoSite(n_modes::Int, max_occ::Int)
    N = log2(max_occ + 1)
    isinteger(N) || throw(ArgumentError(PSEUDOSITE_ERROR))
    N_qubits = Int(N)
    sites = ITensors.Index[]
    for mode in 1:n_modes
        for qubit in 1:N_qubits
            tag = "Qubit,Mode=$mode,Bit=$qubit"
            push!(sites, ITensors.Index(2, tag))
        end
    end
    return PseudoSite(sites, N_qubits, n_modes, max_occ)
end

function Base.:(==)(alg1::PseudoSite, alg2::PseudoSite)
    return alg1.sites == alg2.sites &&
           alg1.n_qubits_per_mode == alg2.n_qubits_per_mode &&
           alg1.n_modes == alg2.n_modes &&
           alg1.original_max_occ == alg2.original_max_occ
end

"""
    get_mode_cluster(alg::PseudoSite, mode::Int)

Get the qubit cluster indices for a specific bosonic mode.

Arguments:
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number (1-indexed)

Returns:
- Vector{ITensors.Index}: Qubit sites for this mode
"""
function get_mode_cluster(alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.n_modes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.n_modes)]"))
    start_idx = (mode - 1) * alg.n_qubits_per_mode + 1
    end_idx = mode * alg.n_qubits_per_mode
    return alg.sites[start_idx:end_idx]
end

"""
    get_mode_indices(alg::PseudoSite, mode::Int)

Get the position indices in MPS for a specific mode's qubit cluster.

Arguments:
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number (1-indexed)

Returns:
- UnitRange{Int}: Position indices for this mode's qubits
"""
function get_mode_indices(alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.n_modes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.n_modes)]"))
    
    start_idx = (mode - 1) * alg.n_qubits_per_mode + 1
    end_idx = mode * alg.n_qubits_per_mode
    
    return start_idx:end_idx
end

"""
    decimal_to_binary_state(n::Int, n_qubits::Int)

Convert decimal occupation number to binary state vector.

Arguments:
- n::Int: Occupation number
- n_qubits::Int: Number of qubits

Returns:
- Vector{Int}: Binary representation [b_0, b_1, ..., b_{N-1}] where bᵢ ∈ {1,2} (ITensor convention)
"""
function decimal_to_binary_state(n::Int, n_qubits::Int)
    n >= 0 || throw(ArgumentError("Occupation number must be non-negative"))
    n < 2^n_qubits || throw(ArgumentError("Occupation $n exceeds max for $n_qubits qubits"))
    binary_state = Vector{Int}(undef, n_qubits)
    for i in 1:n_qubits
        bit = (n >> (i-1)) & 1  
        binary_state[i] = bit + 1  
    end
    return binary_state
end

"""
    binary_state_to_decimal(binary_state::Vector{Int})

Convert binary state vector to decimal occupation number.

Arguments:
- binary_state::Vector{Int}: Binary state [b_0, b_1, ..., b_{N-1}] where bᵢ ∈ {1,2}

Returns:
- Int: Occupation number n = Σᵢ (bᵢ - 1) × 2^(i-1)
"""
function binary_state_to_decimal(binary_state::Vector{Int})
    n = 0
    for (i, bit) in enumerate(binary_state)
        (bit == 1 || bit == 2) || throw(ArgumentError("Binary state values must be 1 or 2"))
        if bit == 2 
            n += 2^(i-1)
        end
    end
    return n
end