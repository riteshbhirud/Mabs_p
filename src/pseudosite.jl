"""
    PseudoSite <: MabsAlg

Algorithm for representing bosonic systems using quantics (binary) mapping.
Maps a bosonic Hilbert space of dimension 2^N to N qubits per mode.

Fields:
- n_modes::Int: Number of bosonic modes
- fock_cutoff::Int: Maximum occupation in bosonic space (must be 2^N - 1)

The quantics mapping represents occupation number states in binary:
|n⟩ → |b_{N-1}⟩⊗|b_{N-2}⟩⊗...⊗|b_0⟩ where n = Σᵢ bᵢ × 2^i
"""
struct PseudoSite <: MabsAlg 
    n_modes::Int
    fock_cutoff::Int
    
    function PseudoSite(n_modes::Int, fock_cutoff::Int)
        # Verify fock_cutoff is of form 2^N - 1
        N = log2(fock_cutoff + 1)
        isinteger(N) || throw(ArgumentError(PSEUDOSITE_ERROR))
        
        return new(n_modes, fock_cutoff)
    end
end

"""
    n_qubits_per_mode(alg::PseudoSite)

Get number of qubits needed per mode: log₂(fock_cutoff + 1)
"""
n_qubits_per_mode(alg::PseudoSite) = Int(log2(alg.fock_cutoff + 1))

"""
    create_qubit_sites(alg::PseudoSite)

Generate qubit sites for PseudoSite algorithm.
Creates n_modes × n_qubits_per_mode qubit indices.

Returns:
- Vector{ITensors.Index}: Qubit sites for the system
"""
function create_qubit_sites(alg::PseudoSite)
    n_qubits = n_qubits_per_mode(alg)
    n_total = alg.n_modes * n_qubits
    sites = Vector{ITensors.Index}(undef, n_total)
    
    idx = 1
    for mode in 1:alg.n_modes
        for qubit in 1:n_qubits
            tag = "Qubit,Mode=$mode,Bit=$qubit"
            sites[idx] = ITensors.Index(2, tag)
            idx += 1
        end
    end
    
    return sites
end

function Base.:(==)(alg1::PseudoSite, alg2::PseudoSite)
    return alg1.n_modes == alg2.n_modes &&
           alg1.fock_cutoff == alg2.fock_cutoff
end

"""
    get_mode_cluster(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)

Get the qubit cluster indices for a specific bosonic mode.

Arguments:
- sites::Vector{ITensors.Index}: Qubit sites for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number (1-indexed)

Returns:
- Vector{ITensors.Index}: Qubit sites for this mode
"""
function get_mode_cluster(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.n_modes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.n_modes)]"))
    
    n_qubits = n_qubits_per_mode(alg)
    start_idx = (mode - 1) * n_qubits + 1
    end_idx = mode * n_qubits
    
    return sites[start_idx:end_idx]
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
    
    n_qubits = n_qubits_per_mode(alg)
    start_idx = (mode - 1) * n_qubits + 1
    end_idx = mode * n_qubits
    
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