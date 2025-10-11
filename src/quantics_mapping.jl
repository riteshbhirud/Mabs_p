"""
    set_cluster_matrix_element!(op::ITensors.ITensor, 
                                cluster_sites::Vector{<:ITensors.Index},
                                bra_state::Vector{Int}, 
                                ket_state::Vector{Int},
                                value::Number)

Set matrix element in qubit cluster operator.

Arguments:
- op::ITensors.ITensor: Operator to modify (must have primed and unprimed cluster indices)
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites (unprimed)
- bra_state::Vector{Int}: Bra state in binary (for primed indices)
- ket_state::Vector{Int}: Ket state in binary (for unprimed indices)
- value::Number: Matrix element value
"""
function set_cluster_matrix_element!(op::ITensors.ITensor, 
                                     cluster_sites::Vector{<:ITensors.Index},
                                     bra_state::Vector{Int}, 
                                     ket_state::Vector{Int},
                                     value::Number)
    n_qubits = length(cluster_sites)
    @assert length(bra_state) == n_qubits "Bra state length mismatch"
    @assert length(ket_state) == n_qubits "Ket state length mismatch"
    
    index_specs = Vector{Pair{ITensors.Index,Int}}(undef, 2 * n_qubits)
    
    idx = 1
    for i in 1:n_qubits
        index_specs[idx] = cluster_sites[i]' => bra_state[i]
        idx += 1
        index_specs[idx] = cluster_sites[i] => ket_state[i]
        idx += 1
    end
    
    op[index_specs...] = convert(ComplexF64, value)
    return nothing
end

"""
    _matrix_to_cluster_operator(matrix::Matrix{ComplexF64}, 
                                cluster_sites::Vector{<:ITensors.Index})

Convert a matrix in Fock basis to ITensor operator in qubit cluster basis.

Arguments:
- matrix::Matrix{ComplexF64}: Operator matrix in Fock basis (dim × dim)
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for cluster

Returns:
- ITensors.ITensor: Operator with primed and unprimed cluster indices
"""
function _matrix_to_cluster_operator(matrix::Matrix{ComplexF64}, 
                                     cluster_sites::Vector{<:ITensors.Index})
    n_qubits = length(cluster_sites)
    dim = size(matrix, 1)
    @assert size(matrix, 2) == dim "Matrix must be square"
    @assert dim == 2^n_qubits "Matrix dimension must match qubit space: $dim ≠ $(2^n_qubits)"
    
    op = ITensors.ITensor(ComplexF64, cluster_sites'..., cluster_sites...)
    
    for m in 0:(dim-1)
        binary_m = decimal_to_binary_state(m, n_qubits)
        for n in 0:(dim-1)
            element = matrix[m+1, n+1]
            if abs(element) > 1e-15  
                binary_n = decimal_to_binary_state(n, n_qubits)
                set_cluster_matrix_element!(op, cluster_sites, binary_m, binary_n, element)
            end
        end
    end
    
    return op
end

"""
    number_op_quantics(cluster_sites::Vector{<:ITensors.Index})

Create number operator in quantics representation.

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode

Returns:
- ITensors.ITensor: Number operator n̂ = Σₙ n|n⟩⟨n|
"""
function number_op_quantics(cluster_sites::Vector{<:ITensors.Index})
    n_qubits = length(cluster_sites)
    max_occ = 2^n_qubits - 1
    n_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    for n in 0:max_occ
        n_matrix[n+1, n+1] = n
    end
    return _matrix_to_cluster_operator(n_matrix, cluster_sites)
end

"""
    create_op_quantics(cluster_sites::Vector{<:ITensors.Index})

Create bosonic creation operator in quantics representation.

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode

Returns:
- ITensors.ITensor: Creation operator a† with ⟨n+1|a†|n⟩ = √(n+1)
"""
function create_op_quantics(cluster_sites::Vector{<:ITensors.Index})
    n_qubits = length(cluster_sites)
    max_occ = 2^n_qubits - 1
    a_dag_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    for n in 0:max_occ-1
        a_dag_matrix[n+2, n+1] = sqrt(n + 1)
    end
    return _matrix_to_cluster_operator(a_dag_matrix, cluster_sites)
end

"""
    destroy_op_quantics(cluster_sites::Vector{<:ITensors.Index})

Create bosonic annihilation operator in quantics representation.

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode

Returns:
- ITensors.ITensor: Annihilation operator a with ⟨n-1|a|n⟩ = √n
"""
function destroy_op_quantics(cluster_sites::Vector{<:ITensors.Index})
    n_qubits = length(cluster_sites)
    max_occ = 2^n_qubits - 1
    a_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    for n in 1:max_occ
        a_matrix[n, n+1] = sqrt(n)
    end
    return _matrix_to_cluster_operator(a_matrix, cluster_sites)
end

"""
    displace_op_quantics(cluster_sites::Vector{<:ITensors.Index}, α::Number)

Create displacement operator in quantics representation.
D(α) = exp(α*a† - α*a)

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode
- α::Number: Displacement amplitude (complex)

Returns:
- ITensors.ITensor: Displacement operator
"""
function displace_op_quantics(cluster_sites::Vector{<:ITensors.Index}, α::Number)
    n_qubits = length(cluster_sites)
    max_occ = 2^n_qubits - 1
    D_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    exp_factor = exp(-abs2(α) / 2)
    for m in 0:max_occ
        for n in 0:max_occ
            D_matrix[m+1, n+1] = _displacement_matrix_element(m, n, α) * exp_factor
        end
    end
    return _matrix_to_cluster_operator(D_matrix, cluster_sites)
end

"""
    _displacement_matrix_element(m::Int, n::Int, α::Number)

Compute matrix element ⟨m|D(α)|n⟩ without exponential prefactor.

Uses the formula:
⟨m|D(α)|n⟩ = α^(m-n)/√((m-n)! m!/n!)  if m ≥ n
           = (-α*)^(n-m)/√((n-m)! n!/m!)  if m < n
"""
function _displacement_matrix_element(m::Int, n::Int, α::Number)
    if m >= n
        k = m - n
        if k == 0
            return 1.0
        end
        return (α^k) / sqrt(_safe_factorial(k)) * 
               sqrt(_safe_factorial(m) / _safe_factorial(n))
    else
        k = n - m
        return ((-conj(α))^k) / sqrt(_safe_factorial(k)) * 
               sqrt(_safe_factorial(n) / _safe_factorial(m))
    end
end

"""
    squeeze_op_quantics(cluster_sites::Vector{<:ITensors.Index}, ξ::Number)

Create squeezing operator in quantics representation.
S(ξ) = exp(0.5*(ξ*a†² - ξ*a²))

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode
- ξ::Number: Squeezing parameter (complex)

Returns:
- ITensors.ITensor: Squeezing operator
"""
function squeeze_op_quantics(cluster_sites::Vector{<:ITensors.Index}, ξ::Number)
    n_qubits = length(cluster_sites)
    max_occ = 2^n_qubits - 1
    S_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    r = abs(ξ)
    φ = angle(ξ)
    for m in 0:max_occ
        for n in 0:max_occ
            if (m + n) % 2 == 0  
                S_matrix[m+1, n+1] = _squeezing_matrix_element(m, n, r, φ)
            end
        end
    end
    
    return _matrix_to_cluster_operator(S_matrix, cluster_sites)
end

"""
    _squeezing_matrix_element(m::Int, n::Int, r::Real, φ::Real)

Compute matrix element ⟨m|S(ξ)|n⟩ for squeezing operator.

"""
function _squeezing_matrix_element(m::Int, n::Int, r::Real, φ::Real)
    k_max = min(m, n) ÷ 2
    element = 0.0 + 0.0im
    cr = cosh(r)
    tr = tanh(r)
    phase = exp(2im * φ)
    prefactor = -0.5 * tr * phase
    inv_sqrt_cr = 1.0 / sqrt(cr)
    for k in 0:k_max
        coeff = sqrt(_safe_factorial(m) * _safe_factorial(n)) / 
               (_safe_factorial(k) * _safe_factorial(m - 2k) * _safe_factorial(n - 2k))
        
        coeff *= (prefactor^k) * inv_sqrt_cr
        if m == n && k == 0
            coeff *= inv_sqrt_cr
        end
        element += coeff
    end
    return element
end

"""
    kerr_op_quantics(cluster_sites::Vector{<:ITensors.Index}, χ::Real, t::Real)

Create Kerr evolution operator in quantics representation.
K(χ,t) = exp(-i*χ*t*n²)

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode
- χ::Real: Kerr nonlinearity strength
- t::Real: Evolution time

Returns:
- ITensors.ITensor: Kerr evolution operator
"""
function kerr_op_quantics(cluster_sites::Vector{<:ITensors.Index}, χ::Real, t::Real)
    n_qubits = length(cluster_sites)
    max_occ = 2^n_qubits - 1
    K_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    for n in 0:max_occ
        K_matrix[n+1, n+1] = exp(-1im * χ * t * n^2)
    end
    return _matrix_to_cluster_operator(K_matrix, cluster_sites)
end

"""
    expect_photon_number(psi::BMPS{<:ITensorMPS.MPS,PseudoSite}, mode::Int)

Compute photon number expectation value for a specific mode.

Arguments:
- psi::BMPS: State in PseudoSite representation
- mode::Int: Mode index (1 to n_modes)

Returns:
- Float64: ⟨n⟩ for the specified mode
"""
function expect_photon_number(psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, mode::Int)
    alg = psi.alg
    sites = ITensorMPS.siteinds(psi.mps)
    cluster_sites = get_mode_cluster(sites, alg, mode)
    n_op = number_op_quantics(cluster_sites)
    n_psi = ITensors.apply(n_op, psi.mps)
    return real(ITensorMPS.inner(psi.mps, n_psi))
end

"""
Build hopping MPO for adjacent modes using direct construction.
Handles â†ᵢ âᵢ₊₁ term.
"""
function build_hopping_mpo(
    sites::Vector{<:ITensors.Index},
    alg::PseudoSite,
    mode_i::Int,
    coeff::Number
)
    n_qubits = n_qubits_per_mode(alg)
    max_occ = alg.fock_cutoff
    
    mode_j = mode_i + 1
    if mode_j > alg.n_modes
        error("Mode index out of range")
    end
    start_i = (mode_i - 1) * n_qubits + 1
    end_i = mode_i * n_qubits
    start_j = (mode_j - 1) * n_qubits + 1
    end_j = mode_j * n_qubits
    n_total = length(sites)
    
    terms = ITensorMPS.MPO[]
    for ni in 0:max_occ, nj in 0:max_occ
        if ni >= max_occ  
            continue
        end
        if nj == 0  
            continue
        end
        mat_elem = sqrt(ni + 1) * sqrt(nj) * coeff
        if abs(mat_elem) < 1e-15
            continue
        end
        bra_states_i = decimal_to_binary_state(ni, n_qubits)
        bra_states_j = decimal_to_binary_state(nj, n_qubits)
        ket_states_i = decimal_to_binary_state(ni + 1, n_qubits)
        ket_states_j = decimal_to_binary_state(nj - 1, n_qubits)
        mpo_tensors = Vector{ITensors.ITensor}(undef, n_total)
        for site_idx in 1:n_total
            s = sites[site_idx]
            if start_i <= site_idx <= end_i
                local_idx = site_idx - start_i + 1
                ket_val = ket_states_i[local_idx]
                bra_val = bra_states_i[local_idx]
                mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
                mpo_tensors[site_idx][s' => bra_val, s => ket_val] = mat_elem
                
            elseif start_j <= site_idx <= end_j
                local_idx = site_idx - start_j + 1
                ket_val = ket_states_j[local_idx]
                bra_val = bra_states_j[local_idx]
                mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
                mpo_tensors[site_idx][s' => bra_val, s => ket_val] = 1.0
                
            else
                mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
                mpo_tensors[site_idx][s' => 1, s => 1] = 1.0
                mpo_tensors[site_idx][s' => 2, s => 2] = 1.0
            end
        end
        term_mpo = ITensorMPS.MPO(mpo_tensors)
        push!(terms, term_mpo)
    end
    for ni in 0:max_occ, nj in 0:max_occ
        if ni == 0  
            continue
        end
        if nj >= max_occ 
            continue
        end
        mat_elem = sqrt(ni) * sqrt(nj + 1) * coeff
        
        if abs(mat_elem) < 1e-15
            continue
        end
        bra_states_i = decimal_to_binary_state(ni, n_qubits)
        bra_states_j = decimal_to_binary_state(nj, n_qubits)
        ket_states_i = decimal_to_binary_state(ni - 1, n_qubits)
        ket_states_j = decimal_to_binary_state(nj + 1, n_qubits)
        mpo_tensors = Vector{ITensors.ITensor}(undef, n_total)
        for site_idx in 1:n_total
            s = sites[site_idx]
            if start_i <= site_idx <= end_i
                local_idx = site_idx - start_i + 1
                ket_val = ket_states_i[local_idx]
                bra_val = bra_states_i[local_idx]
                mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
                mpo_tensors[site_idx][s' => bra_val, s => ket_val] = mat_elem
                
            elseif start_j <= site_idx <= end_j
                local_idx = site_idx - start_j + 1
                ket_val = ket_states_j[local_idx]
                bra_val = bra_states_j[local_idx]
                mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
                mpo_tensors[site_idx][s' => bra_val, s => ket_val] = 1.0
            else
                mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
                mpo_tensors[site_idx][s' => 1, s => 1] = 1.0
                mpo_tensors[site_idx][s' => 2, s => 2] = 1.0
            end
        end
        term_mpo = ITensorMPS.MPO(mpo_tensors)
        push!(terms, term_mpo)
    end
    
    if isempty(terms)
        mpo_tensors = Vector{ITensors.ITensor}(undef, n_total)
        for site_idx in 1:n_total
            s = sites[site_idx]
            mpo_tensors[site_idx] = ITensors.ITensor(ComplexF64, s', s)
            mpo_tensors[site_idx][s' => 1, s => 1] = 0.0
            mpo_tensors[site_idx][s' => 2, s => 2] = 0.0
        end
        return ITensorMPS.MPO(mpo_tensors)
    end
    
    result = terms[1]
    for i in 2:length(terms)
        result = ITensorMPS.add(result, terms[i]; cutoff=1e-15)
    end
    return result
end