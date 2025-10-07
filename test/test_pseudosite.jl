using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "PseudoSite Algorithm Tests" begin
    
    @testset "PseudoSite Construction" begin
        n_modes = 2
        max_occ = 7
        alg = PseudoSite(n_modes, max_occ)
        @test alg.n_modes == n_modes
        @test alg.n_qubits_per_mode == 3
        @test alg.original_max_occ == max_occ
        @test length(alg.sites) == n_modes * 3
        @test_throws ArgumentError PseudoSite(1, 10)
    end

    @testset "Binary Conversion" begin
        n_qubits = 3
        @test decimal_to_binary_state(0, n_qubits) == [1, 1, 1]
        @test decimal_to_binary_state(1, n_qubits) == [2, 1, 1]
        @test decimal_to_binary_state(2, n_qubits) == [1, 2, 1]
        @test decimal_to_binary_state(7, n_qubits) == [2, 2, 2]
        @test binary_state_to_decimal([1, 1, 1]) == 0
        @test binary_state_to_decimal([2, 1, 1]) == 1
        @test binary_state_to_decimal([1, 2, 1]) == 2
        @test binary_state_to_decimal([2, 2, 2]) == 7
        for n in 0:7
            binary = decimal_to_binary_state(n, n_qubits)
            @test binary_state_to_decimal(binary) == n
        end
    end

    @testset "Mode Cluster Access" begin
        alg = PseudoSite(2, 7)
        cluster1 = get_mode_cluster(alg, 1)
        @test length(cluster1) == 3
        @test all(ITensors.dim(s) == 2 for s in cluster1)
        cluster2 = get_mode_cluster(alg, 2)
        @test length(cluster2) == 3
        @test cluster1 != cluster2
        @test_throws ArgumentError get_mode_cluster(alg, 0)
        @test_throws ArgumentError get_mode_cluster(alg, 3)
    end

    @testset "Quantics Number Operator" begin
        n_qubits = 3
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:n_qubits]
        n_op = number_op_quantics(sites)
        for n in 0:7
            binary_state = decimal_to_binary_state(n, n_qubits)
            psi = ITensorMPS.productMPS(sites, binary_state)
            n_psi = ITensors.apply(n_op, psi)
            expectation = real(ITensorMPS.inner(psi, n_psi))
            @test abs(expectation - n) < 1e-10
        end
    end

    @testset "Quantics Creation Operator" begin
        n_qubits = 3
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:n_qubits]
        a_dag = create_op_quantics(sites)
        for n in 0:6
            binary_n = decimal_to_binary_state(n, n_qubits)
            binary_n_plus_1 = decimal_to_binary_state(n + 1, n_qubits)
            psi_n = ITensorMPS.productMPS(sites, binary_n)
            psi_n_plus_1 = ITensorMPS.productMPS(sites, binary_n_plus_1)
            adag_psi = ITensors.apply(a_dag, psi_n)
            ITensorMPS.normalize!(adag_psi)
            overlap = abs(ITensorMPS.inner(psi_n_plus_1, adag_psi))
            @test abs(overlap - 1.0) < 1e-10
        end
    end

    @testset "Quantics Annihilation Operator" begin
        n_qubits = 3
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:n_qubits]
        a = destroy_op_quantics(sites)
        for n in 1:7
            binary_n = decimal_to_binary_state(n, n_qubits)
            binary_n_minus_1 = decimal_to_binary_state(n - 1, n_qubits)
            psi_n = ITensorMPS.productMPS(sites, binary_n)
            psi_n_minus_1 = ITensorMPS.productMPS(sites, binary_n_minus_1)
            a_psi = ITensors.apply(a, psi_n)
            ITensorMPS.normalize!(a_psi)
            overlap = abs(ITensorMPS.inner(psi_n_minus_1, a_psi))
            @test abs(overlap - 1.0) < 1e-10
        end
        binary_0 = decimal_to_binary_state(0, n_qubits)
        psi_0 = ITensorMPS.productMPS(sites, binary_0)
        a_psi_0 = ITensors.apply(a, psi_0)
        @test ITensorMPS.norm(a_psi_0) < 1e-10
    end

    @testset "Bosonic Commutation Relations (Quantics)" begin
        n_qubits = 3
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:n_qubits]
        a = destroy_op_quantics(sites)
        a_dag = create_op_quantics(sites)
        for n in 0:5
            binary_n = decimal_to_binary_state(n, n_qubits)
            psi_n = ITensorMPS.productMPS(sites, binary_n)
            temp1 = ITensors.apply(a, psi_n)
            adag_a_psi = ITensors.apply(a_dag, temp1)
            exp1 = real(ITensorMPS.inner(psi_n, adag_a_psi))
            temp2 = ITensors.apply(a_dag, psi_n)
            a_adag_psi = ITensors.apply(a, temp2)
            exp2 = real(ITensorMPS.inner(psi_n, a_adag_psi))
            commutator = exp2 - exp1
            @test abs(commutator - 1.0) < 1e-10
        end
    end

    @testset "Product State BMPS (PseudoSite)" begin
        alg = PseudoSite(2, 7)
        vac = vacuumstate(alg.sites, alg)
        @test vac isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test length(vac) == 6
        @test abs(norm(vac) - 1.0) < 1e-10
        psi = BMPS(alg.sites, [1, 2], alg)
        @test psi isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test abs(norm(psi) - 1.0) < 1e-10
    end

    @testset "Random BMPS (PseudoSite)" begin
        alg = PseudoSite(2, 7)
        psi = random_bmps(alg.sites, alg; linkdims=4)
        @test psi isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test length(psi) == 6
        @test maxlinkdim(psi) > 0
    end

    @testset "Coherent State (PseudoSite)" begin
        alg = PseudoSite(1, 7)
        α = 1.0
        psi_coherent = coherentstate(alg.sites, α, alg)
        @test psi_coherent isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test abs(norm(psi_coherent) - 1.0) < 1e-6
        n_exp = expect_photon_number(psi_coherent, 1)
        @test abs(n_exp - abs2(α)) < 0.5
    end

    @testset "Quantics Hamiltonians" begin
        alg = PseudoSite(2, 7)
        H_harm = harmonic_chain(alg.sites, alg; ω=1.0)
        @test H_harm isa BMPO{<:ITensorMPS.MPO,<:PseudoSite}
        @test length(H_harm) == 6
        H_kerr = kerr(alg.sites, alg; ω=1.0, χ=0.1)
        @test H_kerr isa BMPO{<:ITensorMPS.MPO,<:PseudoSite}
        @test length(H_kerr) == 6
    end

    @testset "DMRG (PseudoSite)" begin
        alg = PseudoSite(1, 3)
        H = harmonic_chain(alg.sites, alg; ω=1.0)
        psi0 = random_bmps(alg.sites, alg; linkdims=2)
        energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=5, maxdim=10, cutoff=1e-10)
        @test energy isa Real
        @test psi_gs isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test abs(energy) < 1.0
    end
    
    @testset "TEBD (PseudoSite)" begin
        alg = PseudoSite(1, 3)
        psi = vacuumstate(alg.sites, alg)
        id_gate = ITensors.op("Id", alg.sites[1])
        psi_evolved = tebd(psi, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test abs(ITensorMPS.norm(psi_evolved) - 1.0) < 1e-10
    end
    
    @testset "TDVP (PseudoSite)" begin
        alg = PseudoSite(1, 3)
        H = harmonic_chain(alg.sites, alg; ω=1.0)
        psi = random_bmps(alg.sites, alg; linkdims=2)
        ITensorMPS.normalize!(psi.mps)
        dt = 0.01
        psi_evolved = Mabs.tdvp(psi, H, dt; cutoff=1e-8)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        @test psi_evolved !== psi
    end
    
    @testset "BMPS Operations (PseudoSite)" begin
        alg = PseudoSite(1, 3)
        psi1 = random_bmps(alg.sites, alg; linkdims=2)
        psi2 = random_bmps(alg.sites, alg; linkdims=2)
        psi_sum = psi1 + psi2
        @test psi_sum isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
        overlap = ITensorMPS.inner(psi1, psi2)
        @test overlap isa Number
        @test isfinite(overlap)
        ITensorMPS.normalize!(psi1)
        @test abs(ITensorMPS.norm(psi1) - 1.0) < 1e-10
        ITensorMPS.orthogonalize!(psi1, 1)
        @test psi1 isa BMPS{<:ITensorMPS.MPS,<:PseudoSite}
    end
    
    @testset "Displacement Operator (Quantics)" begin
        n_qubits = 3
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:n_qubits]
        α = 0.5
        D = displace_op_quantics(sites, α)
        @test D isa ITensors.ITensor
        binary_0 = decimal_to_binary_state(0, n_qubits)
        psi_vac = ITensorMPS.productMPS(sites, binary_0)
        D_psi = ITensors.apply(D, psi_vac)
        ITensorMPS.normalize!(D_psi)
        @test abs(norm(D_psi) - 1.0) < 1e-8
    end
    
    @testset "Squeeze Operator (Quantics)" begin
        n_qubits = 3
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:n_qubits]
        ξ = 0.3
        S = squeeze_op_quantics(sites, ξ)
        @test S isa ITensors.ITensor
        binary_0 = decimal_to_binary_state(0, n_qubits)
        psi_vac = ITensorMPS.productMPS(sites, binary_0)
        S_psi = ITensors.apply(S, psi_vac)
        ITensorMPS.normalize!(S_psi)
        @test abs(norm(S_psi) - 1.0) < 1e-6
    end
    
    @testset "Comparison: Truncated vs PseudoSite" begin
        max_occ = 7
        sites_trunc = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        psi_trunc = BMPS(sites_trunc, [3], Truncated())
        alg_ps = PseudoSite(1, max_occ)
        psi_ps = BMPS(alg_ps.sites, [2], alg_ps)
        @test abs(norm(psi_trunc) - 1.0) < 1e-10
        @test abs(norm(psi_ps) - 1.0) < 1e-10
        n_op_trunc = number(sites_trunc[1])
        n_psi_trunc = ITensors.apply(n_op_trunc, psi_trunc.mps)
        n_exp_trunc = real(ITensorMPS.inner(psi_trunc.mps, n_psi_trunc))
        n_exp_ps = expect_photon_number(psi_ps, 1)
        @test abs(n_exp_trunc - 2.0) < 1e-10
        @test abs(n_exp_ps - 2.0) < 1e-10
    end
    
    @testset "Error Handling (PseudoSite)" begin
        alg1 = PseudoSite(1, 7)
        alg2 = PseudoSite(1, 7)
        psi1 = random_bmps(alg1.sites, alg1)
        psi2 = random_bmps(alg2.sites, alg2)
        @test_throws ArgumentError psi1 + psi2
        @test_throws ArgumentError BMPS(alg1.sites, [10], alg1)
        @test_throws ArgumentError BMPS(alg1.sites, [1, 2], alg1)
    end

    @testset "Two-mode hopping basics" begin
        alg = PseudoSite(2, 3)
        ω = 1.0
        J = 0.5
        H = harmonic_chain(alg.sites, alg; ω=ω, J=J)
        @test H isa BMPO{<:ITensorMPS.MPO,<:PseudoSite}
        @test length(H) == 4
        psi_10 = BMPS(alg.sites, [1, 0], alg)
        psi_01 = BMPS(alg.sites, [0, 1], alg)
        H_psi_10 = ITensors.apply(H.mpo, psi_10.mps)
        overlap = abs(ITensorMPS.inner(psi_01.mps, H_psi_10))
        @test overlap > 0.1
    end
    
    @testset "Energy conservation with hopping" begin
        alg = PseudoSite(2, 7)
        ω = 2.0
        J = 0.3
        H = harmonic_chain(alg.sites, alg; ω=ω, J=J)
        psi = BMPS(alg.sites, [2, 2], alg)
        ITensorMPS.normalize!(psi.mps)
        H_psi = ITensors.apply(H.mpo, psi.mps)
        energy = real(ITensorMPS.inner(psi.mps, H_psi))
        @test abs(energy - 4 * ω) < 0.5
    end
    
    @testset "Compare with Truncated algorithm" begin
        max_occ = 3
        n_modes = 2
        ω = 1.0
        J = 0.2
        sites_trunc = ITensors.siteinds("Boson", n_modes; dim=max_occ+1)
        H_trunc = harmonic_chain(sites_trunc; ω=ω, J=J)
        psi0_trunc = random_bmps(sites_trunc, Truncated(); linkdims=4)
        energy_trunc, psi_gs_trunc = Mabs.dmrg(
            H_trunc, psi0_trunc; 
            nsweeps=10, 
            maxdim=20, 
            cutoff=1e-10
        )
        alg_ps = PseudoSite(n_modes, max_occ)
        H_ps = harmonic_chain(alg_ps.sites, alg_ps; ω=ω, J=J)
        psi0_ps = random_bmps(alg_ps.sites, alg_ps; linkdims=4)
        energy_ps, psi_gs_ps = Mabs.dmrg(
            H_ps, psi0_ps;
            nsweeps=10,
            maxdim=20,
            cutoff=1e-10
        )
        @test abs(energy_trunc - energy_ps) < 0.1
    end

    @testset "TDVP sanity check" begin
        alg = PseudoSite(1, 3)
        sites = alg.sites
        H_simple = harmonic_chain(sites, alg; ω=1.0, J=0.0)
        psi_simple = BMPS(sites, [1], alg)
        ITensorMPS.normalize!(psi_simple.mps)
        psi_evolved = Mabs.tdvp(psi_simple, H_simple, 0.1; cutoff=1e-12)
        @test abs(ITensorMPS.norm(psi_evolved.mps) - 1.0) < 1e-6
        @test psi_evolved isa BMPS
    end

    @testset "Hopping MPO verification" begin
        alg = PseudoSite(2, 3)
        J = 1.0
        H = harmonic_chain(alg.sites, alg; ω=0.0, J=J)
        psi_20 = BMPS(alg.sites, [2, 0], alg)
        psi_11 = BMPS(alg.sites, [1, 1], alg)
        ITensorMPS.normalize!(psi_20.mps)
        ITensorMPS.normalize!(psi_11.mps)
        H_psi_20 = ITensors.apply(H.mpo, psi_20.mps)
        overlap_11 = abs(ITensorMPS.inner(psi_11.mps, H_psi_20))
        @test overlap_11 > 0.5
        energy_20 = real(ITensorMPS.inner(psi_20.mps, H_psi_20))
        @test abs(energy_20) < 0.1
    end

    @testset "Hopping Hamiltonian verification" begin
        alg = PseudoSite(2, 3)
        ω = 0.0
        J = 1.0
        H = harmonic_chain(alg.sites, alg; ω=ω, J=J)
        psi_10 = BMPS(alg.sites, [1, 0], alg)
        psi_01 = BMPS(alg.sites, [0, 1], alg)
        ITensorMPS.normalize!(psi_10.mps)
        ITensorMPS.normalize!(psi_01.mps)
        H_psi_01 = ITensors.apply(H.mpo, psi_01.mps)
        matrix_element_01 = ITensorMPS.inner(psi_10.mps, H_psi_01)
        @test abs(abs(matrix_element_01) - J) < 0.3
        psi_20 = BMPS(alg.sites, [2, 0], alg)
        psi_11 = BMPS(alg.sites, [1, 1], alg)
        ITensorMPS.normalize!(psi_20.mps)
        ITensorMPS.normalize!(psi_11.mps)
        H_psi_20 = ITensors.apply(H.mpo, psi_20.mps)
        overlap_11 = abs(ITensorMPS.inner(psi_11.mps, H_psi_20))
        @test overlap_11 > 0.8
        energy_10 = real(ITensorMPS.inner(psi_10.mps, ITensors.apply(H.mpo, psi_10.mps)))
        energy_01 = real(ITensorMPS.inner(psi_01.mps, ITensors.apply(H.mpo, psi_01.mps)))
        energy_20 = real(ITensorMPS.inner(psi_20.mps, H_psi_20))
        @test abs(energy_10) < 0.1
        @test abs(energy_01) < 0.1
        @test abs(energy_20) < 0.1
        ITensorMPS.normalize!(H_psi_20)
        overlap_with_11 = abs(ITensorMPS.inner(psi_11.mps, H_psi_20))
        @test overlap_with_11 > 0.5
        H_psi_11 = ITensors.apply(H.mpo, psi_11.mps)
        overlap_20 = abs(ITensorMPS.inner(psi_20.mps, H_psi_11))
        @test overlap_20 > 0.8
        psi_02 = BMPS(alg.sites, [0, 2], alg)
        ITensorMPS.normalize!(psi_02.mps)
        H_psi_02 = ITensors.apply(H.mpo, psi_02.mps)
        overlap_11_from_02 = abs(ITensorMPS.inner(psi_11.mps, H_psi_02))
        @test abs(overlap_11_from_02 - overlap_11) < 0.3
    end

    @testset "Three-mode chain" begin
        alg = PseudoSite(3, 3)
        ω = 1.0
        J = 0.5
        H = harmonic_chain(alg.sites, alg; ω=ω, J=J)
        @test H isa BMPO{<:ITensorMPS.MPO,<:PseudoSite}
        @test length(H) == 6
        psi0 = random_bmps(alg.sites, alg; linkdims=4)
        energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=5, maxdim=16, cutoff=1e-10)
        @test energy isa Real
        @test isfinite(energy)
    end
    
    @testset "Matrix elements" begin
        alg = PseudoSite(2, 3)
        J = 1.0
        H = harmonic_chain(alg.sites, alg; ω=0.0, J=J)
        psi_10 = BMPS(alg.sites, [1, 0], alg)
        psi_01 = BMPS(alg.sites, [0, 1], alg)
        ITensorMPS.normalize!(psi_10.mps)
        ITensorMPS.normalize!(psi_01.mps)
        H_psi_01 = ITensors.apply(H.mpo, psi_01.mps)
        matrix_element = ITensorMPS.inner(psi_10.mps, H_psi_01)
        @test abs(abs(matrix_element) - J) < 0.2
    end
    
    @testset "Hermiticity of hopping Hamiltonian" begin
        alg = PseudoSite(2, 3)
        H = harmonic_chain(alg.sites, alg; ω=1.0, J=0.5)
        psi1 = random_bmps(alg.sites, alg; linkdims=4)
        psi2 = random_bmps(alg.sites, alg; linkdims=4)
        ITensorMPS.normalize!(psi1.mps)
        ITensorMPS.normalize!(psi2.mps)
        H_psi2 = ITensors.apply(H.mpo, psi2.mps)
        elem_12 = ITensorMPS.inner(psi1.mps, H_psi2)
        H_psi1 = ITensors.apply(H.mpo, psi1.mps)
        elem_21 = conj(ITensorMPS.inner(psi2.mps, H_psi1))
        @test abs(elem_12 - elem_21) < 1e-6
    end
    
    @testset "Ground state with hopping" begin
        alg = PseudoSite(2, 7)
        ω = 1.0
        J = 0.3
        H = harmonic_chain(alg.sites, alg; ω=ω, J=J)
        psi0 = random_bmps(alg.sites, alg; linkdims=8)
        energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=15, maxdim=32, cutoff=1e-12)
        n1 = expect_photon_number(psi_gs, 1)
        n2 = expect_photon_number(psi_gs, 2)
        @test n1 < 0.5
        @test n2 < 0.5
        @test energy < 1.5
    end    
end


@testset "Multi-Mode Coherent States" begin
    alg = PseudoSite(2, 7)
    
    @testset "Two modes with different amplitudes" begin
        αs = [1.0, 0.5im]
        psi = coherentstate(alg.sites, αs, alg)
        @test psi isa BMPS
        @test psi.alg isa PseudoSite
        @test length(psi) == 6
        @test abs(norm(psi) - 1.0) < 1e-8
        n1_exp = Mabs.expect_photon_number(psi, 1)
        n2_exp = Mabs.expect_photon_number(psi, 2)
        @test abs(n1_exp - abs2(αs[1])) < 0.5
        @test abs(n2_exp - abs2(αs[2])) < 0.5
    end
    
    @testset "Three modes" begin
        alg3 = PseudoSite(3, 7)
        αs = [0.8, 1.2, 0.5]
        psi = coherentstate(alg3.sites, αs, alg3)
        @test psi isa BMPS
        @test length(psi) == 9
        @test abs(norm(psi) - 1.0) < 1e-8
        for mode in 1:3
            n_exp = Mabs.expect_photon_number(psi, mode)
            @test abs(n_exp - abs2(αs[mode])) < 0.5
        end
    end
    
    @testset "Single amplitude for all modes" begin
        α = 1.0
        psi = coherentstate(alg.sites, α, alg)
        @test psi isa BMPS
        @test abs(norm(psi) - 1.0) < 1e-8
        n1 = Mabs.expect_photon_number(psi, 1)
        n2 = Mabs.expect_photon_number(psi, 2)
        @test abs(n1 - abs2(α)) < 0.5
        @test abs(n2 - abs2(α)) < 0.5
    end
    
    @testset "Small amplitude" begin
        alg_small = PseudoSite(2, 3)
        αs = [0.1, 0.2]
        psi = coherentstate(alg_small.sites, αs, alg_small)
        @test psi isa BMPS
        @test abs(norm(psi) - 1.0) < 1e-8
        @test maxlinkdim(psi) <= 16
        n1 = Mabs.expect_photon_number(psi, 1)
        n2 = Mabs.expect_photon_number(psi, 2)
        @test abs(n1 - abs2(αs[1])) < 0.1
        @test abs(n2 - abs2(αs[2])) < 0.1
    end
    
    @testset "Vacuum as coherent state with α=0" begin
        psi_vac = coherentstate(alg.sites, [0.0, 0.0], alg)
        @test abs(norm(psi_vac) - 1.0) < 1e-8
        n1 = Mabs.expect_photon_number(psi_vac, 1)
        n2 = Mabs.expect_photon_number(psi_vac, 2)
        @test abs(n1) < 0.01
        @test abs(n2) < 0.01
    end
end