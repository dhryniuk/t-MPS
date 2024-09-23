include("tmps.jl")
using DelimitedFiles
using Plots



h=1.0
γ=1.0
#Jx=1.0
#Jy=0.9
#Jz=0.8
Jx=0.0
Jy=0.0
Jz=1.0#0.0

l1 = make_one_body_Lindbladian(h*sx,γ*sm)
l2 = Jx*make_two_body_Lindblad_Hamiltonian(sx,sx) + Jy*make_two_body_Lindblad_Hamiltonian(sy,sy) + Jz*make_two_body_Lindblad_Hamiltonian(sz,sz)
#l2_four = reshape(l2,4,4,4,4)

#l1_boundary = make_one_body_Lindbladian(J/2*sz,0*id)

T = 50.0
τ = 0.05
χ=20
N=2

#U = make_two_body_Lindblad(h*(sx⊗id + id⊗sx) + Jz*(sz⊗sz), γ*(sm⊗id + id⊗sm))

U2 = exp(1*τ*(l1⊗id4 + id4⊗l1 + l2))
#U1 = exp(τ*(l1⊗id4 + id4⊗l1))# + l2))
#U = exp(τ*(two_site_one_body_Lindbladian(h*sx⊗id, γ*sm⊗id) + two_site_one_body_Lindbladian(h*id⊗sx, γ*id⊗sm)))# + l2))

#display(sparse(two_site_one_body_Lindbladian(h*sx⊗id, γ*sm⊗id) + two_site_one_body_Lindbladian(h*id⊗sx, γ*id⊗sm)))

#display(U1-U2)
#error()

U2 = reshape(U2, 4,4,4,4)
U2 = permutedims(U2, (1,3,2,4))

U1 = exp(1*l1*τ)

display(U1)
#error()


mpo_odd, mpo_even = transform_into_mpos(U2)

#x = mpo_odd[2][1,:,:,1]
#display(x)
#x = reshape(x, 4,4)
#display(x)
#error()

mps = MatrixProductState(χ, N)

bond_dim = [[1]; fill(χ, N-1); [1]]
for i in 1:N
    #mps.mp[i] += 0.005*ones(ComplexF64, bond_dim[i], 4, bond_dim[i+1])
end
#mps.mp = [rand(ComplexF64, bond_dim[i], 4, bond_dim[i+1]) for i in 1:N]
mps_init = MatrixProductState(χ, N)
mps_init.mp = deepcopy(mps.mp)


mps, n = left_normalize!(mps)

"""
U1 = exp(l1*τ)
U2 = reshape(exp(l2*τ),4,4,4,4)
## [α_1, α_2, β_1, β_2]
U2 = permutedims(U2, (1,3,2,4))
## [α_1, β_1, α_2, β_2]

#U1_boundary = exp(l1_boundary*τ)

U2_half = reshape(exp(l2*τ/2),4,4,4,4)
## [α_1, α_2, β_1, β_2]
U2_half = permutedims(U2_half, (1,3,2,4))
## [α_1, β_1, α_2, β_2]

#display(U2)
#display(U2_half)
#error()


#U1 = convert(Matrix{ComplexF64},U1)
#U2 = convert(Array{ComplexF64,4},U2)

#mpo_odd, mpo_even = transform_into_mpos(U2)
#mpo = zeros(ComplexF64,1,4,4,4,4,1)
#@tensor mpo[a,x,y,u,v,b] = mpo_odd[1][a,x,y,c]*mpo_odd[2][c,u,v,b]
#mpo = reshape(mpo,4,4,4,4)
#mpo = permutedims(mpo, (1,3,2,4))
"""

#exact_mx = 0.09411764705882311
#exact_my = -0.4235294117647055
#exact_mz = -0.15294117647058847
exact_mx = 0.16494845360824723
exact_my = 0.3711340206185569
exact_mz = -0.2577319587628867

function run(mps)
    mx = []
    my = []
    mz = []
    n=1

    push!(mx, exp_val(mps,sx,n)/trace_norm(mps))
    push!(my, exp_val(mps,sy,n)/trace_norm(mps))
    push!(mz, exp_val(mps,sz,n)/trace_norm(mps))

    for i in 1:T÷τ
        mps = time_evolve(mps, χ, N, U1, U2)
        #mps = time_evolve(mps, χ, N, U, U)
        #mps = time_evolve_second_order_Trotter(mps, χ, N, U1, U2, U2_half)

        #println(exp_val(mps,sz))

        #push!(mx, exp_val(mps,sx))
        #push!(my, exp_val(mps,sy))
        #push!(mz, exp_val(mps,sz))


        push!(mx, exp_val(mps,sx,n)/trace_norm(mps))
        push!(my, exp_val(mps,sy,n)/trace_norm(mps))
        push!(mz, exp_val(mps,sz,n)/trace_norm(mps))

        println(i, ": ", exp_val(mps,sx,n)/trace_norm(mps))
    #    push!(mx, exp_val_site_N(mps,sx)/trace_norm(mps))
    #    push!(my, exp_val_site_N(mps,sy)/trace_norm(mps))
    #    push!(mz, exp_val_site_N(mps,sz)/trace_norm(mps))
    end

    push!(mx, exact_mx)
    push!(my, exact_my)
    push!(mz, exact_mz)

    push!(mx, abs(mx[end-1]-exact_mx))
    push!(my, abs(my[end-1]-exact_my))
    push!(mz, abs(mz[end-1]-exact_mz))

    writedlm("results/qubit_mx.txt", real.(mx))
    writedlm("results/qubit_my.txt", real.(my))
    writedlm("results/qubit_mz.txt", real.(mz))

    plot(real.(mx))
    return mps
end

using BenchmarkTools

converged_mps = run(mps)
#@benchmark run(mps)
#@profview run(mps)

