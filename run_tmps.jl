include("tmps.jl")
using DelimitedFiles



h=0.5
γ=1.0
J=1.0

l1 = make_one_body_Lindbladian(h*sx,γ*sm)
l2 = J*make_two_body_Lindblad_Hamiltonian(sz,sz)
l2_four = reshape(l2,4,4,4,4)


τ = 0.01
χ=4
N=2


mps = MatrixProductState(χ, N)

bond_dim = [[1]; fill(χ, N-1); [1]]
for i in 1:N
    mps.mp[i] += 0.005*ones(ComplexF64, bond_dim[i], 4, bond_dim[i+1])
end
#mps.mp = [rand(ComplexF64, bond_dim[i], 4, bond_dim[i+1]) for i in 1:N]
mps_init = MatrixProductState(χ, N)
mps_init.mp = deepcopy(mps.mp)


mps, n = left_normalize!(mps)

U1 = exp(l1*τ)
U2 = reshape(exp(l2*τ),4,4,4,4)
## [α_1, α_2, β_1, β_2]
U2 = permutedims(U2, (1,3,2,4))
## [α_1, β_1, α_2, β_2]


#U1 = convert(Matrix{ComplexF64},U1)
#U2 = convert(Array{ComplexF64,4},U2)

#mpo_odd, mpo_even = transform_into_mpos(U2)
#mpo = zeros(ComplexF64,1,4,4,4,4,1)
#@tensor mpo[a,x,y,u,v,b] = mpo_odd[1][a,x,y,c]*mpo_odd[2][c,u,v,b]
#mpo = reshape(mpo,4,4,4,4)
#mpo = permutedims(mpo, (1,3,2,4))

mx = []
my = []
mz = []
for i in 1:5000
    global mps = time_evolve(mps, χ, N, U1, U2)
    push!(mx, exp_val_site_1(mps,sx)/trace_norm(mps))
    push!(my, exp_val_site_1(mps,sy)/trace_norm(mps))
    push!(mz, exp_val_site_1(mps,sz)/trace_norm(mps))
end
#println()
writedlm("mx.txt", real.(mx))
writedlm("my.txt", real.(my))
writedlm("mz.txt", real.(mz))