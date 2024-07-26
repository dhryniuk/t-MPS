include("tmps.jl")
using DelimitedFiles



h=1.0
γ=1.0
J=0.5

l1 = make_one_body_Lindbladian(h*sx,γ*sm)
l2 = J*make_two_body_Lindblad_Hamiltonian(sz,sz)
l2_four = reshape(l2,4,4,4,4)


T = 1
τ = 0.001
χ=2
N=4


mps = MatrixProductState(χ, N)

bond_dim = [[1]; fill(χ, N-1); [1]]
for i in 1:N
    #mps.mp[i] += 0.005*ones(ComplexF64, bond_dim[i], 4, bond_dim[i+1])
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

U2_half = reshape(exp(l2*τ/2),4,4,4,4)
## [α_1, α_2, β_1, β_2]
U2_half = permutedims(U2_half, (1,3,2,4))
## [α_1, β_1, α_2, β_2]


#U1 = convert(Matrix{ComplexF64},U1)
#U2 = convert(Array{ComplexF64,4},U2)

#mpo_odd, mpo_even = transform_into_mpos(U2)
#mpo = zeros(ComplexF64,1,4,4,4,4,1)
#@tensor mpo[a,x,y,u,v,b] = mpo_odd[1][a,x,y,c]*mpo_odd[2][c,u,v,b]
#mpo = reshape(mpo,4,4,4,4)
#mpo = permutedims(mpo, (1,3,2,4))

function run(mps)
    mx = []
    my = []
    mz = []
    n=2
    for i in 1:T÷τ
        mps = time_evolve(mps, χ, N, U1, U2)
        #global mps = time_evolve(mps, χ, N, U1, U2, U2_half)

        println(exp_val(mps,sz))

        #push!(mx, exp_val(mps,sx))
        #push!(my, exp_val(mps,sy))
        #push!(mz, exp_val(mps,sz))


        #push!(mx, exp_val(mps,sx,n)/trace_norm(mps))
        #push!(my, exp_val(mps,sy,n)/trace_norm(mps))
        #push!(mz, exp_val(mps,sz,n)/trace_norm(mps))
    #    push!(mx, exp_val_site_N(mps,sx)/trace_norm(mps))
    #    push!(my, exp_val_site_N(mps,sy)/trace_norm(mps))
    #    push!(mz, exp_val_site_N(mps,sz)/trace_norm(mps))
    end
    #writedlm("mx.txt", real.(mx))
    #writedlm("my.txt", real.(my))
    #writedlm("mz.txt", real.(mz))
end

#run(mps)
@profview run(mps)