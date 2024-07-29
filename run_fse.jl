include("tmps.jl")
using DelimitedFiles
using Plots
using BenchmarkTools



h=1.0#1.0
γ=1.0
J=0.5#0.5

l1 = make_one_body_Lindbladian(h*sx,γ*sm)
l2 = J*make_two_body_Lindblad_Hamiltonian(sz,sz)
l2_four = reshape(l2,4,4,4,4)

l1_boundary = make_one_body_Lindbladian(J/2*sz,0*id)

T = 10
τ = 0.01
χ=10
N=4


mps = MatrixProductState(χ, N)

bond_dim = [[1]; fill(χ, N-1); [1]]
mps_init = MatrixProductState(χ, N)
mps_init.mp = deepcopy(mps.mp)


mps, n = left_normalize!(mps)

U1 = exp(l1*τ)
U2 = reshape(exp(l2*τ),4,4,4,4)
## [α_1, α_2, β_1, β_2]
U2 = permutedims(U2, (1,3,2,4))
## [α_1, β_1, α_2, β_2]

U1_boundary = exp(l1_boundary*τ)

U2_half = reshape(exp(l2*τ/2),4,4,4,4)
## [α_1, α_2, β_1, β_2]
U2_half = permutedims(U2_half, (1,3,2,4))
## [α_1, β_1, α_2, β_2]

println(mps.mp[1])
println(mps.mp[2])
println(trace_norm(mps))
println(exp_val(mps,sz,1)/trace_norm(mps))

function run(mps)
    mx = []
    my = []
    mz = []
    n=N÷2
    for i in 1:T÷τ
        mps = time_evolve(mps, χ, N, U1, U1_boundary, U2)
        #mps = time_evolve(mps, χ, N, U1, U2, U2_half)

        push!(mx, exp_val(mps,sx,n)/trace_norm(mps))
        push!(my, exp_val(mps,sy,n)/trace_norm(mps))
        push!(mz, exp_val(mps,sz,n)/trace_norm(mps))

        println(exp_val(mps,sz,n)/trace_norm(mps))
        p = real( compute_purity(mps)/trace_norm(mps)^2 )
        S2 = -log2(p)/N
        println(p, " ; ", S2)
        #sleep(0.2)
    end
    writedlm("mx.txt", real.(mx))
    writedlm("my.txt", real.(my))
    writedlm("mz.txt", real.(mz))

    plot(real.(mx))
    return mps
end


converged_mps = run(mps)
#@benchmark run(mps)
#@profview run(mps)

#println(exp_val(converged_mps,sx,N÷2)/trace_norm(converged_mps))