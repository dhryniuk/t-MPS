using LinearAlgebra
using SparseArrays
using TensorOperations



⊗(x,y) = kron(x,y)

id = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
sx = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
sy = [0.0+0.0im 0.0-1im; 0.0+1im 0.0+0.0im]
sz = [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im]
sp = (sx+1im*sy)/2
sm = (sx-1im*sy)/2

sp_id = sparse(id)
sp_sx = sparse(sx)
sp_sy = sparse(sy)
sp_sz = sparse(sz)
sp_sp = sparse(sp)
sp_sm = sparse(sm)

id4 = id⊗id


function make_one_body_Lindbladian(H, Γ)
    L_H = -1im*(H⊗id - id⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id/2 - id⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end

function make_two_body_Lindblad_Hamiltonian(A, B)
    L_H = -1im*( (A⊗id)⊗(B⊗id) - (id⊗transpose(A))⊗(id⊗transpose(B)) )
    return L_H
end

"""
function make_two_body_Lindblad_Hamiltonian(A, B)
    L_H = -1im*( (A⊗B)⊗id4 - id4⊗transpose(A⊗B) )
    return L_H
end

function two_site_one_body_Lindbladian(H, Γ)
    L_H = -1im*(H⊗id4 - id4⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id4/2 - id4⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end


function make_two_body_Lindblad(H, Γ)
    L_H = -1im*( H⊗id4 - id4⊗transpose(H) )
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id4/2 - id4⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end
"""


mutable struct MatrixProductState
    mp::Vector{Array{ComplexF64, 3}}
    MatrixProductState(mp) = new(mp)
    function MatrixProductState(χ::Int, N::Int)
        bond_dim = [[1]; fill(χ, N-1); [1]]
        mp = [zeros(ComplexF64, bond_dim[i], 4, bond_dim[i+1]) for i in 1:N]
        for i in 1:N
            mp[i][1,1,1] = 1.0
        end
        #=
        for i in 1:N # X=1 pure product state:
            mp[i][1,1,1] = 1/2
            mp[i][1,2,1] = 1/2
            mp[i][1,3,1] = 1/2
            mp[i][1,4,1] = 1/2
        end
        =#
        mps = new(mp)
        for i in 1:N-1
            push_center!(mps, i)
        end
        return mps
    end
end

function push_center!(mps::MatrixProductState, idx::Int)
    orig_shape = size(mps.mp[idx])
    ms = reshape(mps.mp[idx], (prod(orig_shape[1:2]), :))
    f = qr(ms)
    mps.mp[idx] = reshape(Array(f.Q), orig_shape[1], orig_shape[2], :)
    @tensor mps.mp[idx+1][a, c, d] := f.R[a, b] * mps.mp[idx+1][b, c, d]
    mps.mp[idx+1] /= norm(mps.mp[idx+1])
end

function left_normalize!(mps::MatrixProductState)
    for i in 1:N-1
        orig_shape = size(mps.mp[i])
        ms = reshape(mps.mp[i], (prod(orig_shape[1:2]), :))
        #println(typeof(ms))
        #error()
        f = svd(ms,full=false)
        bond_dim = min(length(f.S), size(mps.mp[i+1])[1])
        S::Matrix{Float64} = diagm(f.S)[1:bond_dim,1:bond_dim]
        #println("bond_dim=",bond_dim)
        #println(size(mps.mp[i]),size(mps.mp[i+1]))
        #println(size(f.U),size(S),size(f.Vt))
        mps.mp[i] = reshape(f.U[:,1:bond_dim], (orig_shape[1], orig_shape[2], :))
        Vt::Matrix{ComplexF64} = f.Vt[1:bond_dim,:]
        #println(size(mps.mp[i+1]),size(S),size(Vt))
        @tensor mps.mp[i+1][a, σ, d] := ( S[a, e] * Vt[e, b] ) * mps.mp[i+1][b, σ, d] #TYPE INSTABILITY
    end
    orig_shape = size(mps.mp[N])
    ms = reshape(mps.mp[N], (prod(orig_shape[1:2]), :))
    f = svd(ms,full=false)
    #println(typeof(f))
    #error()
    bond_dim = min(length(f.S), size(mps.mp[N])[1])
    S = diagm(f.S)[1:bond_dim,1:bond_dim]

    mps.mp[N] = reshape(f.U[:,1:bond_dim], (orig_shape[1], orig_shape[2], :))
    norm = S*f.Vt[1:bond_dim,:]

    #println(size(f.S),size(f.Vt))
    #println(size(mps.mp[1]),size(mps.mp[2]))
    #println("NORM=", norm)
    
    return mps, norm
end

function right_normalize!(mps::MatrixProductState)
    for i in N:-1:2
        orig_shape = size(mps.mp[i])
        ms = reshape(mps.mp[i], (:, prod(orig_shape[2:end])))
        f = svd(ms,full=false)

        bond_dim = min(length(f.S), size(mps.mp[i-1])[end])
        S::Matrix{Float64} = diagm(f.S)[1:bond_dim,1:bond_dim]

        mps.mp[i] = reshape(f.Vt[1:bond_dim,:], (:, orig_shape[2], orig_shape[3]))
        U::Matrix{ComplexF64} = f.U[:,1:bond_dim]
        @tensor mps.mp[i-1][a, σ, b] := mps.mp[i-1][a, σ, c] * ( U[c, d] * S[d, b] )
    end
    orig_shape = size(mps.mp[1])
    ms = reshape(mps.mp[1], (:, prod(orig_shape[2:end])))
    f = svd(ms,full=false)

    bond_dim = min(length(f.S), size(mps.mp[1])[end])
    S = diagm(f.S)[1:bond_dim,1:bond_dim]

    mps.mp[1] = reshape(f.Vt[1:bond_dim,:], (:, orig_shape[2], orig_shape[3]))
    norm = f.U*S

    return mps, norm
end

function svd_compress!(mps::MatrixProductState, trun_bond_dim::Int)
    for i in N:-1:2
        orig_shape = size(mps.mp[i])
        ms = reshape(mps.mp[i], (:, prod(orig_shape[2:end])))
        f = svd(ms,full=false)

        bond_dim = min(trun_bond_dim, length(f.S), size(mps.mp[i-1])[end])
        S::Matrix{Float64} = diagm(f.S)[1:bond_dim, 1:bond_dim]

        mps.mp[i] = reshape(f.Vt[1:bond_dim,:], (:, orig_shape[2], orig_shape[3]))
        U::Matrix{ComplexF64} = f.U[:,1:bond_dim]
        @tensor mps.mp[i-1][a, σ, b] := mps.mp[i-1][a, σ, c] * ( U[c, d] * S[d, b] )
    end
    orig_shape = size(mps.mp[1])
    ms = reshape(mps.mp[1], (:, prod(orig_shape[2:end])))
    f = svd(ms,full=false)

    bond_dim = min(trun_bond_dim, length(f.S), size(mps.mp[1])[end])
    S = diagm(f.S)[1:bond_dim, 1:bond_dim]

    mps.mp[1] = reshape(f.Vt[1:bond_dim,:], (:, orig_shape[2], orig_shape[3]))
    norm = f.U*S

    return mps, norm
end

function transform_into_mpos(U2)

    _u,s,_v = svd(reshape(U2, (16,16)))
    u = -reshape(_u,(1,4,4,16))
    v = -reshape(adjoint(_v),(16,4,4,1))
    sq_s = diagm(sqrt.(s))
    id_op = reshape(Matrix{ComplexF64}(I, 4, 4), 1, 4, 4, 1)

    mpo_odd = [id_op for _ in 1:N]
    for i in 1:2:N-1
        @tensor mpo_odd[i][a,x,y,c] := u[a,x,y,b]*sq_s[b,c]
    end
    for i in 2:2:N
        @tensor mpo_odd[i][c,x,y,a] := sq_s[c,b]*v[b,x,y,a]
    end

    mpo_even = [id_op for _ in 1:N]
    for i in 2:2:N-1
        @tensor mpo_even[i][a,x,y,c] := u[a,x,y,b]*sq_s[b,c]
    end
    for i in 3:2:N 
        @tensor mpo_even[i][c,x,y,a] := sq_s[c,b]*v[b,x,y,a]
    end

    return mpo_odd, mpo_even
end

function transform_into_mpos(U2, U2_half)

    _u,s,_v = svd(reshape(U2_half, (16,16)))
    u = reshape(_u,(1,4,4,16))
    v = reshape(adjoint(_v),(16,4,4,1))
    sq_s = diagm(sqrt.(s))
    id_op = reshape(Matrix{ComplexF64}(I, 4, 4), 1, 4, 4, 1)

    mpo_odd = [id_op for _ in 1:N]
    for i in 1:2:N-1
        @tensor mpo_odd[i][a,x,y,c] := u[a,x,y,b]*sq_s[b,c]
    end
    for i in 2:2:N
        @tensor mpo_odd[i][c,x,y,a] := sq_s[c,b]*v[b,x,y,a]
    end

    _u,s,_v = svd(reshape(U2, (16,16)))
    u = reshape(_u,(1,4,4,16))
    v = reshape(adjoint(_v),(16,4,4,1))
    sq_s = diagm(sqrt.(s))
    id_op = reshape(Matrix{ComplexF64}(I, 4, 4), 1, 4, 4, 1)

    mpo_even = [id_op for _ in 1:N]
    for i in 2:2:N-1
        @tensor mpo_even[i][a,x,y,c] := u[a,x,y,b]*sq_s[b,c]
    end
    for i in 3:2:N 
        @tensor mpo_even[i][c,x,y,a] := sq_s[c,b]*v[b,x,y,a]
    end

    return mpo_odd, mpo_even
end

function apply(mpo, mps::MatrixProductState)
    mp = mps.mp
    Nmp::Vector{Array{ComplexF64,5}} = [zeros(ComplexF64, (size(mp[i])[1], size(mpo[i])[1], 4, size(mp[i])[3], size(mpo[i])[4])) for i in 1:N]
    mp2 = Array{Array}(undef,N)
    for i in 1:N
        @tensor Nmp[i][a,a',x',b,b'] = mp[i][a,x,b]*mpo[i][a',x',x,b'] #?
        #@tensor Nmp[i][a,a',x',b,b'] = mp[i][a,x,b]*mpo[i][a',x,x',b']
        org_dims = size(Nmp[i])
        mp2[i]=reshape(Nmp[i], org_dims[1]*org_dims[2], org_dims[3], org_dims[4]*org_dims[5])
    end
    mps.mp=mp2
    return mps
end

function time_evolve(mps::MatrixProductState, χ::Int, N::Int, U1, U2)
    mp = mps.mp

    mps2 = MatrixProductState(χ, N)
    mps2.mp = deepcopy(mps.mp)
    mp2 = mps2.mp

    ### U1:
    #for i in 1:N
    #    @tensor mp2[i][a,x',b] = mp[i][a,x,b]*U1[x',x]
    #end
    #mps2, _ = left_normalize!(mps2)
    #svd_compress!(mps2,χ)

    ### U2:
    mpo_odd, mpo_even = transform_into_mpos(U2)

    #=
    display(mpo_odd[1][1,:,:,1])
    display(mpo_odd[2][1,:,:,1])
    error()

    trunc_mpo_odd = [mpo_odd[i][1,:,:,1] for i in 1:2]

    display(trunc_mpo_odd[1][1,:,:,1])
    display(trunc_mpo_odd[2][1,:,:,1])
    error()
    =#

    #mpo_odd[1] = mpo_odd[1][1,:,:,1]
    #mpo_odd[2] = mpo_odd[2][1,:,:,1]

    #display(mpo_odd[:][1,:,:,1])
    #error()

    mps2 = apply(mpo_odd, mps2)
    mps2 = apply(mpo_even, mps2)
    mps2, _ = left_normalize!(mps2)
    svd_compress!(mps2,χ)

    return mps2
end

#"""
function ST1_decomp(U)

    _u,s,_v = svd(reshape(U, (16,16)))
    #display(s)
    #error()
    u = reshape(_u,(1,4,4,16))
    v = reshape(adjoint(_v),(16,4,4,1))
    sq_s = diagm(sqrt.(s))
    id_op = reshape(Matrix{ComplexF64}(I, 4, 4), 1, 4, 4, 1)

    mpo_odd = [id_op for _ in 1:N]
    for i in 1:2:N-1
        @tensor mpo_odd[i][a,x,y,c] := u[a,x,y,b]*sq_s[b,c]
    end
    for i in 2:2:N
        @tensor mpo_odd[i][c,x,y,a] := sq_s[c,b]*v[b,x,y,a]
    end

    mpo_even = [id_op for _ in 1:N]
    for i in 2:2:N-1
        @tensor mpo_even[i][a,x,y,c] := u[a,x,y,b]*sq_s[b,c]
    end
    for i in 3:2:N 
        @tensor mpo_even[i][c,x,y,a] := sq_s[c,b]*v[b,x,y,a]
    end

    return mpo_odd, mpo_even
end

#U_odd_mpo, U_even_mpo = ST1_decomp(U1,U2)
#U_odd_mpo, U_even_mpo = ST2_decomp(U1,U2)
#"""

"""
function time_evolve(mps::MatrixProductState, χ::Int, N::Int, U)
#function time_evolve(mps::MatrixProductState, χ::Int, N::Int, U1, U2)
    mp = mps.mp

    mps2 = MatrixProductState(χ, N)
    mps2.mp = deepcopy(mps.mp)
    mp2 = mps2.mp

    mpo_odd, mpo_even = ST1_decomp(U)

    #mpo_odd, mpo_even = transform_into_mpos(U)

    ### U1:
    #for i in 1:N
    #    @tensor mp2[i][a,x',b] = mp[i][a,x,b]*U1[x',x]
    #end

    mps2 = apply(mpo_odd,mps2)
    mps2 = apply(mpo_even,mps2)
    mps2, _ = left_normalize!(mps2)
    svd_compress!(mps2,χ)

    return mps2
end

function time_evolve_second_order_Trotter(mps::MatrixProductState, χ::Int, N::Int, U1, U2, U2_half)
    mp = mps.mp

    mps2 = MatrixProductState(χ, N)
    mps2.mp = deepcopy(mps.mp)
    mp2 = mps2.mp

    ### U1:
    for i in 1:N
        @tensor mp2[i][a,x',b] = mp[i][a,x,b]*U1[x',x]
    end
    #mps2, _ = left_normalize!(mps2)
    #svd_compress!(mps2,χ)

    ### U2:
    #mpo_odd, mpo_even = transform_into_mpos(U1, U2, U2_half)

    mps2 = apply(mpo_odd,mps2)
    #mps2, _ = left_normalize!(mps2)
    #svd_compress!(mps2,χ)
    #mps2 = apply(mpo_odd,mps2)

    mps2 = apply(mpo_even,mps2)
    #mps2, _ = left_normalize!(mps2)
    #svd_compress!(mps2,χ)

    mps2 = apply(mpo_odd,mps2)
    mps2, _ = left_normalize!(mps2)
    svd_compress!(mps2,χ)

    return mps2
end
"""

"""
function time_evolve(mps::MatrixProductState, χ::Int, N::Int, U1, U1_boundary, U2)
#function time_evolve(mps::MatrixProductState, χ::Int, N::Int, U1, U2)
    mp = mps.mp

    mps2 = MatrixProductState(χ, N)
    mps2.mp = deepcopy(mps.mp)
    mp2 = mps2.mp

    ### U1:
    #for i in 1:N
    #    @tensor mp2[i][a,x',b] = mp[i][a,x,b]*U1[x',x]
    #end
    
    ### U1 boundary:
    @tensor mp2[1][a,x',b] := mp2[1][a,x,b]*U1_boundary[x',x]
    @tensor mp2[N][a,x',b] := mp2[N][a,x,b]*U1_boundary[x',x]


    mpo_odd, mpo_even = transform_into_mpos(U2)

    mps2 = apply(mpo_odd,mps2)
    mps2 = apply(mpo_even,mps2)

    mps2, _ = left_normalize!(mps2)
    
    svd_compress!(mps2,χ)
    return mps2
end
"""

"""
function time_evolve(mps::MatrixProductState, χ::Int, N::Int, U1, U2, U2_half)
    mp = mps.mp

    mps2 = MatrixProductState(χ, N)
    mps2.mp = deepcopy(mps.mp)
    mp2 = mps2.mp

    ### U1:
    for i in 1:N
        @tensor mp2[i][a,x',b] = mp[i][a,x,b]*U1[x',x]
    end

    mpo_odd, mpo_even = transform_into_mpos(U2, U2_half)

    mps2 = apply(mpo_odd,mps2)
    mps2 = apply(mpo_even,mps2)
    mps2 = apply(mpo_odd,mps2)

    mps2, _ = left_normalize!(mps2)
    
    svd_compress!(mps2,χ)

    return mps2
end
"""

function exp_val(mps::MatrixProductState, op::Matrix, site::Int)
    #N = length(mps.mp)
    if site < 1 || site > N
        throw(ArgumentError("Site must be between 1 and $N"))
    end

    E = Matrix{ComplexF64}(I, (1, 1))

    for i in 1:N
        ms = mps.mp[i]
        orig_shape = size(ms)
        ms = reshape(ms, (orig_shape[1], 2, 2, orig_shape[3]))

        if i == site
            @tensor E[a,c] := E[a,b] * ms[b,u,v,c] * op[u,v]
        else
            @tensor E[a,c] := E[a,b] * ms[b,u,u,c]
        end
    end

    return tr(E)
end


function exp_val(mps::MatrixProductState, op::Matrix)
    ev = zero(ComplexF64)
    for i in 1:N
        ev += exp_val(mps, op, i)
    end
    return ev/trace_norm(mps)/N
end

function exp_val_site_1(mps::MatrixProductState, op::Matrix)
    E = Matrix{ComplexF64}(I,(1,1))
    ms = mps.mp[1]
    orig_shape = size(ms)
    ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))
    @tensor E[a,c] := E[a,b]*ms[b,u,v,c]*op[u,v]
    for i in 2:N
        ms = mps.mp[i]
        orig_shape = size(ms)
        ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))
        @tensor E[a,c] := E[a,b]*ms[b,u,u,c]
    end
    return tr(E)
end

function exp_val_site_N(mps::MatrixProductState, op::Matrix)
    E = Matrix{ComplexF64}(I,(1,1))
    for i in 1:N-1
        ms = mps.mp[i]
        orig_shape = size(ms)
        ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))
        @tensor E[a,c] := E[a,b]*ms[b,u,u,c]
    end
    ms = mps.mp[N]
    orig_shape = size(ms)
    ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))
    @tensor E[a,c] := E[a,b]*ms[b,u,v,c]*op[u,v]
    return tr(E)
end

function trace_norm(mps::MatrixProductState)
    E = Matrix{ComplexF64}(I,(1,1))
    for i in 1:N
        ms = mps.mp[i]
        orig_shape = size(ms)
        ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))
        @tensor E[a,c] := E[a,b]*ms[b,u,u,c]
    end
    return tr(E)
end

function compute_purity(mps::MatrixProductState)
    N = length(mps.mp)
    χs = [size(mps.mp[i], 1) for i in 1:N]
    χs = [1; χs; 1]


    ms = mps.mp[1]
    orig_shape = size(ms)
    ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))

    #println(size(ms))

    B = zeros(ComplexF64, (orig_shape[1],orig_shape[3],orig_shape[1],orig_shape[3]))
    @tensor B[a,b,p,q] = ms[a,x,y,b]*ms[p,y,x,q]

    #B = zeros(ComplexF64,1,1,1,1)
    for i in 2:N
        ms = mps.mp[i]
        orig_shape = size(ms)
        ms=reshape(ms,(orig_shape[1],2,2,orig_shape[3]))
        #println(size(B))
        #println(size(ms))
        @tensor B[a,b,p,q] := B[a,c,p,r]*ms[c,x,y,b]*ms[r,y,x,q]
    end
    
    purity = @tensor B[a,a,b,b]

    #println(mps.mp[1])
    #println(mps.mp[2])
    #println(trace_norm(mps))
    #println(exp_val(mps,sz,1)/trace_norm(mps))
    #error()

    #println(trace_norm(mps))
    #println(size(B))
    #println(B)
    #println(B./trace_norm(mps)^2 )
    #println(purity)

    #error()
    return real(purity)
end