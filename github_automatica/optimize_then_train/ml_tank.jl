using PyPlot
using JuMP
using Ipopt
using Flux
using Random, Distributions
import Distributions: Uniform
using JLD

N=20                                            #Total steps in Opt
N_sim = 100
h=3                                             #time step  0.002h
Tf=h*N                                          #Process time (h)
T=collect(1:N)                                  #a set containing all steps
Tm=collect(1:(N-1))                     	#a set containing all steps except the last step
mT=collect(2:(N))      # backoff parameters

xs = [14, 14, 14.2, 21.3];
us = [43.4, 35.4];
vmin = [-43.4,-35.4];
vmax = [16.6, 24.6];
zmin = [-6.5, -6.5, -10.7, -16.8]
zmax = [14, 14, 13.8, 6.7]

g=981
A=[50.27,50.27, 28.27,28.27]
a=[0.233, 0.242, 0.127, 0.127]
gama = [0.4, 0.4]
α = 1000

Random.seed!(0)

l1 = 10

ntrain = 200
z1_train = rand(Uniform(zmin[1],zmax[1]),ntrain)
z2_train = rand(Uniform(zmin[2],zmax[2]),ntrain)
z3_train = rand(Uniform(zmin[3],zmax[3]),ntrain)
z4_train = rand(Uniform(zmin[4],zmax[4]),ntrain)

d = load("test_data_1k_v2.jld")
z1_test = d["z1"]
z2_test = d["z2"]
z3_test = d["z3"]
z4_test = d["z4"]
ntest = length(z1_test)





function normalizeZ(z,ind=0)
    if ind==0
        return (z.-zmin)./(zmax-zmin)
    else
        return (z.-zmin[ind])./(zmax[ind]-zmin[ind])
    end
end
function denormalizeZ(z,ind=0)
    if ind==0
        return zmin .+ z.*(zmax-zmin)
    else
        return zmin[ind] .+ z.*(zmax[ind]-zmin[ind])
    end
end

function normalizeV(v,ind=0)
    if ind==0
        return (v.-   (vmin .+ vmax) ./ 2      )./((vmax-vmin)./2)
    else
        return (v[ind].-   (vmin[ind] .+ vmax[ind]) ./ 2      )./((vmax[ind]-vmin[ind])./2)
    end
end



function denormalizeV(v,ind=0)
    if ind==0
        return (vmax .+ vmin)/2 .+ (vmax .- vmin) ./ 2 .* v
    else
        return (vmax[ind] .+ vmin[ind])/2 .+ (vmax[ind] .- vmin[ind]) ./ 2 .* v[ind]
    end
end


function step_model(z, v)
    zold = copy(z)
    z1 = zold[1] + h*(- a[1]/A[1]*sqrt(2*g*(zold[1]+xs[1])) + a[3]/A[1]*sqrt(2*g*(zold[3]+xs[3])) + gama[1]/A[1]*(v[1]+us[1]))
    z2 = zold[2] + h*(- a[2]/A[2]*sqrt(2*g*(zold[2]+xs[2])) + a[4]/A[2]*sqrt(2*g*(zold[4]+xs[4])) + gama[2]/A[2]*(v[2]+us[2]))
    z3 = zold[3] + h*(- a[3]/A[3]*sqrt(2*g*(zold[3]+xs[3]))   			              + (1-gama[2])/A[3]*(v[2]+us[2]))
    z4 = zold[4] + h*(- a[4]/A[4]*sqrt(2*g*(zold[4]+xs[4]))                                   + (1-gama[1])/A[4]*(v[1]+us[1]))
    return [z1,z2,z3,z4]
end


function simulation(z0, v)
    N = size(v)[2]
    nz = size(z0)[1]
    z = zeros(Float64,nz,N+1)
    z[:,1] = z0
    for t = 1:N
    	z[:,t+1] = step_model(z[:,t], v[:,t])
    end
    return z
end
z0 = [0,0,0,0]
v = zeros(Float64, 2, N)
v[1,:] .= 0
v[2,:] .= 0


# ideal model predictive control
function OptimalControl(z0)
    m = Model(with_optimizer(Ipopt.Optimizer,print_level = 0))

    @variable(m, zmin[i]<=z[i in 1:4, t in 1:(N+1)]<=zmax[i], start=0)
    @variable(m, vmin[i]<=v[i in 1:2, t in 1:N]<=vmax[i], start=0)


    @constraint(m, [i in 1:4], z[i,1] == z0[i])

    #z1 = zold[1] + h*(- a[1]/A[1]*sqrt(2*g*(zold[1]+xs[1])) + a[3]/A[1]*sqrt(2*g*(zold[3]+xs[3])) + gama[1]/A[1]*(v[1]+us[1]))
    @NLconstraint(m, [t in 1:N], z[1,t+1] - z[1,t]  == h*( - a[1]/A[1]*sqrt(2*g*(z[1,t]+xs[1])) + a[3]/A[1]*sqrt(2*g*(z[3,t]+xs[3])) + gama[1]/A[1]*(v[1,t]+us[1])))

    #z2 = zold[2] + h*(- a[2]/A[2]*sqrt(2*g*(zold[2]+xs[2])) + a[4]/A[2]*sqrt(2*g*(zold[4]+xs[4])) + gama[2]/A[2]*(v[2]+us[2]))
    @NLconstraint(m, [t in 1:N], z[2,t+1] - z[2,t]  == h*( - a[2]/A[2]*sqrt(2*g*(z[2,t]+xs[2])) + a[4]/A[2]*sqrt(2*g*(z[4,t]+xs[4])) + gama[2]/A[2]*(v[2,t]+us[2])))

    #z3 = zold[3] + h*(- a[3]/A[3]*sqrt(2*g*(zold[3]+xs[3]))                                   + (1-gama[2])/A[3]*(v[2]+us[2]))
    @NLconstraint(m, [t in 1:N], z[3,t+1] - z[3,t]  == h*(-  a[3]/A[3]*sqrt(2*g*(z[3,t]+xs[3]))                                  + (1-gama[2])/A[3]*(v[2,t]+us[2])))

    #z4 = zold[4] + h*(- a[4]/A[4]*sqrt(2*g*(zold[4]+xs[4]))                                   + (1-gama[1])/A[4]*(v[1]+us[1]))
    @NLconstraint(m, [t in 1:N], z[4,t+1] - z[4,t]  == h*(-  a[4]/A[4]*sqrt(2*g*(z[4,t]+xs[4]))                                  + (1-gama[1])/A[4]*(v[1,t]+us[1])))

    @NLconstraint(m, 6.55*z[1,N+1]^2 + 6.55*z[2,N+1]^2 + 7.92*z[3,N+1]^2 + 31.7*z[4,N+1]^2 <= α)


    @objective(m, Min, sum((z[1,t])^2 for t in 2:(N)) + sum((z[2,t])^2 for t in 2:(N))
    + 1e-2*sum((v[1,t])^2 for t in 1:N) + 1e-2*sum((v[2,t])^2 for t in 1:N) + 6.55*z[1,N+1]^2 + 6.55*z[2,N+1]^2 + 7.92*z[3,N+1]^2 + 31.7*z[4,N+1]^2)
    JuMP.optimize!(m)
    return [JuMP.value(v[1,1]),JuMP.value(v[2,1])]
end


z_train = zeros(Float64,4,ntrain*N)
v_train = zeros(Float64,2,ntrain*N)
for i = 1:ntrain
            z = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N
                v = OptimalControl(z)
                z_train[:,(i-1)*N+t] = z
                v_train[:,(i-1)*N+t] = v
                z = step_model(z, v)
            end
end

# ml = Chain(
#   Dense(4, l1, tanh),
#   Dense(l1, 2, tanh)
# )

for trial = 2:6

if trial == 2
ml = Chain(
  Dense(4, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, 2, tanh)
)
elseif trial ==	3
ml = Chain(
  Dense(4, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, 2, tanh)
)
elseif trial ==	4
ml = Chain(
  Dense(4, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, 2, tanh)
)
elseif trial == 5
ml = Chain(
  Dense(4, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, 2, tanh)
)
elseif trial == 6
ml = Chain(
  Dense(4, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, l1, tanh),
  Dense(l1, 2, tanh)
)
end

loss(x, y) = Flux.mse(ml(x),  y)   #sum((ml(x) .- y).^2)

data = [(z_train[:,i],normalizeV(v_train[:,i])) for i in 1:ntrain*N]
opt = ADAM()

ml_train_start = time()
# Flux.params() extract the parameters in the dense operation
Flux.@epochs 5000 Flux.train!(loss, Flux.params(ml), data, opt) #Flux. @epochs num run multiple epochs
println(" NN train   ", (time() - ml_train_start))


ml_test_start = time()
cost_ml_train = zeros(0)
vio_ml_train = zeros(0)
vio_percent_ml_train = zeros(ntrain)
for i = 1:ntrain
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(ml(z[:,t]))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end
            append!(cost_ml_train,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_ml_train,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_ml_train[i] >= 1e-5
                vio_percent_ml_train[i] = 1
            end
end
cost_ml_test = zeros(0)
vio_ml_test = zeros(0)
vio_percent_ml_test = zeros(0)
for i = 1:ntest
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(ml(z[:,t]))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end

            append!(cost_ml_test,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_ml_test,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_ml_test[i] >= 1e-5
                append!(vio_percent_ml_test,1)
            else
                append!(vio_percent_ml_test,0)
            end
end
# println(" NN test   ", (time() - ml_test_start)/N_sim/(ntrain+ntest))
#
# println(" ML mean cost train:    ",mean(cost_ml_train))
# println(" ML mean cost test :    ",mean(cost_ml_test))
# println(" ML vio train:    ", maximum(vio_ml_train))
# println(" ML vio test:     ", maximum(vio_ml_test))
# println(" ML vio percent in train", sum(vio_percent_ml_train)/ntrain)
# println(" ML vio percent in train", sum(vio_percent_ml_test)/ntest)

open("myfile.txt","a") do io
    println(io, " Result with trail", trial)
    println(io, " ML mean cost train:    ",mean(cost_ml_train))
    println(io, " ML mean cost test :    ",mean(cost_ml_test))
    println(io, " ML vio train:    ", maximum(vio_ml_train))
    println(io, " ML vio test:     ", maximum(vio_ml_test))
    println(io, " ML vio percent in train", sum(vio_percent_ml_train)/ntrain)
    println(io, " ML vio percent in train", sum(vio_percent_ml_test)/ntest)
end
end
