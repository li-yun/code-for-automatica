#using PyPlot
using JuMP
using Ipopt
using Flux
using Random, Distributions
import Distributions: Uniform
using JLD

N=20                                            #Total steps in Opt
N_sim = 500
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



l1 = 10

Random.seed!(0)
ntrain = 200
z1_train = rand(Uniform(zmin[1],zmax[1]),ntrain)
z2_train = rand(Uniform(zmin[2],zmax[2]),ntrain)
z3_train = rand(Uniform(zmin[3],zmax[3]),ntrain)
z4_train = rand(Uniform(zmin[4],zmax[4]),ntrain)

ntest = 1000
d = load("test_data_1k.jld")
z1_test = d["z1"]
z2_test = d["z2"]
z3_test = d["z3"]
z4_test = d["z4"]






ntrain_add = 0
z1_train_add = zeros(ntrain_add)
z2_train_add = zeros(ntrain_add)
z3_train_add = zeros(ntrain_add)
z4_train_add = zeros(ntrain_add)

  # generate additional traning scenarios
# s = 1
# for i = 0:1
#     for j = 0:1
#         for k = 0:1
#             for l = 0:1
#                 z1_train_add[s] = zmin[1] + (zmax[1] - zmin[1])*i
#                	z2_train_add[s] = zmin[2] + (zmax[2] - zmin[2])*j
#                 z3_train_add[s] = zmin[3] + (zmax[3] - zmin[3])*k
#                 z4_train_add[s] = zmin[4] + (zmax[4] - zmin[4])*l
#
#                 global s += 1
#             end
#         end
#     end
# end


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
simulation(z0,v)




# ideal model predictive control
function OptimalControl(z0)
    m = Model(with_optimizer(Ipopt.Optimizer,print_level = 1))

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



# calculate the MPC based the optimal control input and the constraints violation
MPC_start = time() #get the system time in seconds since he epoch
cost_train = zeros(0)
vio_train = zeros(0)
vio_percent_train = zeros(0)
for i = 1:ntrain # n_train = 81
            v_trial = zeros(Float64,2,N_sim)
            z_trial = zeros(Float64,4,N_sim+1)
            z_trial[:,1] = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N_sim
                v_trial[:,t] = OptimalControl(z_trial[:,t])
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[:,t])
            end
            append!(cost_train, sum(z_trial[1,2:N_sim+1].^2 .+ z_trial[2,2:N_sim+1].^2) + sum(0.01*v_trial[1,:].^2 + 0.01*v_trial[2,:].^2))
            append!(vio_train,
                    maximum(max.(z_trial[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z_trial[1,:],0)) +
                    maximum(max.(z_trial[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z_trial[2,:],0)) +
                    maximum(max.(z_trial[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z_trial[3,:],0)) +
                    maximum(max.(z_trial[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z_trial[4,:],0))
            )
end

cost_test = zeros(0)
vio_test = zeros(0)
vio_percent_test = zeros(0)
for i = 1:ntest
    	    v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
	        z[:,1] = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
            for t = 1:N_sim
                v[:,t] = OptimalControl(z[:,t])
		        z[:,t+1] = step_model(z[:,t], v[:,t])
            end
            append!(cost_test, sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_test,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_test[i] >= 1e-5
                append!(vio_percent_test,1)
            else
                append!(vio_percent_test,0)
            end
end


println("MPC    time per step: ", (time() - MPC_start)/N_sim/(ntrain+ntest))

println("MPC mean cost train:    ",mean(cost_train))
println("MPC mean cost test :    ",mean(cost_test))
println("MPC vio train:    ",maximum(vio_train))
println("MPC vio test:     ",maximum(vio_test))
println("MPC vio percent in test", sum(vio_percent_test)/ntest)



# optimize then train method
data_start = time()
z_train = zeros(Float64,4,ntrain)
v_train = zeros(Float64,2,ntrain)
for i = 1:ntrain
            z = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            v = OptimalControl(z)
            z_train[:,i] = z
            v_train[:,i] = v
end

# z_test = zeros(Float64,4,ntest*N)
# v_test = zeros(Float64,2,ntest*N)
# for i = 1:ntest
#             z = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
#             for t = 1:N
#                 v = OptimalControl(z)
#                 z_test[:,(i-1)*N+t] = z
#                 v_test[:,(i-1)*N+t] = v
#                 z = step_model(z, v)
#             end
# end
println("ML data generation time: ",  (time() - data_start)/N/(ntrain+ntest))

#=
z_train = zeros(Float64,4,ntrain)
v_train = zeros(Float64,2,ntrain)
for i = 1:ntrain
      	    z = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            v = OptimalControl(z)
            z_train[:,i] = z
            v_train[:,i] = v
end
z_test = zeros(Float64,4,ntest)
v_test = zeros(Float64,2,ntest)
for i = 1:ntest
	    z = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
            v = OptimalControl(z)
            z_test[:,i] = z
            v_test[:,i] = v
end
=#



# for trial = 1 : 6
trial = 1
println("ML trial        ", trial)

if trial == 1
ml = Chain(
  Dense(4, l1, tanh),
  Dense(l1, 2, tanh)
)
elseif trial == 2
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
mae(x, y) = mean(abs.(ml(x).- y))

data = [(normalizeZ(z_train[:,i]),normalizeV(v_train[:,i])) for i in 1:ntrain]
opt = ADAM()

ml_train_start = time()
# Flux.params() extract the parameters in the dense operation
Flux.@epochs 5000 Flux.train!(loss, Flux.params(ml), data, opt) #Flux. @epochs num run multiple epochs
println(" NN train   ", (time() - ml_train_start))


# loss(normalizeZ(z_train), normalizeV(v_train))
# mae(normalizeZ(z_train), normalizeV(v_train))
# mean(abs.(denormalizeV(ml(normalizeZ(z_train))).- v_train))
# mean(abs.((denormalizeV(ml(normalizeZ(z_train))).- v_train)./v_train))
#
# loss(normalizeZ(z_test), normalizeV(v_test))
# mae(normalizeZ(z_test), normalizeV(v_test))
# mean(abs.(denormalizeV(ml(normalizeZ(z_test))).- v_test))


ml_test_start = time()
cost_ml_train = zeros(0)
vio_ml_train = zeros(0)
vio_percent_ml_train = zeros(ntrain)
for i = 1:ntrain
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(ml(normalizeZ(z[:,t])))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end
            append!(cost_ml_train,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_ml_train,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_ml_train[i] >= 1e-3
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
                v[:,t] = denormalizeV(ml(normalizeZ(z[:,t])))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end

            append!(cost_ml_test,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_ml_test,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_ml_test[i] >= 1e-3
                append!(vio_percent_ml_test,1)
            else
                append!(vio_percent_ml_test,0)
            end
	    #=
            figure("x1_ml")
            plot(1:(N+1), xs[1].+z[1,:], color="grey",linewidth=0.5);
            figure("x2_ml")
            plot(1:(N+1), xs[2].+z[2,:], color="grey",linewidth=0.5);
            figure("x3_ml")
            plot(1:(N+1), xs[3].+z[3,:], color="grey",linewidth=0.5);
            figure("x4_ml")
            plot(1:(N+1), xs[4].+z[4,:], color="grey",linewidth=0.5);
            figure("u1_ml")
            plot(1:(N), us[1].+v[1,:], color="grey",linewidth=0.5);
            figure("u2_ml")
            plot(1:(N), us[2].+v[2,:], color="grey",linewidth=0.5);
            figure("z_v1_ml")
            plot(z[2,1:N], v[1,:], color="grey",linewidth=0.5);
            figure("z_v2_ml")
            plot(z[2,1:N], v[2,:], color="grey",linewidth=0.5);
	    =#
end
println(" NN test   ", (time() - ml_test_start)/N_sim/(ntrain+ntest))

println(" Result with trail", trial)
println(" ML mean cost train:    ",mean(cost_ml_train))
println(" ML mean cost test :    ",mean(cost_ml_test))
println(" ML vio train:    ", maximum(vio_ml_train))
println(" ML vio test:     ", maximum(vio_ml_test))
println(" ML vio percent in train", sum(vio_percent_ml_train)/ntrain)
println(" ML vio percent in train", sum(vio_percent_ml_test)/ntest)
#
#

# =#
W_1 = [-0.103344    -0.0633228    0.0172157   0.0149421   0.00287376  0.0411244   0.0133963  -0.133743   -0.113091  -0.0962923;
-0.115449    -0.110261    -0.104771   -0.0114781   0.0153535   0.0698421   0.289195   -0.0774494   0.145404   0.130912;
 0.0676212   -0.0452765   -0.158796    0.0253688   0.0743145   0.0247819   0.222859    0.0550906   0.292291   0.0618062;
 0.00947731   0.00873047  -0.164223   -0.0125125  -0.211912    0.0592898  -0.270902    0.0767465  -0.232411  -0.158894]

W_2 =  [-0.712345   0.375706;
  0.687372   2.24052;
  1.11517    0.325917;
 -1.15804   -0.10380;
  1.23283    0.12851;
 -1.02575    0.137941;
 -1.13549    0.103082;
  1.7301     0.244558;
  0.455891  -0.443142;
  0.484054  -0.31362]

b_1 = [0.6527387072654691, 1.2231244930731933, -0.1817360949759704, 0.1825773185245452, 0.15355758401898506, 0.5217358122240028, 0.007506493832543926, 0.2373779848330991, 0.397644471764949, 0.5157039939332881]
b_2 = [0.24525605479496537, -1.695211037174189]


α = 1000
epsilon = 1.5 # constraints backoff parameter
#### Estmiate W b using opt
NS = ntrain
NS_all = ntrain + ntrain_add  # add additional scenarios into the traning set

z1_train_all = append!(z1_train, z1_train_add)
z2_train_all = append!(z2_train, z2_train_add)
z3_train_all = append!(z3_train, z3_train_add)
z4_train_all = append!(z4_train, z4_train_add)


m = Model(with_optimizer(Ipopt.Optimizer, bound_relax_factor = 1e-8,max_iter=3000))

    @variable(m, zmin[i] <=z[i in 1:4, 1:NS_all, t in 1:(N+1)] <= zmax[i], start=rand())
    @variable(m, vmin[i]<=v[i in 1:2, 1:NS_all, t in 1:N]<=vmax[i], start=rand())

    # set contraints about initial values of the training example
    @constraint(m, [s in 1:NS_all], z[1,s,1] == z1_train_all[s]);
    @constraint(m, [s in 1:NS_all], z[2,s,1] == z2_train_all[s]);
    @constraint(m, [s in 1:NS_all], z[3,s,1] == z3_train_all[s]);
    @constraint(m, [s in 1:NS_all], z[4,s,1] == z4_train_all[s]);
    @constraint(m, [i in 1:4, s in 1:NS_all, t in 3:(N+1)], zmin[i] + epsilon <= z[i,s,t] <= zmax[i]-epsilon)

# contraints about the system dynamics
    #z1 = zold[1] + h*(- a[1]/A[1]*sqrt(2*g*(zold[1]+xs[1])) + a[3]/A[1]*sqrt(2*g*(zold[3]+xs[3])) + gama[1]/A[1]*(v[1]+us[1]))
    @NLconstraint(m, [s in 1:NS_all, t in 1:N], (z[1,s,t+1] - z[1,s,t])/h  == ( - a[1]/A[1]*sqrt(2*g*(z[1,s,t]+xs[1])) + a[3]/A[1]*sqrt(2*g*(z[3,s,t]+xs[3]))
    + gama[1]/A[1]*(v[1,s,t]+us[1])))

    #z2 = zold[2] + h*(- a[2]/A[2]*sqrt(2*g*(zold[2]+xs[2])) + a[4]/A[2]*sqrt(2*g*(zold[4]+xs[4])) + gama[2]/A[2]*(v[2]+us[2]))
    @NLconstraint(m, [s in 1:NS_all, t in 1:N], (z[2,s,t+1] - z[2,s,t])/h  == ( - a[2]/A[2]*sqrt(2*g*(z[2,s,t]+xs[2])) + a[4]/A[2]*sqrt(2*g*(z[4,s,t]+xs[4]))
    + gama[2]/A[2]*(v[2,s,t]+us[2])))

    #z3 = zold[3] + h*(- a[3]/A[3]*sqrt(2*g*(zold[3]+xs[3]))                                   + (1-gama[2])/A[3]*(v[2]+us[2]))
    @NLconstraint(m, [s in 1:NS_all, t in 1:N], (z[3,s,t+1] - z[3,s,t])/h  == (-  a[3]/A[3]*sqrt(2*g*(z[3,s,t]+xs[3]))
    + (1-gama[2])/A[3]*(v[2,s,t]+us[2])))

    #z4 = zold[4] + h*(- a[4]/A[4]*sqrt(2*g*(zold[4]+xs[4]))                                   + (1-gama[1])/A[4]*(v[1]+us[1]))
    @NLconstraint(m, [s in 1:NS_all, t in 1:N], (z[4,s,t+1] - z[4,s,t])/h  == (-  a[4]/A[4]*sqrt(2*g*(z[4,s,t]+xs[4]))
    + (1-gama[1])/A[4]*(v[1,s,t]+us[1])))

    @NLconstraint(m, [s in 1:NS_all], 6.55*z[1,s,N+1]^2 + 6.55*z[2,s,N+1]^2 + 7.92*z[3,s,N+1]^2 + 31.7*z[4,s,N+1]^2 <= α)

#nominal explict control law
@variable(m, -100<=W1[i in 1:4, j in 1:l1]<=100, start = W_1[i,j]);
@variable(m, -100 <= b1[i in 1:l1]<=100, start = b_1[i]);
@variable(m, -100 <= W2[i in 1:l1, j in 1:2]<=100, start = W_2[i,j]);
@variable(m, -100<=b2[i in 1:2]<=100, start = b_2[i]);


@variable(m, -1<=hidden[s in 1:NS_all, 1:N, 1:l1]<=1, start=rand());
@variable(m, sl1[1:NS_all, 1:N, 1:l1], start=rand());
@NLconstraint(m, [s in 1:NS_all, t in 1:N, j in 1:l1], 0 == (sl1[s,t,j] - W1[1,j]*z[1,s,t] - W1[2,j]*z[2,s,t] - W1[3,j]*z[3,s,t] - W1[4,j]*z[4,s,t] - b1[j]));
@NLconstraint(m, [s in 1:NS_all, t in 1:N, j in 1:l1], 0 == (hidden[s,t,j] - tanh(sl1[s,t,j])))
#@NLconstraint(m, [s in 1:NS, t in 1:N, j in 1:l1], hidden[s,t,j] ==  1 / (1+ exp(-sl1[s,t,j])));

@variable(m, sl2[1:NS_all, 1:N, 1:2], start=rand());
@variable(m, -1<=hidden2[s in 1:NS_all, 1:N, 1:2]<=1, start=rand());
@NLconstraint(m, [s in 1:NS_all, t in 1:N, i in 1:2], 0 == (sl2[s,t,i] -  sum(  W2[j,i]*hidden[s,t,j]  for j in 1:l1)  - b2[i]))
@NLconstraint(m, [s in 1:NS_all, t in 1:N, i in 1:2], 0 == (hidden2[s,t,i] -  tanh( sl2[s,t,i])))
@NLconstraint(m, [s in 1:NS_all, t in 1:N, i in 1:2], 0 == (v[i,s,t] -  (vmax[i] + vmin[i])/2 - (vmax[i] - vmin[i])/2*hidden2[s,t,i]))


@objective(m, Min, sum((z[1,s, t])^2 for s in 1:NS for t in 2:(N)) + sum((z[2,s, t])^2 for s in 1:NS for t in 2:(N))
+ 1e-2*sum((v[1,s,t])^2 for s in 1:NS for t in 1:N) + 1e-2*sum((v[2,s,t])^2 for s in 1:NS for t in 1:N)
+ sum( 6.55*z[1,s,N+1]^2 + 6.55*z[2,s,N+1]^2 + 7.92*z[3,s,N+1]^2 + 31.7*z[4,s,N+1]^2 for s in 1:NS ))
#+ 1e-4*sum(W1[i,j]^2 for i in 1:4 for j in 1:l1) + 1e-4*sum(W2[i,j]^2 for i in 1:l1 for j in 1:2) + 1e-4*sum(b1[i]^2 for i in 1:l1)  + 1e-4*sum(b2[i]^2 for i in 1:2))



SS_train = time()
JuMP.optimize!(m)
println(" SS train   ", (time() - SS_train))

using JLD
T_W1 = value.(W1)
T_W2 = value.(W2)
T_b1 = value.(b1)
T_b2 = value.(b2)
save("tank.jld", "W1", T_W1, "W2", T_W2, "b1", T_b1, "b2", T_b2)

opt = Chain(
  Dense(4, l1, tanh),
  Dense(l1, 2, tanh))

opt.layers[1].W[:] = value.(W1)'
opt.layers[1].b[:] = value.(b1)
opt.layers[2].W[:] = value.(W2)'
opt.layers[2].b[:] = value.(b2)



# v_trial is generated by v_train (which is optimized by explict computation of ipopt)
function denormalizeV(v)
    return (vmax .+ vmin)/2 .+ (vmax .- vmin) ./ 2 .* v
    #return vmin .+ v.*(vmax-vmin)
end
SS_test_start = time()
cost_opt_train = zeros(0)
vio_opt_train = zeros(0)
vio_percent_opt_train = zeros(0)

for i = 1:ntrain
            v_trial = zeros(Float64,2,N_sim)
            z_trial = zeros(Float64,4,N_sim+1)
            z_trial[:,1] = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N_sim
                v_trial[:,t] = denormalizeV(opt(z_trial[:,t]))
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[:,t])
            end

            append!(cost_opt_train,  sum(z_trial[1,2:N_sim+1].^2 + z_trial[2,2:N_sim+1].^2) + sum(0.01*v_trial[1,:].^2 + 0.01*v_trial[2,:].^2))
            append!(vio_opt_train,
                    maximum(max.(z_trial[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z_trial[1,:],0)) +
                    maximum(max.(z_trial[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z_trial[2,:],0)) +
                    maximum(max.(z_trial[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z_trial[3,:],0)) +
                    maximum(max.(z_trial[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z_trial[4,:],0))
            )
            if vio_opt_train[i] >= 1e-3
                vio_percent_opt_train[i] = 1
            end
end

cost_opt_test = zeros(0)
vio_opt_test = zeros(0)
vio_percent_opt_test = zeros(0)
for i = 1:ntest
            v_trial = zeros(Float64,2,N_sim)
            z_trial = zeros(Float64,4,N_sim+1)
            z_trial[:,1] = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
            for t = 1:N_sim
                v_trial[:,t] = denormalizeV(opt(z_trial[:,t]))
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[:,t])
            end
            append!(cost_opt_test,  sum(z_trial[1,2:N_sim+1].^2 + z_trial[2,2:N_sim+1].^2) + sum(0.01*v_trial[1,:].^2 + 0.01*v_trial[2,:].^2))
            append!(vio_opt_test,
                    maximum(max.(z_trial[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z_trial[1,:],0)) +
                    maximum(max.(z_trial[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z_trial[2,:],0)) +
                    maximum(max.(z_trial[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z_trial[3,:],0)) +
                    maximum(max.(z_trial[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z_trial[4,:],0))
            )
            if vio_opt_test[i] >= 1e-3
                append!(vio_percent_opt_test,1)
            else
                append!(vio_percent_opt_test,0)
            end
end
println(" SS test   ", (time() - SS_test_start)/N/(ntrain+ntest))

println("Ipopt mean cost train:    ",mean(cost_opt_train))
println("Ipopt mean cost test :    ",mean(cost_opt_test))
println("Ipopt vio train:    ",maximum(vio_opt_train))
println("Ipopt vio test:     ",maximum(vio_opt_test))
println("Ipopt vio percent in train", sum(vio_percent_opt_train)/ntrain)
println("Ipopt vio percent in test", sum(vio_percent_opt_test)/ntest)




cost_opt_train = zeros(0)
vio_opt_train = zeros(0)
vio_percent_opt_train = zeros(ntrain)
for i = 1:ntrain
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(opt(z[:,t]))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end
            append!(cost_opt_train,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_opt_train,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_opt_train[i] >= 1e-3
                vio_percent_opt_train[i] = 1
            end
end
cost_opt_test = zeros(0)
vio_opt_test = zeros(0)
vio_percent_opt_test = zeros(0)
for i = 1:ntest
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(opt(z[:,t]))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end

            append!(cost_opt_test,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_opt_test,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_opt_test[i] >= 1e-3
                append!(vio_percent_opt_test,1)
            else
                append!(vio_percent_opt_test,0)
            end
end


println(" Ipopt mean cost train:    ",mean(cost_opt_train))
println(" Ipopt mean cost test:    ",mean(cost_opt_test))
println(" Ipopt vio train:    ", maximum(vio_opt_train))
println(" Ipopt vio test:     ", maximum(vio_opt_test))
println(" Ipopt vio percent in train:  ", (sum(vio_percent_opt_train))/ntrain)
println(" Ipopt vio percent in train:  ", (sum(vio_percent_opt_test))/ntest)


# mean(cost_train)
# mean(cost_test)
# maximum(vio_train)
# maximum(vio_test)
#
# mean(cost_ml_train)
# mean(cost_ml_test)
# maximum(vio_ml_train)
# maximum(vio_ml_test)
#
# mean(cost_rnn_train)
# mean(cost_rnn_test)
# maximum(vio_rnn_train)
# maximum(vio_rnn_test)
#
# mean(cost_opt_train)
# mean(cost_opt_test)
# maximum(vio_opt_train)
# maximum(vio_opt_test)
