using Flux
using Random, Distributions
import Distributions: Uniform
using JLD


Random.seed!(0)
N=20
N_sim = 500
h=3
T=collect(1:N)

xs = [14, 14, 14.2, 21.3]
us = [43.4, 35.4]
vmin = [-43.4,-35.4]
vmax = [16.6, 24.6]
zmin = [-6.5, -6.5, -10.7, -16.8]
zmax = [14, 14, 13.8, 6.7]

g=981
A= [50.27,50.27, 28.27,28.27]
a= [0.233, 0.242, 0.127, 0.127]
gama = [0.4, 0.4]

ntrain = 600
z1_train = rand(Uniform(zmin[1],zmax[1]),ntrain)
z2_train = rand(Uniform(zmin[2],zmax[2]),ntrain)
z3_train = rand(Uniform(zmin[3],zmax[3]),ntrain)
z4_train = rand(Uniform(zmin[4],zmax[4]),ntrain)

d = load("test_data_1k.jld")
z1_test = d["z1"]
z2_test = d["z2"]
z3_test = d["z3"]
z4_test = d["z4"]
ntest = length(z1_test)

# function denormalizeV(v)
#         return (vmin .+ v.*(vmax .- vmin))
# end

function denormalizeV(v)
        return (vmax .+ vmin)/2 .+ (vmax .- vmin) ./ 2 .* v
end

function step_model(z, v)
    zold = copy(z)
    z1 = zold[1] + h*(- a[1]/A[1]*sqrt(2*g*(zold[1]+xs[1])) + a[3]/A[1]*sqrt(2*g*(zold[3]+xs[3])) + gama[1]/A[1]*(v[1]+us[1]))
    z2 = zold[2] + h*(- a[2]/A[2]*sqrt(2*g*(zold[2]+xs[2])) + a[4]/A[2]*sqrt(2*g*(zold[4]+xs[4])) + gama[2]/A[2]*(v[2]+us[2]))
    z3 = zold[3] + h*(- a[3]/A[3]*sqrt(2*g*(zold[3]+xs[3]))   			              + (1-gama[2])/A[3]*(v[2]+us[2]))
    z4 = zold[4] + h*(- a[4]/A[4]*sqrt(2*g*(zold[4]+xs[4]))                                   + (1-gama[1])/A[4]*(v[1]+us[1]))
    return [z1,z2,z3,z4]
end

println(step_model([0,0,0,0], [0,0]))

l1 = 10
 control_law_rnn = Chain(
    Dense(4, l1, tanh) |> f64,
    Dense(l1, 2, tanh)|> f64
  )


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

control_law_rnn.layers[1].W[:] = W_1'
control_law_rnn.layers[1].b[:] = b_1
control_law_rnn.layers[2].W[:] = W_2'
control_law_rnn.layers[2].b[:] = b_2



 println(control_law_rnn([0,0,0,0]))

function loss(z0,y)
    z_stage = copy(z0)
    z = z_stage'  # z_stage 需要为列向量
    v = [0,0] # equal to the None in Python
    for t = 1:N
        v_stage =  denormalizeV(control_law_rnn(z_stage))		#denormalizeV(control_law_rnn((normalizeZ(x))))
        z_stage = step_model(z_stage, v_stage) #z_stage = Tracker.collect(step_model(z_stage, v_stage))
	    z = vcat(z,z_stage') # vertical concatenate
        if t == 1
	    v = v_stage'
        else
	    v = vcat(v,v_stage')
        end
    end
    return (sum((z[1:N-1,1]).^2) + sum((z[1:N-1,2]).^2) + 0.01*sum((v[:,1]).^2) + 0.01*sum((v[:,2]).^2) + 6.55*z[N,1]^2 + 6.55*z[N,2]^2
    + 7.92*z[N,3]^2 + 31.7*z[N,4]^2 +
    50*(sum(max.(z[:,1] .- zmax[1], 0)) + sum(max.(zmin[1].-z[:,1],0)) +
    sum(max.(z[:,2] .- zmax[2], 0)) + sum(max.(zmin[2].-z[:,2],0)) +
    sum(max.(z[:,3] .- zmax[3], 0)) +sum(max.(zmin[3].-z[:,3],0)) +
    sum(max.(z[:,4] .- zmax[4], 0)) + sum(max.(zmin[4].-z[:,4],0))))
end


data=[([z1_train[i],z2_train[i],z3_train[i],z4_train[i]], 0) for i in 1:ntrain]
opt = ADAM()
rnn_start = time()
Flux.@epochs 100 Flux.train!(loss, Flux.params(control_law_rnn), data, opt)
println("RNN train time is ", time() - rnn_start)


using BSON: @save
@save "rnn_control.bson" control_law_rnn




cost_rnn_train = zeros(0)
vio_rnn_train = zeros(0)
vio_percent_rnn_train = zeros(ntrain)
for i = 1:ntrain
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_train[i], z2_train[i], z3_train[i], z4_train[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(control_law_rnn(z[:,t]))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end
            append!(cost_rnn_train,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_rnn_train,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_rnn_train[i] >= 1e-3
                vio_percent_rnn_train[i] = 1
            end
end


cost_rnn_test = zeros(0)
vio_rnn_test = zeros(0)
vio_percent_rnn_test = zeros(0)
SS_test_start = time()
for i = 1:ntest
            v = zeros(Float64,2,N_sim)
            z = zeros(Float64,4,N_sim+1)
            z[:,1] = [z1_test[i], z2_test[i], z3_test[i], z4_test[i]]
            for t = 1:N_sim
                v[:,t] = denormalizeV(control_law_rnn(z[:,t]))
                z[:,t+1] = step_model(z[:,t], v[:,t])
            end
            append!(cost_rnn_test,  sum(z[1,2:N_sim+1].^2 .+ z[2,2:N_sim+1].^2) + sum(0.01*v[1,:].^2 + 0.01*v[2,:].^2))
            append!(vio_rnn_test,
                    maximum(max.(z[1,:] .- zmax[1], 0) .+ max.(zmin[1].-z[1,:],0)) +
                    maximum(max.(z[2,:] .- zmax[2], 0) .+ max.(zmin[2].-z[2,:],0)) +
                    maximum(max.(z[3,:] .- zmax[3], 0) .+ max.(zmin[3].-z[3,:],0)) +
                    maximum(max.(z[4,:] .- zmax[4], 0) .+ max.(zmin[4].-z[4,:],0))
            )
            if vio_rnn_test[i] >= 1e-3
                append!(vio_percent_rnn_test,1)
            else
                append!(vio_percent_rnn_test,0)
            end
end
#
#
println(" SS test   ", (time() - SS_test_start)/N/(ntrain+ntest))

println("rnn mean cost train:    ",mean(cost_rnn_train))
println("rnn mean cost test :    ",mean(cost_rnn_test))
println("rnn vio train:    ",maximum(vio_rnn_train))
println("rnn vio test:     ",maximum(vio_rnn_test))
println("rnn vio percent in train", sum(vio_percent_rnn_train)/ntrain)
println("rnn vio percent in test", sum(vio_percent_rnn_test)/ntest)
