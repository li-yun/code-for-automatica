using JuMP
using Ipopt
using Flux
using Random, Distributions
import Distributions: Uniform
using BSON: @save

Random.seed!(1)
N = 10


xmin = [-10 -10]  ## s1 v1 s2 v2
xmax = [10 10]

umin = [-10]
umax = [10]
x1ref = 1
x2ref = 1.26
xmin_train = xmin
xmax_train = xmax


ntrain = 10000
n_x1 = 100
n_x2 = 100

# x1_train = rand(Uniform(xmin_train[1],xmax_train[1]),ntrain)
# x2_train = rand(Uniform(xmin_train[2],xmax_train[2]),ntrain)
x1_train = zeros(ntrain)
x2_train = zeros(ntrain)

x1 = LinRange(xmin[1],xmax[1],n_x1)
x2 = LinRange(xmin[2],xmax[2],n_x2)


for i in 1:n_x1
	for j in 1:n_x2
		x1_train[(i-1)*n_x2 + j] = x1[i]
		x2_train[(i-1)*n_x2 + j] = x2[j]
	end
end



ntest = 1000
x1_test = rand(Uniform(xmin_train[1],xmax_train[1]),ntest)
x2_test = rand(Uniform(xmin_train[2],xmax_train[2]),ntest)

N_sim = 20
l1 =20
function OptimalControl(x0)
    m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))

    x_initial_guess = zeros(2, N+1)
    u_initial_guess = zeros(N)
    x_initial_guess[1,:] = rand(Uniform(xmin[1],xmax[1]), N+1)
    x_initial_guess[2,:] = rand(Uniform(xmin[2],xmax[2]), N+1)
    u_initial_guess = rand(Uniform(umin[1],umax[1]), N)

    @variable(m, x[i in 1:2, t in 1:(N+1)], start=  x_initial_guess[i,t])
    @variable(m, umin[1] <= u[t in 1:N] <= umax[1], start =  u_initial_guess[t])

    @constraint(m, [i in 1:2], x[i,1] == x0[i])
    @NLconstraint(m, [t in 1:N], x[1,t+1] == 0.95*x[1,t] - 0.25*x[1,t]*x[2,t] + x[2,t])
    @NLconstraint(m, [t in 1:N], x[2,t+1] == 0.7*x[2,t] + u[t])
    @objective(m, Min, sum((x[1,t] - x1ref)^2 + (x[2,t] - x2ref)^2 for t in 2:N+1)  )


    JuMP.optimize!(m)
#    println(termination_status(m), "   ",objective_value(m), "   ",JuMP.value(u[1]))
    return JuMP.value(u[1]), termination_status(m)
end



function denormalizeU(u,ind=0)
    if ind==0
        return (umax .+ umin)/2 .+ (umax .- umin) ./ 2 .* u
    else
        return (umax[ind] .+ umin[ind])/2 .+ (umax[ind] .- umin[ind]) ./ 2 .* u[ind]
    end
end

function normalizeU(u,ind=0)
    if ind==0
        return (u.-   (umin .+ umax) ./ 2      )./((umax-umin)./2)
    else
        return (u[ind].-   (umin[ind] .+ umax[ind]) ./ 2      )./((umax[ind]-umin[ind])./2)
    end
end

function step_model(x,u)
    xold = copy(x)
    x1 = 0.95*xold[1] - 0.25*xold[1]*xold[2] + xold[2]
	x2 = 0.7*xold[2] + u
    return [x1, x2]
end


### train for model predictive control
#
# cost_train = zeros(0)
# vio_train = zeros(0)
# vio_percent_train = zeros(0)
# for i = 1:ntrain # n_train = 81
#             v_trial = zeros(Float64,1,N_sim)
#             z_trial = zeros(Float64,2,N_sim+1)
#             z_trial[:,1] = [x1_train[i], x2_train[i]]
#             for t = 1:N_sim
#                 v_trial[1,t], _ = OptimalControl(z_trial[:,t])
#                 z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[1,t])
#             end
#             append!(cost_train, sum((z_trial[1,t] - x1ref)^2 + (z_trial[2,t] - x2ref)^2 for t in 2:N_sim+1))
#             append!(vio_train,
#                     maximum(max.(z_trial[1,:] .- xmax[1], 0) .+ max.(xmin[1].-z_trial[1,:],0)) +
#                     maximum(max.(z_trial[2,:] .- xmax[2], 0) .+ max.(xmin[2].-z_trial[2,:],0))
#             )
# end
#
# println("MPC mean cost train:    ",mean(cost_train))
# println("MPC vio train:    ",maximum(vio_train))


### test for model predictive control
cost_test = zeros(0)
vio_test = zeros(0)
vio_percent_test = zeros(0)
for i = 1:ntest # n_train = 81
            v_trial = zeros(Float64,1,N_sim)
            z_trial = zeros(Float64,2,N_sim+1)
            z_trial[:,1] = [x1_test[i], x2_test[i]]
            for t = 1:N_sim
                v_trial[1,t], _ = OptimalControl(z_trial[:,t])
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[1,t])
            end
            append!(cost_test, sum((z_trial[1,t] - x1ref)^2 + (z_trial[2,t] - x2ref)^2 for t in 2:N_sim+1))
            append!(vio_test,
                    maximum(max.(z_trial[1,:] .- xmax[1], 0) .+ max.(xmin[1].-z_trial[1,:],0)) +
                    maximum(max.(z_trial[2,:] .- xmax[2], 0) .+ max.(xmin[2].-z_trial[2,:],0))
            )
end

println("MPC mean cost test:    ",mean(cost_test))
println("MPC vio test:    ",maximum(vio_test))





z_train = zeros(Float64,2,ntrain)
v_train = zeros(Float64,1,ntrain)
for i = 1:ntrain
            z = [x1_train[i], x2_train[i]]
            v, _ = OptimalControl(z)
            z_train[:,i] = z
            v_train[1,i] = v[1]
end


ml = Chain(
  Dense(2, l1, tanh),
  Dense(l1, 1, tanh)
)

loss(x, y) = Flux.mse(ml(x),  y)   #sum((ml(x) .- y).^2)
mae(x, y) = mean(abs.(ml(x).- y))

data = [(z_train[:,i],normalizeU(v_train[1,i])) for i in 1:ntrain]
opt = ADAM()
Flux.@epochs 10000 Flux.train!(loss, Flux.params(ml), data, opt)


### test for optimize then train approach
cost_test_ml = zeros(0)
vio_test_ml = zeros(0)
vio_percent_test_ml = zeros(0)
for i = 1:ntest # n_train = 81
            v_trial = zeros(Float64,1,N_sim)
            z_trial_ml = zeros(Float64,2,N_sim+1)
            z_trial_ml[:,1] = [x1_test[i], x2_test[i]]
            for t = 1:N_sim
                                v_trial[:,t] = denormalizeU(ml(z_trial_ml[:,t]))
                z_trial_ml[:,t+1] = step_model(z_trial_ml[:,t], v_trial[1,t])
            end
            append!(cost_test_ml, sum((z_trial_ml[1,t] - x1ref)^2 + (z_trial_ml[2,t] - x2ref)^2 for t in 2:N_sim+1))
            append!(vio_test_ml,
                    maximum(max.(z_trial_ml[1,:] .- xmax[1], 0) .+ max.(xmin[1].-z_trial_ml[1,:],0)) +
                    maximum(max.(z_trial_ml[2,:] .- xmax[2], 0) .+ max.(xmin[2].-z_trial_ml[2,:],0))
            )
end

println("ML mean cost test:    ",mean(cost_test_ml))
println("ML vio test:    ",maximum(vio_test_ml))
@save "ml_model_new.bson" ml







### optimize and train approach
dnn_rnn = Chain(
  Dense(2, l1, tanh),
  Dense(l1, 1, tanh)
)

function loss(x0, y)
    x_stage = copy(x0)
    x = x_stage'
    for t = 1:N
        v_stage = denormalizeU(dnn_rnn(x_stage))
        x_stage = step_model(x_stage, v_stage[1])
        x = vcat(x,x_stage')
    end
    #cost = (sum((x[t,1] - x1ref)^2 + (x[t,2] - x2ref)^2 for t in 2:N+1))
    #println("cost    ",cost)
    return (sum((x[t,1] - x1ref)^2 + (x[t,2] - x2ref)^2 for t in 2:N+1))
end


data=[([x1_train[i],x2_train[i]], 0) for i in 1:ntrain]
opt = ADAM()
Flux.@epochs 5000 Flux.train!(loss, Flux.params(dnn_rnn), data, opt)

cost_train_rnn = zeros(0)
for i = 1:ntrain # n_train = 81
            v_trial = zeros(Float64,1,N_sim)
            z_trial = zeros(Float64,2,N_sim+1)
            z_trial[:,1] = [x1_train[i], x2_train[i]]
            for t = 1:N_sim
                v_trial[:,t] = denormalizeU(dnn_rnn(z_trial[:,t]))
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[1,t])
            end
            append!(cost_train_rnn, sum((z_trial[1,t] - x1ref)^2 + (z_trial[2,t] - x2ref)^2 for t in 2:N_sim+1))
end
println("RNN mean cost train:    ",mean(cost_train_rnn))

cost_test_rnn = zeros(0)
for i = 1:ntest # n_train = 81
            v_trial = zeros(Float64,1,N_sim)
            z_trial = zeros(Float64,2,N_sim+1)
            z_trial[:,1] = [x1_test[i], x2_test[i]]
            for t = 1:N_sim
				v_trial[:,t] = denormalizeU(dnn_rnn(z_trial[:,t]))
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[1,t])
            end
            append!(cost_test_rnn, sum((z_trial[1,t] - x1ref)^2 + (z_trial[2,t] - x2ref)^2 for t in 2:N_sim+1))

end
println("RNN mean cost test:    ",mean(cost_test_rnn))




@save "rnn_model_new.bson" dnn_rnn

using JLD
JLD.save("train_data.jld", "x_train", z_train, "v_train", v_train)
