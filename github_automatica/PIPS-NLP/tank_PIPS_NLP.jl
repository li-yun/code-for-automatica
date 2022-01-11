using Ipopt
using JuMP
using Plasmo

using Random, Distributions
using Distributions: Uniform

using PipsSolver
using MPI
MPI.Init()
comm = MPI.COMM_WORLD
ncores = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)



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


N=20                                            #Total steps in Opt
h=3                                             #time step  0.002h

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
w1max = 10
w1min = -10
b1max = 10
b1min = -10
w2max = 50
w2min = -50
b2max = 10
b2min = -10
l1 = 10


opt(x) = tanh.(W_2'*tanh.(W_1'*x + b_1)+b_2)
opt_1(x) = W_2'*tanh.(W_1'*x + b_1)+b_2
_layer11(x) = tanh.(W_1'*x + b_1)
_layer12(x) = W_1'*x + b_1

Random.seed!(0)
ntrain = 200
NS = ntrain # total number of node
nn = 1 # number of scenarios per node
z1_train = rand(Uniform(zmin[1],zmax[1]),(nn,ntrain))
z2_train = rand(Uniform(zmin[2],zmax[2]),(nn,ntrain))
z3_train = rand(Uniform(zmin[3],zmax[3]),(nn,ntrain))
z4_train = rand(Uniform(zmin[4],zmax[4]),(nn,ntrain))


# adding critical scenarios
n_split = 5
ntrain_add = n_split^4
z1_train_add = zeros((nn,ntrain_add))
z2_train_add = zeros((nn,ntrain_add))
z3_train_add = zeros((nn,ntrain_add))
z4_train_add = zeros((nn,ntrain_add))

s = 1
for i = 0:n_split-1
    for j = 0:n_split-1
        for k = 0:n_split-1
            for l = 0:n_split-1
                z1_train_add[nn,s] = zmin[1] + (zmax[1] - zmin[1])*i/(n_split-1)
               	z2_train_add[nn,s] = zmin[2] + (zmax[2] - zmin[2])*j/(n_split-1)
                z3_train_add[nn,s] = zmin[3] + (zmax[3] - zmin[3])*k/(n_split-1)
                z4_train_add[nn,s] = zmin[4] + (zmax[4] - zmin[4])*l/(n_split-1)
                global s += 1
            end
        end
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


x_init = zeros(4,N+1,nn,NS)
v_init = zeros(2,N,nn,NS)
for ns = 1:ntrain
    for n_n in 1:nn
            v_trial = zeros(Float64,2,N)
            z_trial = zeros(Float64,4,N+1)
            z_trial[:,1] = [z1_train[n_n,ns], z2_train[n_n,ns], z3_train[n_n,ns], z4_train[n_n,ns]]
            for t = 1:N
                v_trial[:,t] = denormalizeV(opt(z_trial[:,t]))
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[:,t])
            end
            x_init[:,:,n_n,ns] = z_trial
            v_init[:,:,n_n,ns] = v_trial
    end
end

x_init_add = zeros(4,N+1,nn, ntrain_add)
v_init_add = zeros(2,N,nn, ntrain_add)
for ns = 1:ntrain_add
    for n_n in 1:nn
            v_trial = zeros(Float64,2,N)
            z_trial = zeros(Float64,4,N+1)
            z_trial[:,1] = [z1_train_add[n_n,ns], z2_train_add[n_n,ns], z3_train_add[n_n,ns], z4_train_add[n_n,ns]]
            for t = 1:N
                v_trial[:,t] = denormalizeV(opt(z_trial[:,t]))
                z_trial[:,t+1] = step_model(z_trial[:,t], v_trial[:,t])
            end
            x_init_add[:,:,n_n,ns] = z_trial
            v_init_add[:,:,n_n,ns] = v_trial
    end
end




function create_simple_node(ns)
    node = OptiNode()
    @variable(node, zmin[i] <= z[i in 1:4, s in 1:nn, t in 1:(N+1)] <= zmax[i], start=x_init[i,t,s,ns]) ## set inital = 0 can reduce restoration
    @variable(node,  vmin[i] <= v[i in 1:2, s in 1:nn, t in 1:N] <= vmax[i], start=v_init[i,t,s,ns])

    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[1,s,t+1] + z[1,s,t])/h + ( - a[1]/A[1]*sqrt(2*g*(z[1,s,t]+xs[1])) + a[3]/A[1]*sqrt(2*g*(z[3,s,t]+xs[3]))
    + gama[1]/A[1]*(v[1,s,t]+us[1])))

    #z2 = zold[2] + h*(- a[2]/A[2]*sqrt(2*g*(zold[2]+xs[2])) + a[4]/A[2]*sqrt(2*g*(zold[4]+xs[4])) + gama[2]/A[2]*(v[2]+us[2]))
    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[2,s,t+1] + z[2,s,t])/h + ( - a[2]/A[2]*sqrt(2*g*(z[2,s,t]+xs[2])) + a[4]/A[2]*sqrt(2*g*(z[4,s,t]+xs[4]))
    + gama[2]/A[2]*(v[2,s,t]+us[2])))

    #z3 = zold[3] + h*(- a[3]/A[3]*sqrt(2*g*(zold[3]+xs[3]))                                   + (1-gama[2])/A[3]*(v[2]+us[2]))
    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[3,s,t+1] + z[3,s,t])/h + (-  a[3]/A[3]*sqrt(2*g*(z[3,s,t]+xs[3]))
    + (1-gama[2])/A[3]*(v[2,s,t]+us[2])))

    #z4 = zold[4] + h*(- a[4]/A[4]*sqrt(2*g*(zold[4]+xs[4]))                                   + (1-gama[1])/A[4]*(v[1]+us[1]))
    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[4,s,t+1] + z[4,s,t])/h + (-  a[4]/A[4]*sqrt(2*g*(z[4,s,t]+xs[4]))
    + (1-gama[1])/A[4]*(v[1,s,t]+us[1])))


    @variable(node, W1[i in 1:4, j in 1:l1], start = W_1[i,j])
    @variable(node, b1[i in 1:l1], start = b_1[i])
    @variable(node, W2[i in 1:l1, j in 1:2], start = W_2[i,j])
    @variable(node, b2[i in 1:2], start = b_2[i])

    #=
    @variable(node, w1min <= W1[i in 1:4, j in 1:l1] <= w1max, start = W_1[i,j])
    @variable(node, b1min <= b1[i in 1:l1] <= b1max, start = b_1[i])
    @variable(node, w2min <= W2[i in 1:l1, j in 1:2] <= w2max, start = W_2[i,j])
    @variable(node, b2min <= b2[i in 1:2] <= b2max, start = b_2[i])
    =#

    @variable(node, -1<=hidden[s in 1:nn, t in 1:N, i in 1:l1]<=1, start = _layer11(x_init[:,t,s,ns])[i]); #start=rand());
    @variable(node, -1e3<=sl1[s in 1:nn, t in 1:N, i in 1:l1]<=1e3, start= _layer12(x_init[:,t,s,ns])[i]);
    @NLconstraint(node, [s in 1:nn, t in 1:N, j in 1:l1], 0 == (sl1[s,t,j] - W1[1,j]*z[1,s,t] - W1[2,j]*z[2,s,t] - W1[3,j]*z[3,s,t] - W1[4,j]*z[4,s,t] - b1[j]));
    @NLconstraint(node, [s in 1:nn, t in 1:N, j in 1:l1], hidden[s,t,j] == tanh(sl1[s,t,j]))

    @variable(node, -1e3<=sl2[s in 1:nn, t in 1:N, i in 1:2]<=1e3, start= opt_1(x_init[:,t,s,ns])[i]);
    @variable(node, -1<=hidden2[s in 1:nn, t in 1:N, i in 1:2]<=1, start= opt(x_init[:,t,s,ns])[i]);
    @NLconstraint(node, [s in 1:nn, t in 1:N, i in 1:2], 0 == (sl2[s,t,i] -  sum(  W2[j,i]*hidden[s,t,j]  for j in 1:l1)  - b2[i]))
    @NLconstraint(node, [s in 1:nn, t in 1:N, i in 1:2], hidden2[s,t,i] == tanh( sl2[s,t,i]))
    @NLconstraint(node, [s in 1:nn, t in 1:N, i in 1:2], 0 == (v[i,s,t] -  (vmax[i] + vmin[i])/2 - (vmax[i] - vmin[i])/2*hidden2[s,t,i]))
    @variable(node, ter_cost[s in 1:nn]<= α )
    @NLconstraint(node, [s in 1:nn], ter_cost[s] == 6.55*z[1,s,N+1]^2 + 6.55*z[2,s,N+1]^2 + 7.92*z[3,s,N+1]^2 + 31.7*z[4,s,N+1]^2 )

    @constraint(node, [s in 1:nn], z[1,s,1] == z1_train[s,ns]);
    @constraint(node, [s in 1:nn], z[2,s,1] == z2_train[s,ns]);
    @constraint(node, [s in 1:nn], z[3,s,1] == z3_train[s,ns]);
    @constraint(node, [s in 1:nn], z[4,s,1] == z4_train[s,ns]);

    @objective(node, Min, sum((z[1,s,t])^2 for t in 2:N for s in 1:nn) + sum((z[2,s,t])^2 for t in 2:N for s in 1:nn)
    + 1e-2*sum((v[1,s,t])^2 for t in 1:N for s in 1:nn) + 1e-2*sum((v[2,s,t])^2 for t in 1:N for s in 1:nn) +
    sum(ter_cost[s] for s in 1:nn)
    + 1e-3*(sum(W1[i,j]^2 for i = 1:4 for j = 1:l1) +  sum(b1[i]^2 for i = 1:l1)
            + sum(W2[i,j]^2 for i = 1:l1 for j = 1:2)  +  sum(b2[i]^2 for i = 1:2))  )
    # set_optimizer(node,with_optimizer(Ipopt.Optimizer,print_level=0))
    # optimize!(node)
    return node
end


function create_adding_node(ns)
    node = OptiNode()
    @variable(node, zmin[i] <= z[i in 1:4, s in 1:nn, t in 1:(N+1)] <= zmax[i], start=x_init_add[i,t,s,ns]) ## set inital = 0 can reduce restoration
    @variable(node,  vmin[i] <= v[i in 1:2, s in 1:nn, t in 1:N] <= vmax[i], start=v_init_add[i,t,s,ns])

    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[1,s,t+1] + z[1,s,t])/h + ( - a[1]/A[1]*sqrt(2*g*(z[1,s,t]+xs[1])) + a[3]/A[1]*sqrt(2*g*(z[3,s,t]+xs[3]))
    + gama[1]/A[1]*(v[1,s,t]+us[1])))

    #z2 = zold[2] + h*(- a[2]/A[2]*sqrt(2*g*(zold[2]+xs[2])) + a[4]/A[2]*sqrt(2*g*(zold[4]+xs[4])) + gama[2]/A[2]*(v[2]+us[2]))
    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[2,s,t+1] + z[2,s,t])/h + ( - a[2]/A[2]*sqrt(2*g*(z[2,s,t]+xs[2])) + a[4]/A[2]*sqrt(2*g*(z[4,s,t]+xs[4]))
    + gama[2]/A[2]*(v[2,s,t]+us[2])))

    #z3 = zold[3] + h*(- a[3]/A[3]*sqrt(2*g*(zold[3]+xs[3]))                                   + (1-gama[2])/A[3]*(v[2]+us[2]))
    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[3,s,t+1] + z[3,s,t])/h + (-  a[3]/A[3]*sqrt(2*g*(z[3,s,t]+xs[3]))
    + (1-gama[2])/A[3]*(v[2,s,t]+us[2])))

    #z4 = zold[4] + h*(- a[4]/A[4]*sqrt(2*g*(zold[4]+xs[4]))                                   + (1-gama[1])/A[4]*(v[1]+us[1]))
    @NLconstraint(node, [s in 1:nn, t in 1:N], 0 == (-z[4,s,t+1] + z[4,s,t])/h + (-  a[4]/A[4]*sqrt(2*g*(z[4,s,t]+xs[4]))
    + (1-gama[1])/A[4]*(v[1,s,t]+us[1])))


    @variable(node, W1[i in 1:4, j in 1:l1], start = W_1[i,j])
    @variable(node, b1[i in 1:l1], start = b_1[i])
    @variable(node, W2[i in 1:l1, j in 1:2], start = W_2[i,j])
    @variable(node, b2[i in 1:2], start = b_2[i])

    @variable(node, -1<=hidden[s in 1:nn, t in 1:N, i in 1:l1]<=1, start = _layer11(x_init_add[:,t,s,ns])[i]); #start=rand());
    @variable(node, -1e3<=sl1[s in 1:nn, t in 1:N, i in 1:l1]<=1e3, start= _layer12(x_init_add[:,t,s,ns])[i]);
    @NLconstraint(node, [s in 1:nn, t in 1:N, j in 1:l1], 0 == (sl1[s,t,j] - W1[1,j]*z[1,s,t] - W1[2,j]*z[2,s,t] - W1[3,j]*z[3,s,t] - W1[4,j]*z[4,s,t] - b1[j]));
    @NLconstraint(node, [s in 1:nn, t in 1:N, j in 1:l1], hidden[s,t,j] == tanh(sl1[s,t,j]))

    @variable(node, -1e3<=sl2[s in 1:nn, t in 1:N, i in 1:2]<=1e3, start= opt_1(x_init_add[:,t,s,ns])[i]);
    @variable(node, -1<=hidden2[s in 1:nn, t in 1:N, i in 1:2]<=1, start= opt(x_init_add[:,t,s,ns])[i]);
    @NLconstraint(node, [s in 1:nn, t in 1:N, i in 1:2], 0 == (sl2[s,t,i] -  sum(  W2[j,i]*hidden[s,t,j]  for j in 1:l1)  - b2[i]))
    @NLconstraint(node, [s in 1:nn, t in 1:N, i in 1:2], hidden2[s,t,i] == tanh( sl2[s,t,i]))
    @NLconstraint(node, [s in 1:nn, t in 1:N, i in 1:2], 0 == (v[i,s,t] -  (vmax[i] + vmin[i])/2 - (vmax[i] - vmin[i])/2*hidden2[s,t,i]))
    @variable(node, ter_cost[s in 1:nn] )
    @NLconstraint(node, [s in 1:nn], ter_cost[s] == 6.55*z[1,s,N+1]^2 + 6.55*z[2,s,N+1]^2 + 7.92*z[3,s,N+1]^2 + 31.7*z[4,s,N+1]^2 )

    @constraint(node, [s in 1:nn], z[1,s,1] == z1_train_add[s,ns]);
    @constraint(node, [s in 1:nn], z[2,s,1] == z2_train_add[s,ns]);
    @constraint(node, [s in 1:nn], z[3,s,1] == z3_train_add[s,ns]);
    @constraint(node, [s in 1:nn], z[4,s,1] == z4_train_add[s,ns]);

    @objective(node, Min, 1e-3*(sum(W1[i,j]^2 for i = 1:4 for j = 1:l1) +  sum(b1[i]^2 for i = 1:l1)
            + sum(W2[i,j]^2 for i = 1:l1 for j = 1:2)  +  sum(b2[i]^2 for i = 1:2)) )
    return node
end



graph = OptiGraph()
first_stage = @optinode(graph)
@variable(first_stage, W1[i in 1:4, j in 1:l1], start = W_1[i,j])
@variable(first_stage, b1[i in 1:l1], start = b_1[i])
@variable(first_stage, W2[i in 1:l1, j in 1:2], start = W_2[i,j])
@variable(first_stage, b2[i in 1:2], start = b_2[i])
#@objective(first_stage, Min, 1e-3*(sum(W1[i,j]^2 for i = 1:4 for j = 1:l1) +  sum(b1[i]^2 for i = 1:l1)
#            + sum(W2[i,j]^2 for i = 1:l1 for j = 1:2)  +  sum(b2[i]^2 for i = 1:2)))

subgraph = OptiGraph()
add_subgraph!(graph,subgraph)

# adding training scenarios in the graph
for s in 1:NS
    node = create_simple_node(s)
    add_node!(subgraph,node)
    @linkconstraint(graph, [i in 1:4, j in 1:l1], node[:W1][i,j] == first_stage[:W1][i,j] )
    @linkconstraint(graph, [i in 1:l1], node[:b1][i] == first_stage[:b1][i] )
    @linkconstraint(graph, [i in 1:l1, j in 1:2], node[:W2][i,j] == first_stage[:W2][i,j] )
    @linkconstraint(graph, [i in 1:2], node[:b2][i] == first_stage[:b2][i])
end

# adding critical scenarios in the graph
for s in 1:ntrain_add
    node = create_adding_node(s)
    add_node!(subgraph,node)
    @linkconstraint(graph, [i in 1:4, j in 1:l1], node[:W1][i,j] == first_stage[:W1][i,j] )
    @linkconstraint(graph, [i in 1:l1], node[:b1][i] == first_stage[:b1][i] )
    @linkconstraint(graph, [i in 1:l1, j in 1:2], node[:W2][i,j] == first_stage[:W2][i,j] )
    @linkconstraint(graph, [i in 1:2], node[:b2][i] == first_stage[:b2][i])
end

start_time = time()
pipsnlp_solve(graph)
if rank == 0
    println("Solving with PIPS-NLP")
end

using JLD
T_W1 = nodevalue.(first_stage[:W1])
T_W2 = nodevalue.(first_stage[:W2])
T_b1 = nodevalue.(first_stage[:b1])
T_b2 = nodevalue.(first_stage[:b2])
save("tank_nlp.jld", "W1", T_W1, "W2", T_W2, "b1", T_b1, "b2", T_b2)

# for i in 1:4
#     z_max = maximum(nodevalue.(getnodes(subgraph)[1][:z])[i,:,:])
#     z_min = minimum(nodevalue.(getnodes(subgraph)[1][:z])[i,:,:])
#     print("maximum value of z",string(i),": ", z_max)
#     print("If violate upper constraint: ", z_max >= zmax[i],"\n")
#     print("minimum value of z",string(i),": ", z_max)
#     print("If violate lower constraint: ", z_max >= zmax[i],"\n")
# end

# print("maximum value of hidden1: ", maximum(nodevalue.(getnodes(subgraph)[1][:hidden])), "\n")
# print("minimum value of hidden1: ", minimum(nodevalue.(getnodes(subgraph)[1][:hidden])), "\n")
#
# print("maximum value of sl1: ", maximum(nodevalue.(getnodes(subgraph)[1][:sl1])), "\n")
# print("minimum value of sl1: ", minimum(nodevalue.(getnodes(subgraph)[1][:sl1])), "\n")
#
# print("maximum value of hidden2: ", maximum(nodevalue.(getnodes(subgraph)[1][:hidden2])), "\n")
# print("minimum value of hidden2: ", minimum(nodevalue.(getnodes(subgraph)[1][:hidden2])), "\n")
#
# print("maximum value of sl2: ", maximum(nodevalue.(getnodes(subgraph)[1][:sl2])), "\n")
# print("minimum value of sl2: ", minimum(nodevalue.(getnodes(subgraph)[1][:sl2])), "\n")
#
# print("maximum value of W1: ", maximum(nodevalue.(getnodes(subgraph)[1][:W1])), "\n")
# print("minimum value of b1: ", minimum(nodevalue.(getnodes(subgraph)[1][:b1])), "\n")
#
# print("maximum value of W2: ", maximum(nodevalue.(getnodes(subgraph)[1][:W2])), "\n")
# print("minimum value of b2: ", minimum(nodevalue.(getnodes(subgraph)[1][:b2])), "\n")
