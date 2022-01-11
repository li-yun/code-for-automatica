import pyomo.environ as pyo
import numpy as np
W_1 = np.array([[-0.1033441, -0.11544891,  0.06762116,  0.00947731],
       [-0.06332282, -0.11026141, -0.04527647,  0.00873047],
       [ 0.01721568, -0.10477057, -0.15879607, -0.16422321],
       [ 0.01494209, -0.01147807,  0.02536875, -0.01251246],
       [ 0.00287376,  0.01535347,  0.0743145 , -0.21191224],
       [ 0.04112443,  0.06984211,  0.02478192,  0.05928976],
       [ 0.01339631,  0.28919548,  0.22285867, -0.27090208],
       [-0.13374296, -0.07744943,  0.0550906 ,  0.07674646],
       [-0.11309138,  0.14540432,  0.292291  , -0.23241068],
       [-0.09629228,  0.13091159,  0.06180625, -0.15889398]])
W_2 = np.array([[-0.71234458,  0.68737219,  1.11517332, -1.15804122,  1.23283323,
        -1.02574593, -1.13549462,  1.73010419,  0.45589093,  0.48405397],
       [ 0.37570565,  2.24051834,  0.32591709, -0.10380836,  0.12851047,
         0.13794112,  0.10308233,  0.24455835, -0.44314246, -0.31361955]])
b_1 = np.array([ 0.65273871,  1.22312449, -0.18173609,  0.18257732,  0.15355758,
        0.52173581,  0.00750649,  0.23737798,  0.39764447,  0.51570399])
b_2= np.array([ 0.24525605, -1.69521104])



N = 20
h = 3
xs = [14, 14, 14.2, 21.3]
us = [43.4, 35.4]
vmin = [-43.4, -35.4]
vmax = [16.6, 24.6]
zmin = [-6.5, -6.5, -10.7, -16.8]
zmax = [14, 14, 13.8, 6.7]
g = 981
A = [50.27, 50.27, 28.27, 28.27]
a = [0.233, 0.242, 0.127, 0.127]
gama = [0.4, 0.4]
alpha = 1000
l1 = 10
NS = 100

model  = pyo.AbstractModel()
model.dimz = pyo.RangeSet(0,3)
model.dimv = pyo.RangeSet(0,1)
model.NS = pyo.RangeSet(0,NS-1)
model.l1 = pyo.RangeSet(0,l1-1)
model.N = pyo.RangeSet(0,N-1)
model.N1 = pyo.RangeSet(0,N)
# values of initial scenarios
model.z_init = pyo.Param(model.dimz, model.NS)# initialize = lambda model,i,s: np.random.uniform(zmin[i],zmax[i]))

def z_bounds(model,i,s,t):
    return (zmin[i], zmax[i])
def zv_init(model, i,s,t):
    return np.random.rand()
model.var_z= pyo.Var(model.dimz, model.NS, model.N1, bounds = z_bounds, initialize = zv_init)

def v_bounds(model, i, k, j):
    return (vmin[i],vmax[i])
model.var_v = pyo.Var(model.dimv, model.NS, model.N, bounds = v_bounds, initialize = zv_init)

# initial state contraints for all scenarios
def z_rule(model,i,s):
    return model.var_z[i,s,0] == model.z_init[i,s]
model.cons_1 = pyo.Constraint(model.dimz, model.NS, rule = z_rule)

# system dynamic constraints
def z1_dyn(model, s, t): # i: ith scenario j: time step
    return (model.var_z[0, s, t+1] - model.var_z[0,s,t])/h == (-a[0]/A[0]*pyo.sqrt(2*g*(model.var_z[0,s,t]+xs[0]))\
            + a[2]/A[0]*pyo.sqrt(2*g*(model.var_z[2,s,t]+xs[2])) + gama[0]/A[0]*(model.var_v[0,s,t]+us[0]))
model.sys_1 = pyo.Constraint(model.NS, model.N, rule = z1_dyn)

def z2_dyn(model, s, t):
    return (model.var_z[1,s,t+1] - model.var_z[1,s,t])/h == (-a[1]/A[1]*pyo.sqrt(2*g*(model.var_z[1,s,t]+xs[1]))\
            +a[3]/A[1]*pyo.sqrt(2*g*(model.var_z[3,s,t]+xs[3])) + gama[1]/A[1]*(model.var_v[1,s,t]+us[1]))
model.sys_2 = pyo.Constraint(model.NS, model.N, rule = z2_dyn)

def z3_dyn(model, s,t):
    return (model.var_z[2,s,t+1]-model.var_z[2,s,t])/h == (-a[2]/A[2]*pyo.sqrt(2*g*(model.var_z[2,s,t]+xs[2]))\
            +(1-gama[1])/A[2]*(model.var_v[1,s,t]+us[1]))
model.sys_3 = pyo.Constraint(model.NS, model.N, rule = z3_dyn)

def z4_dyn(model, s, t):
    return (model.var_z[3,s,t+1]-model.var_z[3,s,t])/h == (-a[3]/A[3]*pyo.sqrt(2*g*(model.var_z[3,s,t]+xs[3]))\
                                      + (1-gama[0])/A[3]*(model.var_v[0,s,t]+us[0]))
model.sys_4 = pyo.Constraint(model.NS, model.N, rule = z4_dyn)


# NN weights and bias
def W1_init(model,i,j):
    return W_1[j,i]
def b1_init(model, i):
    return b_1[i]
def W2_init(model,i,j):
    return W_2[j,i]
def b2_init(model, i):
    return b_2[i]


model.W1 = pyo.Var(model.dimz, model.l1, initialize = W1_init, bounds = (None,None))
model.b1 = pyo.Var(model.l1, initialize = b1_init, bounds = (None, None))
model.W2 = pyo.Var(model.l1, model.dimv, initialize = W2_init, bounds = (None,None))
model.b2 = pyo.Var(model.dimv, initialize = b2_init, bounds = (None,None))

def hidden_init(model, s,t,i):
    return np.random.rand()
model.hidden1 = pyo.Var(model.NS, model.N, model.l1, initialize = hidden_init, bounds = (-1,1))
model.sl1  = pyo.Var(model.NS, model.N, model.l1, initialize = hidden_init, bounds = (-1e3,1e3))

def layer1_cons1(model, s,t,j):
    return 0 == (model.sl1[s,t,j] - model.W1[0,j]*model.var_z[0,s,t] - model.W1[1,j]*model.var_z[1,s,t] \
                 - model.W1[2,j]*model.var_z[2,s,t] - model.W1[3,j]*model.var_z[3,s,t] - model.b1[j])
def layer1_cons2(model, s,t,j):
    return 0 == model.hidden1[s,t,j] - pyo.tanh(model.sl1[s,t,j])

model.layer11 = pyo.Constraint(model.NS, model.N, model.l1, rule = layer1_cons1)
model.layer12 = pyo.Constraint(model.NS, model.N, model.l1, rule = layer1_cons2)

model.hidden2 = pyo.Var(model.NS, model.N, model.dimv, bounds = (-1,1), initialize = hidden_init)
model.sl2 = pyo.Var(model.NS, model.N, model.dimv, initialize = hidden_init, bounds = (-1e3,1e3))

def layer2_cons1(model, s,t,i):
    return (0 == model.sl2[s,t,i] - sum(model.W2[j,i]*model.hidden1[s,t,j] for j in range(l1)) - model.b2[i])
model.layer21 = pyo.Constraint(model.NS, model.N, model.dimv, rule = layer2_cons1)

def layer2_cons2(model, s,t,i):
    return 0 == model.hidden2[s,t,i] - pyo.tanh(model.sl2[s,t,i])
model.layer22 = pyo.Constraint(model.NS, model.N, model.dimv, rule = layer2_cons2)

model.input = pyo.Constraint(model.NS, model.N, model.dimv, rule = lambda model,s,t,i: 0 == (model.var_v[i,s,t] - \
                            (vmax[i]+vmin[i])/2 - (vmax[i]-vmin[i])/2*model.hidden2[s,t,i]))

model.ter_cost = pyo.Var(model.NS, bounds = (None, alpha), initialize = 0)

#def terminal_cons(model, s):
    return 6.55*(model.var_z[0,s,N])**2 + 6.55*(model.var_z[1,s,N])**2 + 7.92*(model.var_z[2,s,N])**2 + 31.7*(model.var_z[3,s,N])**2 \
    == model.ter_cost[s]
#model.terminal = pyo.Constraint(model.NS, rule = terminal_cons)

def ObjRule2(model):
    return sum((model.var_z[0,s,t])**2 for s in range(NS) for t in range(1,N)) + sum((model.var_z[1,s,t])**2 for s in range(NS)\
                for t in range(1,N)) + 1e-2*sum((model.var_v[0,s,t])**2 for s in range(NS) for t in range(N)) \
                + 1e-2*sum((model.var_v[1,s,t])**2 for s in range(NS) for t in range(N)) \
           + sum(6.55*(model.var_z[0,s,N])**2 + 6.55*(model.var_z[1,s,N])**2 + 7.92*(model.var_z[2,s,N])**2 + \
                 31.7*(model.var_z[3,s,N])**2 for s in range(NS))

def ObjRule1(model):
    return 1e-3*(sum(model.W1[i,j]**2 for i in range(4) for j in range(l1)) + sum(model.b1[i]**2 for i in range(l1))\
                +sum(model.W2[i,j]**2 for i in range(l1) for j in range(2))+sum(model.b2[i]**2 for i in range(2)))

model.SecondStageCost = pyo.Expression(rule = ObjRule2)
model.FirstStageCost = pyo.Expression(rule = ObjRule1)

def total_cost(model):
    return model.FirstStageCost + model.SecondStageCost
model.cost = pyo.Objective(rule = total_cost, sense = pyo.minimize)







