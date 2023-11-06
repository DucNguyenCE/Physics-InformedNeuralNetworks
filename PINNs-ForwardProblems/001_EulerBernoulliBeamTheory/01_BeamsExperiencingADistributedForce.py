import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os, time, scipy.optimize, scipy.io, sys

# Hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'} 0 (default) show all, 1 to filter out INFOR logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs

np.random.seed(1234)
tf.random.set_seed(1234)
print("Start! Check TensorFlow version: {}".format(tf.__version__))

########################################################################################################################
##### PINN ALGORITHM #####

class Sequentialmodel(tf.Module):
    def __init__(self, layers, name=None):
        self.W = [] # Weight and biases
        self.parameters = 0 # Total number of parameters
        
        # Generalize the initial parameters of the Physics-Informed Neural Network (PINN).
        for i in range(len(layers)-1):
            input_dim = layers[i] # Number of inputs
            output_dim = layers[i+1] # Number of outputs
            # Initialization: *Xavier*
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            # Add weights and bias to the parameters vectors
            w = tf.Variable(w,trainable=True,name='w'+str(i+1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]),dtype='float64'),trainable=True,name='b'+str(i+1))
            self.W.append(w)
            self.W.append(b)
            self.parameters += input_dim * output_dim + output_dim
            
    def evaluate(self,x):
        # Normalize the inputs into the range [0,1]
        x = (x-lb)/(ub-lb)
        a = x
        for i in range(len(layers)-2):
            z = tf.add(tf.matmul(a,self.W[2*i]),self.W[2*i+1]) 
            a = tf.nn.tanh(z)   # Activation: tanh(x)
        a = tf.add(tf.matmul(a,self.W[-2]),self.W[-1]) # Output layers use linear function.
        return a
    
    def get_weights(self):
        parameters_1d = []
        for i in range(len(layers)-1):
            w_1d = tf.reshape(self.W[2*i],[-1])
            b_1d = tf.reshape(self.W[2*i+1],[-1])
            parameters_1d = tf.concat([parameters_1d,w_1d],0)
            parameters_1d = tf.concat([parameters_1d,b_1d],0)
        return parameters_1d
    
    def set_weights(self, parameters):
        for i in range(len(layers)-1):
            shape_w = tf.shape(self.W[2*i]).numpy()
            size_w = tf.size(self.W[2*i]).numpy()
            shape_b = tf.shape(self.W[2*i+1]).numpy()
            size_b  = tf.size(self.W[2*i+1]).numpy()
            pick_w = parameters[0:size_w]
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) 
            parameters = np.delete(parameters, np.arange(size_w),0)
            pick_b = parameters[0:size_b]
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b))
            parameters = np.delete(parameters, np.arange(size_b),0)
            
    def loss_BC(self,lb,ub):
        x_lb = tf.Variable(lb, dtype='float64', trainable=False)
        x_ub = tf.Variable(ub, dtype='float64', trainable=False)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_lb, x_ub])
            u_lb = self.evaluate(x_lb)
            u_lb_x = tape.gradient(u_lb,x_lb)
            u_lb_xx = tape.gradient(u_lb_x,x_lb)
            u_ub = self.evaluate(x_ub)
            u_ub_x = tape.gradient(u_ub,x_ub)
            u_ub_xx = tape.gradient(u_ub_x,x_ub)

        u_ub_xxx = tape.gradient(u_ub_xx,x_ub)
        u_lb_xxx = tape.gradient(u_lb_xx,x_lb)
        del tape

        '''u:      deformation;    u_x:    rotation;    u_xx:   shear force;    u_xxx:  moment   '''
        loss_bc1 = BC[0]*tf.reduce_mean(tf.square(u_lb)) + abs(BC[0]-1) * tf.reduce_mean(tf.square(u_lb_xx))
        loss_bc2 = BC[1]*tf.reduce_mean(tf.square(u_lb_x)) + abs(BC[1]-1) * tf.reduce_mean(tf.square(u_lb_xxx))
        loss_bc3 = BC[2]*tf.reduce_mean(tf.square(u_ub)) + abs(BC[2]-1) * tf.reduce_mean(tf.square(u_ub_xx))
        loss_bc4 = BC[3]*tf.reduce_mean(tf.square(u_ub_x)) + abs(BC[3]-1) * tf.reduce_mean(tf.square(u_ub_xxx))
        return loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        x_f = g[:,0:1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)            
            z = self.evaluate(x_f)
            u_x = tape.gradient(z,x_f)
            u_xx = tape.gradient(u_x,x_f)
            u_xxx = tape.gradient(u_xx,x_f)

        u_xxxx = tape.gradient(u_xxx,x_f)
        # Computes the gradient using operations recorded in context of this tape.
        del tape
        
        f = u_xxxx + 1 
        loss_f = tf.reduce_mean(tf.square(f))
        return loss_f
    
    def loss(self,x,lb,ub):
        loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss_BC(lb, ub)
        loss_pde = self.loss_PDE(x)        
        return loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables) # trainable_variables is W (w+b)
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
            loss_val = loss_pde + loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
        grads = tape.gradient(loss_val, self.trainable_variables)
        del tape
        
        grads_1d = [] # flatten grads
        for i in range(len(layers)-1):
            grads_w_1d = tf.reshape(grads[2*i],[-1]) # flatten weights
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) # flatten biases
            
            grads_1d = tf.concat([grads_1d, grads_w_1d],0) #concat grad_weights
            grads_1d = tf.concat([grads_1d, grads_b_1d],0) #concat grad_biases
            
        return loss_val.numpy(), grads_1d.numpy()
    
    def optimizer_callback(self, parameters):
        global counter
        if counter % 100 == 0:
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
            tf.print("%6d" % counter, "%10.3E" % loss_pde, "%10.3E" % loss_bc1,  "%10.3E" % loss_bc2, "%10.3E" % loss_bc3, "%10.3E" % loss_bc4)
        counter += 1

########################################################################################################################
##### DATA PREPARATION #####
BC = [1, 0, 1, 0] # Set up the boundray conditions
# Check whether the boundary conditions are correct or not.
if any(element not in [0,1] for element in BC):
    print("Your inputs are incorrect. Only 0 or 1 can be used!")
    sys.exit()
# Check whether the boundary conditions are stable or unstable.
if sum(BC) < 2 or (sum(BC)== 2 and BC[1] == BC[3] and BC[1] == 1):
    print("The structure is not stable.")
    sys.exit()

# Import the training data
train = np.loadtxt("train.dat")
x_train = train[:,0:1]
counter = 0
lb = np.array([np.min(x_train)])[:,None] # Low boundary
ub = np.array([np.max(x_train)])[:,None] # Up boundary

########################################################################################################################
##### MODEL TRAINING AND TESTING #####
# Set up the neural networks
layers = np.array([1,20,20,20,1]) 
PINN = Sequentialmodel(layers)
init_params = PINN.get_weights().numpy()

start_time = time.time() # Start measuring the calculation time of the program.
# Train the model with Scipy L-BFGS optimizer
results = scipy.optimize.minimize(fun=PINN.optimizerfunc,
                                  x0 = init_params,
                                  args=(),
                                  method='L-BFGS-B',
                                  jac=True, 
                                  callback=PINN.optimizer_callback,
                                  options={'disp':None,
                                          'maxcor': 200,
                                          'ftol': 1*np.finfo(float).eps,
                                          'gtol': 5e-7,
                                          'maxfun': 10000, 
                                          'maxiter': 5000, 
                                          'iprint': -1,
                                          'maxls': 20})
elapsed = time.time() - start_time # Calculate the time of the program.
print('Training time: %.2f' %(elapsed))
PINN.set_weights(results.x) # Set the calculated weights in the PINN.

########################################################################################################################
##### FEM #####
def FEM_EBBeams(BC,num_elements):
    # Define the beam properties
    length = 1.0  # Length of the beam (in meters)
    element_length = length / num_elements  # Length of each element

    # Initialize global stiffness matrix and load vector
    num_nodes = num_elements + 1
    num_DOFs = num_nodes*2 # Number of degrees of freedoms
    K_global = np.zeros((num_DOFs, num_DOFs))
    F_global = np.zeros((num_DOFs,1))

    # Define the element stiffness matrix
    def element_stiffness(L):
        return np.array([[12/L**3, 6/L**2, -12/L**3, 6/L**2],[6/L**2, 4/L, -6/L**2, 2/L], [-12/L**3, -6/L**2, 12/L**3, -6/L**2],[6/L**2, 2/L, -6/L**2, 4/L]])

    # Compute the global stiffness matrix and load vector
    for i in range(num_elements):
        # Assemble element stiffness matrix
        ke = element_stiffness(element_length)

        # Add the element's contribution to the global stiffness matrix
        K_global[2*i:2*i+4, 2*i:2*i+4] += ke

        # Apply distributed load to the global load vector
        F_global[2*i:2*i+4] -= np.array([[element_length/2], [element_length**2/12], [element_length/2], [-element_length**2/12]])

    # Apply boundary conditions (fixed support at one end)
    K = K_global
    F = F_global
    if BC[3] == 1:
        K = np.delete(K,num_DOFs-1,0)
        K = np.delete(K,num_DOFs-1,1)
        F = np.delete(F,num_DOFs-1)

    if BC[2] == 1:
        K = np.delete(K,num_DOFs-2,0)
        K = np.delete(K,num_DOFs-2,1)
        F = np.delete(F,num_DOFs-2)

    if BC[1] == 1:
        K = np.delete(K,1,0)
        K = np.delete(K,1,1)
        F = np.delete(F,1)

    if BC[0] == 1:
        K = np.delete(K,0,0)
        K = np.delete(K,0,1)
        F = np.delete(F,0)

    # Solve for displacements
    T = la.solve(K, F)
    if BC[0] == 1:
        T = np.append(0, T)
    if BC[3] == 1:
        T = np.append(T,0)
    if BC[1] == 1:
        T = np.insert(T,1,0,axis = 0)
    if BC[2] == 1:  
        T = np.insert(T,-1,0,axis = 0)

    return T[::2]
########################################################################################################################
##### USING THE PINN TO CALCULATE #####
# Import the testing data
test = np.loadtxt("test.dat")
x_test = test[:,0:1]
x_test = x_test[x_test[:, 0].argsort()] # Sort the testing data in ascending order.
y_test = PINN.evaluate(x_test)

# Calculate using FEM
num_elements = 30 # Number of elements
x_FEM = np.arange(0,1.0001,1/num_elements)
y_FEM = FEM_EBBeams(BC, num_elements)

fig, ax = plt.subplots()
ax.plot(x_FEM, y_FEM,'bo', linewidth=2.0)
ax.plot(x_test, y_test, linewidth=2.0)
ax.set(xlim=(-0.1, 1.1), xticks=np.arange(0, 1), ylim=(-0.2, 0.1), yticks=np.arange(-0.2, 0.1))

plt.show()