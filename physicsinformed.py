import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from utilities import get_derivative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedContinuous:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, layers, t0,x0,y0,z0,tb,xb,yb,zb, t_xz_lb,x_xz_lb,y_xz_lb,z_xz_lb, t_xz_ub,x_xz_ub,
                                      y_xz_ub,z_xz_ub, t_yz_lb,x_yz_lb,y_yz_lb,z_yz_lb, t_yz_ub,x_yz_ub,
                                      y_yz_ub,z_yz_ub, t_int, t_f, x_int, y_int, z_int_l3, x_f, y_f, z_f, Q1):
        """Construct a PhysicsInformedBar model"""

        self.t0 = t0
        self.x0 = x0
        self.y0 = x0
        self.z0 = x0
        self.tb = tb
        self.xb = xb
        self.yb = xb
        self.zb = xb
        self.t_xz_lb = t_xz_lb
        self.x_xz_lb = x_xz_lb
        self.y_xz_lb = y_xz_lb
        self.z_xz_lb = z_xz_lb 
        self.t_xz_ub = t_xz_ub
        self.x_xz_ub = x_xz_ub
        self.y_xz_ub = y_xz_ub
        self.z_xz_ub =  z_xz_ub
        self.t_yz_lb = t_yz_lb
        self.x_yz_lb = x_yz_lb
        self.y_yz_lb = y_yz_lb
        self.z_yz_lb = z_yz_lb 
        self.t_yz_ub = t_yz_ub
        self.x_yz_ub = x_yz_ub
        self.y_yz_ub = y_yz_ub
        self.z_yz_ub = z_yz_ub 
        self.t_int =  t_int
        self.t_f = t_f 
        self.x_int = x_int 
        self.y_int =  y_int
        self.z_int_l3 =  z_int_l3
        self.x_f = x_f 
        self.y_f = y_f 
        self.z_f =  z_f
        self.Q1 = Q1
        
        self.l1 = 0.08*10**(-3) # thickness of the layer(m)
        self.rho1 = 0.0012*10**6 # density of the layer (kg/m^3)
        self.C1 = 3.6*10**(3) # specific heat (J/kg. deg Celsius)
        self.Cb1 = 4.2*10**(3) # specific heat of the blood (J/kg. deg Celsius)
        self.K1 = 0.00026*10**(3) # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb1 = 0 # Blood perfusion rate (g/mm^3. s)
        self.alpha1 = 0.1 #laser absorbtivity of the first layer
        self.Reff1 = 0.93 # Laser reflectivity of the first layer
        
        self.l2 = 2*10**(-3) # thickness of the layer(m)
        self.rho2 = 0.0012*10**6 # density of the layer (kg/m^3)
        self.C2 = 3.4*10**(3) # specific heat (J/kg. deg Celsius)
        self.Cb2 = 4.2*10**(3) # specific heat of the blood (J/kg. deg Celsius)
        self.K2 = 0.00052*10**(3) # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb2 = 5 * 10**(-7)*10**6 # Blood perfusion rate (kg/m^3. s)
        self.alpha2 = 0.08 #laser absorbtivity of the second layer
        self.Reff2 = 0.93 # Laser reflectivity of the second layer 
        
        self.l3 = 10*10**(-3) # thickness of the layer(m)
        self.rho3 = 0.001*10**6 # density of the layer (kg/m^3)
        self.C3 = 3.06*10**(3) # specific heat (J/kg. deg Celsius)
        self.Cb3 = 4.2*10**(3) # specific heat of the blood (J/kg. deg Celsius)
        self.K3 = 0.00021*10**(3) # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb3 = 5 * 10**(-7)*10**6 # Blood perfusion rate (kg/m^3. s)
        self.alpha3 = 0.04 #laser absorbtivity of the third layer
        self.Reff3 = 0.93 # Laser reflectivity of the third layer 
        
        self.P = 6.4
       
        self.model = self.build_model(layers[0], layers[1:-1], layers[-1])
        self.train_cost_history = []
        self.f1 = None
        self.f2 = None
        self.f3 = None

    def build_model(self, input_dimension, hidden_dimension, output_dimension):
        """Build a neural network of given dimensions."""

        nonlinearity = torch.nn.Tanh()
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(nonlinearity)
        for i in range(len(hidden_dimension)-1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
            modules.append(nonlinearity)

        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))

        model = torch.nn.Sequential(*modules).to(device)
        print(model)
        print('model parameters on gpu:', next(model.parameters()).is_cuda)
        return model

    def u_nn(self, t, x , y, z):
        """Predict temperature at (t,x)."""

        u = self.model(torch.cat((t,x,y,z),1))
        return u

    def f_nn(self, t, x ,y, z , layer_lth_function = None): # if we want the function for the interfaces of the tissue we                                                                     # can use layer_lth_function which takes values 1,2 or 3
        """Compute differential equation -> Pennes heat equation"""

        u = self.u_nn(t, x, y, z)
        u_t = get_derivative(u, t, 1)
        u_x = get_derivative(u, x, 1)
        u_y = get_derivative(u, y, 1)
        u_z = get_derivative(u, z, 1)
        u_xx = get_derivative(u, x, 2)
        u_yy = get_derivative(u, y, 2)
        u_zz = get_derivative(u, z, 2)
        
        '''
        #writing these outside 'if' may increase time for simulation because all 3 calculations will be performed every time
        if layer_lth_function == 1:
             f = rho1*C1*u_t - K1*(u_xx+u_yy+u_zz) + Wb1*Cb1*u - self.Q1(x,y,z) 
        if layer_lth_function == 2:
             f = rho2*C2*u_t - K2*(u_xx+u_yy+u_zz) + Wb2*Cb2*u - self.Q2(x,y,z)
        if layer_lth_function == 3:
             f = rho3*C3*u_t - K3*(u_xx+u_yy+u_zz) + Wb3*Cb3*u - self.Q3(x,y,z)
        '''
        ''' # there may be a case where z[0,0]==lic[0] or lic[1] or lic[2]...?!!may be use sum(z>lic[i])>2 then fi
        if z[0,0] < lic[0]:       # for layer l1 of the tissue
            f = f1 #f = rho1*C1*u_t - K1*(u_xx+u_yy+u_zz) + Wb1*Cb1*u - self.Q1(x,y,z)
        if z[0,0] < lic[1] and z[0,0]>lic[0]:    # for layer l2 of the tissue
            f = f2 #f = rho2*C2*u_t - K2*(u_xx+u_yy+u_zz) + Wb2*Cb2*u - self.Q2(x,y,z)
        if z[0,0] < lic[2] and z[0,0]>lic[1]:    # for layer l3 of the tissue
            f = f3 #f = rho3*C3*u_t - K3*(u_xx+u_yy+u_zz) + Wb3*Cb3*u - self.Q3(x,y,z)'''    
        f = self.rho1*self.C1*u_t - self.K1*(u_xx+u_yy+u_zz) + self.Wb1*self.Cb1*u - self.Q1(x,y,z)
        return f

   

    def cost_function(self):
        """Compute cost function."""
        
        
        u0_pred = self.u_nn(self.t0, self.x0, self.y0, self.z0)
        
        # initial condition loss @ t = 0  #include all layer loss if assumtion is not taken
        
        mse_0 = torch.mean((u0_pred)**2)
        
        # surface boundary condition loss @ z = 0  #z=0 is at layer 1 so no other layer loss eq required
        
        u_b_pred = self.u_nn(self.tb, self.xb, self.yb, self.zb)
        u_zb_b_pred = get_derivative(u_b_pred, self.zb, 1)
        
        mse_b = torch.mean(u_zb_b_pred**2) 
        
        # interface surface boundary condition loss, for our assumption only bottom layer is considered @ z = l1+l2+l3  
        # we will have to include other two interface(z=l1 & z=l1+l2) loss equtions if assumption is neglected
        '''
        u_int_pred_l1 = self.u_nn(self.t_int, self.x_int, self.y_int, self.z_int_l1)   # u predicted at the interface1 &z=lic[0]
        u_z_int_pred_l1 = get_derivative(u_int_pred_l1, self.z_int_l1, 1)     # derivative wrt z_int_l1      
        u_int_pred_l2 = self.u_nn(self.t_int, self.x_int, self.y_int, self.z_int_l2)   # u predicted at the interface 2
        u_z_int_pred_l2 = get_derivative(u_int_pred_l2, self.z_int_l2, 1)     # derivative wrt z_int_l2'''
        u_int_pred_l3 = self.u_nn(self.t_int, self.x_int, self.y_int, self.z_int_l3)#u predicted at interface3(bottom surface)
        u_z_int_pred_l3 = get_derivative(u_int_pred_l3, self.z_int_l3, 1)     # derivative wrt z_int_l3
        
        #mse_b+= torch.mean((u_z_int_pred_l3)**2) # loss for bottom layer
        
        ## lateral surface walls loss condition (4 walls)
        #for xz plane walls
        u_xz_lb_pred = self.u_nn(self.t_xz_lb, self.x_xz_lb, self.y_xz_lb, self.z_xz_lb) #wall 1
        u_x_xz_lb_pred = get_derivative(u_xz_lb_pred, self.x_xz_lb, 1)
        u_y_xz_lb_pred = get_derivative(u_xz_lb_pred, self.y_xz_lb, 1)
        u_z_xz_lb_pred = get_derivative(u_xz_lb_pred, self.z_xz_lb, 1)
        
        u_xz_ub_pred = self.u_nn(self.t_xz_ub, self.x_xz_ub, self.y_xz_ub, self.z_xz_ub) #wall 2 opposite to wall 1
        u_x_xz_ubb_pred = get_derivative(u_xz_ub_pred, self.x_xz_ub, 1)
        u_y_xz_ubb_pred = get_derivative(u_xz_ub_pred, self.y_xz_ub, 1)
        u_z_xz_ubb_pred = get_derivative(u_xz_ub_pred, self.z_xz_ub, 1)
        
        mse_b+= torch.mean(u_x_xz_lb_pred**2+u_y_xz_lb_pred**2+u_z_xz_lb_pred**2+
                           u_x_xz_ubb_pred**2+u_y_xz_ubb_pred**2+u_z_xz_ubb_pred**2)
        
        #for yz plane walls
        u_yz_lb_pred = self.u_nn(self.t_yz_lb, self.x_yz_lb, self.y_yz_lb, self.z_yz_lb) #wall 3
        u_x_yz_lb_pred = get_derivative(u_yz_lb_pred, self.x_yz_lb, 1)
        u_y_yz_lb_pred = get_derivative(u_yz_lb_pred, self.y_yz_lb, 1)
        u_z_yz_lb_pred = get_derivative(u_yz_lb_pred, self.z_yz_lb, 1)
        
        u_yz_ub_pred = self.u_nn(self.t_yz_ub, self.x_yz_ub, self.y_yz_ub, self.z_yz_ub) #wall 4 opposite to wall 3
        u_x_yz_ubb_pred = get_derivative(u_yz_ub_pred, self.x_yz_ub, 1)
        u_y_yz_ubb_pred = get_derivative(u_yz_ub_pred, self.y_yz_ub, 1)
        u_z_yz_ubb_pred = get_derivative(u_yz_ub_pred, self.z_yz_ub, 1)
        
        mse_b+= torch.mean(u_x_yz_lb_pred**2+u_y_yz_lb_pred**2+u_z_yz_lb_pred**2+
                           u_x_yz_ubb_pred**2+u_y_yz_ubb_pred**2+u_z_yz_ubb_pred**2)
        
        # for the function loss
        f_pred = self.f_nn(self.t_f,self.x_f,self.y_f,self.z_f)
        mse_f = np.exp(-18)*torch.mean((f_pred)**2)  # 5e-4 is a good value for balancing

        return 10**4*mse_0, 10**4*mse_b, mse_f

    def train(self, epochs, optimizer='Adam', **kwargs):
        """Train the model."""

        # Select optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

        ########################################################################
        elif optimizer=='L-BFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters())

            def closure():
                self.optimizer.zero_grad()
                mse_0, mse_b, mse_f = self.cost_function()
                cost = mse_0 + mse_b + mse_f
                cost.backward(retain_graph=True)
                return cost
        ########################################################################

        # Training loop
        for epoch in range(epochs):
            mse_0, mse_b, mse_f = self.cost_function()
            cost = mse_0 + mse_b + mse_f
            self.train_cost_history.append([cost.cpu().detach(), mse_0.cpu().detach(), mse_b.cpu().detach(), mse_f.cpu().detach()])

            if optimizer=='Adam':
                # Set gradients to zero.
                self.optimizer.zero_grad()

                # Compute gradient (backwardpropagation)
                cost.backward(retain_graph=True)

                # Update parameters
                self.optimizer.step()

            ########################################################################
            elif optimizer=='L-BFGS':
                self.optimizer.step(closure)
            ########################################################################

            if epoch % 100 == 0:
                # print("Cost function: " + cost.detach().numpy())
                print(f'Epoch ({optimizer}): {epoch}, Cost: {cost.detach().cpu().numpy()}, Bound_loss: {mse_b.detach().cpu().numpy()}, Fun_loss: {mse_f.detach().cpu().numpy()}')

    def plot_training_history(self, yscale='log'):
        """Plot the training history."""

        train_cost_history = np.asarray(self.train_cost_history, dtype=np.float32)

        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function C")
        plt.yscale(yscale)

        # Plot data
        mse_0, mse_b, mse_f = ax.plot(train_cost_history[:,1:4])
        mse_0.set(color='r', linestyle='dashed', linewidth=2)
        mse_b.set(color='k', linestyle='dotted', linewidth=2)
        mse_f.set(color='silver', linewidth=2)
        plt.legend([mse_0, mse_b, mse_f], ['MSE_0', 'MSE_b', 'MSE_f'], loc='lower left')
        plt.tight_layout()
        plt.savefig('cost-function-history.eps')
        plt.show()





