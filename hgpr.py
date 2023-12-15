import numpy as np

class HGPR():
    """
    The class implements Heterscedastic Gaussian Process Regression Algorithm.
    The algorithm is based on the paper "Most Likely Heteroscedastic Gaussian Process Regression" by 
    Kristian Kersting, Christian Plagemann, Patrick Pfaff and Wolfram Burgard
    http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf

    Params:
        model: GaussianProcessRegressor from gpr_custom.py. Fits the target.
        model_noise: GaussianProcessRegressor from gpr_custom.py. Fits the noise.
        num_iters: number of iterations until the model is converged (according to item 5 from Section 4 Kersting et al.)
        sample_size: number of points to be generated (corresponds to s in the formula from Section 4 Kersting et al.)
    """
    
    def __init__(
        self,
        model=None,
        model_noise=None,
        num_iters=5,
        sample_size=150,
        ):
        self.model = model
        self.model_noise = model_noise
        self.num_iters = num_iters
        self.sample_size = sample_size

    def fit(self, X, y, print_noise_rmse=False):
        """
        Fit function, the same architecture as in sklearn module.

        Paramas:
            X:  input data
            y:  output
            print_noise_rmse: if True, prints rmse of the noise model at each iteration
        """
        
        for i in range(self.num_iters):

            if i == 0:
                self.model.alpha_fit=True
                self.model.noise_x_dep=None
                self.model.fit(X, y)
            else:
                self.model.alpha_fit=False
                self.model.alpha=1
                self.model.noise_x_dep = noise_x_dep
                self.model.fit(X, y)


            mean_pred, std_pred =  self.model.predict(X, return_std=True)  
            if i > 0:
                std_pred= np.sqrt(std_pred**2+np.exp(self.model_noise.predict(X)))

                
            # Fit noise
            
            # Define sample matrix t_i^j from Section 4 Kersting et al.
            sample_matrix = np.zeros((len(y), self.sample_size))

            for j in range(0, self.sample_size):
                sample_matrix[:, j] = np.random.multivariate_normal(mean_pred.reshape(len(mean_pred)), np.eye(len(std_pred))*std_pred)

            # Estimate variance according to the formula from Section 4 Kersting et al.
            variance_estimator = (0.5 / self.sample_size) * np.sum((np.asarray(y) - sample_matrix.T) ** 2, axis=0)
            std_estimator = np.log(variance_estimator+10**(-10)) #np.sqrt(variance_estimator)
            
            self.model_noise.fit(X, std_estimator)

            noise_x_dep = np.exp(self.model_noise.predict(X))
            
            if print_noise_rmse:
                print('RMSE_noise = ', np.sqrt(mean_squared_error(self.model_noise.predict(X), std_estimator)))
            
            # At the final iteration step we have to update the input-dependent noise in the model 
            if i == (self.num_iters-1):
                self.model.noise_x_dep = noise_x_dep
        
        return self
    
    def predict(self, X, return_std=False, return_al_std=False, return_ep_std=False):
        """
        Make a prediction for X input. Standard deviation can be separated to two types: aleatoric (inherent noise from the data)
        and epistemic (uncertainty of the model itself).


        Params:
            X:  input data for which to make a prediction
            return_std: if True, returns mean and the full std (aleatoric+epistemic)
            return_al_std: if True, returns mean and aleatoric std
            return_ep_std: if True, returns mean and epistemic std

        """
        
        if return_std==False and return_al_std==False and return_ep_std==False:
            result = self.model.predict(X)
        else:
            mean, std_ep = self.model.predict(X, return_std=True) 
            if return_ep_std:
                # Epistemic std
                std = std_ep
            if return_al_std:
                # Aleatoric std
                std = np.sqrt(np.exp(self.model_noise.predict(X)))
            if return_std:
                # Full std (epistemic + aleatoric)
                var_ep=std_ep**2
                var_al = np.exp(self.model_noise.predict(X))
                std=np.sqrt(var_ep+var_al)
            
            result = mean, std
                
            
        return result