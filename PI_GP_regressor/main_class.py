from jax import numpy as jnp
from jax import jit, grad, vmap, jacfwd
import matplotlib.pyplot as plt
import numpy as np
from skopt.space import Real
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import qmc
import GPy

class PhysicsInformedGP_regressor():
    """class for the GP"""
    _name_kernel = ""
    def __init__(self,kernel: callable, timedependence: bool,params:list) -> None:
        assert callable(kernel[0]), "Please provide a valid kernel" 
        assert(type(timedependence) == bool), "Please provide a valid boolean value for timedependence"

        self.gram_matrix = kernel[0]
        self.k_uu,self.k_uf, self.k_fu,self.k_ff = kernel[1], kernel[2], kernel[3], kernel[4]
        self.timedependence = timedependence
        self.params = dict(zip(params, 1*np.ones(len(params))))
        self.X = None
        self.Y = None
        self.u_train, self.f_train = None, None
        self.targets = None
        self.raw_data = None
        self.noise = None
        self.result = None
        self.mean_u = None
        self.var_u = None
        self.mean_f = None
        self.var_f = None
        self.validation_set = None
        self.MSE = {"u":None,"f":None}
        self.filename = None
        self.results_list = None
        self.GPy_models = None
        
    def __str__(self) -> str:
        string = "-----------------------------------------------\n"
        string += "GP with kernel: " + str(self.__class__._name_kernel) + "\n"
        string += "Training data: " + str(self.X.shape) + "\n"
        string += "Training targets: " + str(self.targets.shape) + "\n"
        string += "Hyperparameters: " + str(self.params) + "\n"
        string += "Log marginal likelihood: " + str(self.result.fun) + "\n"
        string += "Mean squared error: " + str(self.MSE) + "\n"
        string += "-----------------------------------------------\n"
        return string
    def set_name_kernel(self,name:str):
        """sets the name of the kernel"""
        self._name_kernel = name
        pass
    def set_training_data(self,filename:str, n_training_points,noise,seeds_training: list = [40,14]):
        """sets the training data and the raw data"""
        if self.timedependence:
            x_u, x_f, t_u, t_f, u_train, f_train,self.raw_data  = self.get_data_set_2d(filename, n_training_points,noise)
            self.X = np.hstack([x_u, t_u])
            self.Y = np.hstack([x_f, t_f])
            self.targets = np.concatenate([u_train, f_train])
            self.u_train = u_train
            self.f_train = f_train
            assert self.X.shape[1] == 2, "Please provide a valid training data set"

        else:
            x_u, u_train, x_f, f_train,self.raw_data = self.get_data_set_1d(filename, n_training_points,noise,seeds_training)
            self.X = x_u
            self.Y = x_f
            self.targets = np.concatenate([u_train, f_train])
            self.u_train = u_train
            self.f_train = f_train
        self.filename = filename
        self.noise = noise
        pass        
    def set_validation_data(self,n_validation_points):
        """sets the validation set"""
        assert self.X is not None, "Please set the training data first"
        if self.timedependence:
            self.validation_set = self.get_validation_set_2d(self.filename,n_validation_points,self.noise)
        else:
            self.validation_set = self.get_validation_set_1d(self.filename,n_validation_points,self.noise)
        pass

    def set_params(self, new_params: list):
        """sets the hyperparameters of the kernel"""
        for i,key in enumerate(self.params.keys()):
            self.params[key] = new_params[i]
        pass
    def get_params(self):
        """returns the hyperparameters of the kernel"""
        return np.array(list(self.params.values()))
    
    def log_marginal_likelohood(self, params):
        """computes the log marginal likelihood of the GP"""
        K = self.gram_matrix(self.X, self.Y, params,self.noise)
    
        L = jnp.linalg.cholesky(K + 1e-8 * jnp.zeros(K.shape)) #add some jitter for stability
    
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
        mll = 1/2 * jnp.dot(self.targets.T,alpha) +0.5*jnp.sum(jnp.log(jnp.diagonal(L))) + len(self.X)/2 * jnp.log(2*jnp.pi)
        return jnp.squeeze(mll)
    
    
    def log_marginal_likelihood_to_optimize(self):
        """returns the log marginal likelihood as a function of the hyperparameters, so it can be used in the optimization function"""
        def function_to_optimize(hyperparams):
            mll = self.log_marginal_likelohood(hyperparams)
            return mll
        return function_to_optimize
    

    def train(self,method: str, n_restarts: int, n_threads: int, opt_dictionary: dict) -> None:
        """optimizes the hyperparameters of the kernel"""
        assert self.X is not None, "Please set the training data first"
        assert method in ['L-BFGS-B', 'TNC', "CG"], "Please choose a valid optimization method"
        assert n_restarts > 0, "Please choose a valid number of restarts"
        if method == "CG":
            opt_result = self.optimization_restarts_parallel_CG(n_restarts, n_threads, opt_dictionary)
        elif method == "TNC":
            opt_result = self.optimization_restarts_parallel_TNC(n_restarts, n_threads, opt_dictionary)
        elif method == "L-BFGS-B":
            opt_result = self.optimization_restarts_parallel_LBFGSB(n_restarts, n_threads, opt_dictionary)
        self.set_params(opt_result.x)
        self.result = opt_result
        pass
    
    

    def optimization_restarts_parallel_CG(self, n_restarts: int, n_threads: int,opt_dictionary: dict) -> dict:
        """
            performs the optimization of the hyperparameters in parallel and returns the best result.
            n_restarts: number of restarts of the optimization
            n_threads: number of threads to use for the parallelization
            opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """
        
        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()
        
            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                           method='CG')
            return res

    
        results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        #all positive parameters
        results = [res for res in results if  np.all(res.x > 0) ]#and res.success
        self.result_list = results
        best_result = min(results, key=lambda x: x.fun)
        print(best_result)
        
        return best_result
    
    def optimization_restarts_parallel_TNC(self, n_restarts: int, n_threads: int,opt_dictionary: dict) -> dict:
        """
        performs the optimization of the hyperparameters in parallel and returns the best result.
        n_restarts: number of restarts of the optimization
        n_threads: number of threads to use for the parallelization
        opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """
        
        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()
            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                        method='TNC', jac=jit(self.grad_log_marginal_likelihood()),bounds=opt_dictionary['bounds'],
                        tol = opt_dictionary['gtol'])
            return res
        results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        #all positive parameters
        results = [res for res in results if  np.all(res.x > 0) and res.success]
        self.results_list = results
        best_result = min(results, key=lambda x: x.fun)
        print(best_result)
        
        return best_result
    
    def optimization_restarts_parallel_LBFGSB(self, n_restarts: int, n_threads: int,opt_dictionary: dict) -> dict:
        """
        performs the optimization of the hyperparameters in parallel and returns the best result.
        n_restarts: number of restarts of the optimization
        n_threads: number of threads to use for the parallelization
        opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """
        
        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()
            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                        method='L-BFGS-B', bounds=opt_dictionary['bounds'],
                       options={'gtol': opt_dictionary['gtol']})
            return res
        results = Parallel(n_jobs=n_threads)(delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        #all positive parameters
        results = [res for res in results if  np.all(res.x > 0) and res.success]
        self.results_list = results
        best_result = min(results, key=lambda x: x.fun)
        print(best_result)
        
        return best_result


    def predict_model(self,X_star):
        self.mean_u, self.var_u = self.predict_u(X_star)
        self.mean_f, self.var_f = self.predict_f(X_star)

    def predict_u(self, X_star):
        """predicts the mean and variance of the GP at the points X_star"""
        params = self.get_params()

        K = self.gram_matrix(self.X, self.Y, params,self.noise)
        L = jnp.linalg.cholesky(K + 1e-6 * jnp.zeros(K.shape)) #add some jitter for stability
        if self.timedependence:
            assert X_star.shape[1] == 2, "Please provide a valid test data set"
            q_1 = self.k_uu(X_star,self.X, params) 
            q_2 = self.k_uf(X_star, self.Y, params)
            q = jnp.hstack((q_1,q_2))

            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
            f_star = jnp.dot(q,alpha)
            alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, q.T))
            cov_f_star = self.k_uu(X_star,X_star, params) - q@alpha_var
            var = jnp.diag(cov_f_star)
        else:
            assert X_star.shape[1] == 1, "Please provide a valid test data set"
            q_1 = self.k_uu(X_star, self.X, params) 
            q_2 = self.k_uf(X_star, self.Y, params)
            q = jnp.hstack((q_1,q_2))
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
            f_star = jnp.dot(q,alpha)
            alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, q.T))
            cov_f_star = self.k_uu(X_star,X_star, params) - q@alpha_var
            var = jnp.diag(cov_f_star)
        return f_star, var
        
    def predict_f(self,X_star):
        """predicts the mean and variance of the GP at the points X_star
        X_star: points at which the GP is evaluated(for 2d case: (x,t) meshgrid)
        returns: mean and variance tuple of the GP at the points X_star
        """
        params = self.get_params()
     
        K = self.gram_matrix(self.X, self.Y, params,self.noise)
        L = jnp.linalg.cholesky(K + 1e-6 * jnp.zeros(K.shape)) #add some jitter for stability
        
        if self.timedependence:
            assert X_star.shape[1] == 2, "Please provide a valid test data set"
            q_1 = self.k_fu(X_star,self.X, params) 
            q_2 = self.k_ff(X_star, self.Y, params)
            q = jnp.hstack((q_1,q_2))
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
            f_star = jnp.dot(q,alpha)
            alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, q.T))
            cov_f_star = self.k_ff(X_star,X_star, params) - q@alpha_var
            var = jnp.diag(cov_f_star)
            
        else:
            assert X_star.shape[1] == 1, "Please provide a valid test data set"
            q_1 = self.k_fu(X_star, self.X, params) 
            q_2 = self.k_ff(X_star, self.Y, params)
            q = jnp.hstack((q_1,q_2))
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
            f_star = jnp.dot(q,alpha)
            alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, q.T))
            cov_f_star = self.k_ff(X_star,X_star, params) - q@alpha_var
            var = jnp.diag(cov_f_star)
            
        return f_star, var

    def error(self):
        """computes the mean squared error of the computed model"""
        assert self.validation_set is not None, "Please set the validation set first"
        if self.timedependence:
            x_star, t_star = self.validation_set[0].reshape(-1,1), self.validation_set[1].reshape(-1,1)
            X_star_u = np.hstack((x_star,t_star))
            x_star, t_star = self.validation_set[2].reshape(-1,1), self.validation_set[3].reshape(-1,1)
            X_star_f = np.hstack((x_star,t_star))
            u_values = self.validation_set[4]
            f_values = self.validation_set[5]
        else: 
            X_star_u = self.validation_set[0]
            X_star_f = self.validation_set[2]
            u_values = self.validation_set[1]
            f_values = self.validation_set[3]


        mean_validation_set_u,var = self.predict_u(X_star_u)
        mean_validation_set_f,var = self.predict_f(X_star_f)
        self.MSE["u"] = np.mean((mean_validation_set_u.ravel() - u_values.ravel())**2).item()
        self.MSE["f"] = np.mean((mean_validation_set_f.ravel() - f_values.ravel())**2).item()


    def use_GPy(self,X_star,save_path:str=None):
        """uses the GPy library to compute the GP"""
        if not self.timedependence:
            kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
            model_GPy = GPy.models.GPRegression(self.X, self.u_train, kernel)
            model_GPy.Gaussian_noise.variance.fix(self.noise[0])
            model_GPy.optimize_restarts(num_restarts = 20, verbose=False)

            fig, ax = plt.subplots(1,2,figsize=(12, 6))
            model_GPy.plot(ax  = ax[0])
            ax[0].scatter(self.validation_set[0],self.validation_set[1], label = "validation set", color = "orange", marker = "x", s = 15)
            ax[0].set_xlim(0,max(X_star))
            ax[0].set_title("u(t) prediction")
            ax[0].grid(alpha = 0.7)
            ax[0].legend()

            kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
            model_GPy2 = GPy.models.GPRegression(self.Y, self.f_train, kernel2)
            model_GPy2.Gaussian_noise.variance.fix(self.noise[1])
            model_GPy2.optimize_restarts(num_restarts = 20, verbose=False)

            model_GPy2.plot(ax = ax[1])
            ax[1].scatter(self.validation_set[2],self.validation_set[3], label = "validation set", color = "orange", marker = "x", s = 15)
            ax[1].set_xlim(0,max(X_star))
            ax[1].set_title("f(t) prediction")
            ax[1].grid(alpha = 0.7)
            ax[1].legend()
            plt.suptitle("GPy predictions")
            if save_path != None:
                plt.savefig(save_path)

        else:
            kernel_1 = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1., ARD = True)
            model_GPy = GPy.models.GPRegression(self.X, self.u_train, kernel_1)
            model_GPy.Gaussian_noise.variance.fix(self.noise[0])
            model_GPy.optimize_restarts(num_restarts = 20, verbose=False)

            kernel_2 = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1., ARD = True)
            model_GPy_2 = GPy.models.GPRegression(self.Y, self.f_train, kernel_2)
            model_GPy_2.Gaussian_noise.variance.fix(self.noise[1])
            model_GPy_2.optimize_restarts(num_restarts = 20, verbose=False)
            mean, var = model_GPy.predict(X_star)
            original_shape = (int(np.sqrt(len(mean))),int(np.sqrt(len(mean))))
            mean = mean.reshape(original_shape)
            mean2, var2 = model_GPy_2.predict(X_star)
            mean2 = mean2.reshape(original_shape)

            fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"},figsize=(12,5))
            ax[0].plot_surface(X_star[:,0].reshape(mean.shape), X_star[:,1].reshape(mean.shape), mean, cmap='viridis', edgecolor='none', alpha=0.5)
            ax[0].set_title('Predictive mean for u(t,x))')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('t')
            ax[0].scatter(self.X[:,0], self.X[:,1], self.u_train, c='r', marker='o')

            ax[1].plot_surface(X_star[:,0].reshape(mean2.shape), X_star[:,1].reshape(mean2.shape), mean2, cmap="viridis", edgecolor='none', alpha=0.5)
            ax[1].scatter(self.Y[:,0], self.Y[:,1], self.f_train, c='b', marker='o')
            ax[1].set_title('Predictive mean for f(t,x)')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('t')
            plt.legend()
            self.GPy_models = [model_GPy,model_GPy_2]
            plt.suptitle("GPy predictions")
            if save_path != None:
                plt.savefig(save_path)
    
    def plot_prediction(self,X_star, title:str, save_path:str):
        """plots the prediction of the GP at the points X_star"""
        assert self.mean_u is not None, "Please predict the mean and variance first"
        assert self.var_u is not None, "Please predict the mean and variance first"
        assert self.mean_f is not None, "Please predict the mean and variance first"
        assert self.var_f is not None, "Please predict the mean and variance first"
        
        if self.timedependence:
            assert X_star.shape[1] == 2, "Please provide a valid test data set"
            x_star, t_star = X_star[:,0].reshape(-1,1), X_star[:,1].reshape(-1,1)
            original_size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
            #x_star, t_star = np.meshgrid(x_star,t_star)
        else:
            assert X_star.shape[1] == 1, "Please provide a valid test data set"
            x_star = X_star
        
        if self.timedependence:
            self.mean_u = self.mean_u.reshape(original_size)
            self.mean_f = self.mean_f.reshape(original_size)
            fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"},figsize=(12,5))

            ax[0].plot_surface(x_star.reshape(self.mean_u.shape), t_star.reshape(self.mean_u.shape), self.mean_u, cmap='viridis', edgecolor='none', alpha=0.5)
            ax[0].set_title('Predictive mean for u(t,x)')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('t')
            ax[0].scatter(self.X[:,0], self.X[:,1], self.u_train, c='r', marker='o')

            ax[1].plot_surface(x_star.reshape(self.mean_f.shape), t_star.reshape(self.mean_f.shape), self.mean_f, cmap="viridis", edgecolor='none', alpha=0.5)
            ax[1].scatter(self.Y[:,0], self.Y[:,1], self.f_train, c='b', marker='o')
            ax[1].set_title('Predictive mean for f(t,x)')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('t')
            plt.legend()
            plt.suptitle(title)
            if save_path != None:
                plt.savefig(save_path,bbox_inches='tight')
        else:
            fig,ax = plt.subplots(1,2,figsize=(12, 6))
            ax[0].scatter(self.validation_set[0], self.validation_set[1], c='orange', marker='x', label='Validation set', alpha=0.5, s=15)
            ax[0].plot(self.X, self.u_train, 'r.', markersize=10, label='Observations')
            ax[0].plot(x_star, self.mean_u, 'b', label='Prediction')
            ax[0].fill_between(x_star.flatten(), self.mean_u.flatten() - 2 * np.sqrt(self.var_u.flatten()),
                            self.mean_u.flatten() + 2 * np.sqrt(self.var_u.flatten()), alpha=0.2, color='blue')
            ax[0].legend(loc='upper right', fontsize=10)
            ax[0].set_xlabel('$t$')
            ax[0].set_ylabel('$u(t)$')
            ax[0].set_title("u(t) prediction")
            ax[0].grid(alpha = 0.7)

            ax[1].scatter(self.validation_set[2], self.validation_set[3], c='orange', marker='x', label='Validation set', alpha=0.5, s=15)
            ax[1].plot(self.Y, self.f_train, 'r.', markersize=10, label='Observations')
            ax[1].plot(x_star, self.mean_f, 'b', label='Prediction')
            ax[1].fill_between(x_star.flatten(), self.mean_f.flatten() - 2 * np.sqrt(self.var_f.flatten()),
                                self.mean_f.flatten() + 2 * np.sqrt(self.var_f.flatten()), alpha=0.2, color='blue')
            ax[1].legend(loc='upper right', fontsize=10)
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$f(t)$')
            ax[1].set_title("f(t) prediction")
            ax[1].grid(alpha = 0.7)
            fig.suptitle(title)
            if save_path != None:
                plt.savefig(save_path,bbox_inches='tight')
        pass 
    
    def plot_difference(self, title:str, save_path:str):
        """plots the difference between the analytical solution and the predicted mean"""
        assert self.timedependence  == True, "Contourplot of difference only for 2d case"
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(-1,1), self.raw_data[1].reshape(-1,1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
        mean_u, var = self.predict_u(np.hstack((x_star,t_star)))
        mean_f, var = self.predict_f(np.hstack((x_star,t_star)))
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        cont = ax[0].contourf(x_star.reshape(mean_u.shape), t_star.reshape(mean_u.shape), np.abs(mean_u - u_grid.reshape(size)), cmap='viridis', alpha=0.8)
        ax[0].set_title(' |$f_*$ - u(t,x)|')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('t')
        ax[0].scatter(self.X[:,0], self.X[:,1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[0])

        cont2 = ax[1].contourf(x_star.reshape(mean_f.shape), t_star.reshape(mean_f.shape), np.abs(mean_f - f_grid.reshape(size)), cmap="viridis", alpha=0.8)
        ax[1].scatter(self.Y[:,0], self.Y[:,1], c='b', marker='o')
        ax[1].set_title(' |$f_*$ - f(t,x)|')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('t')
        fig.colorbar(cont2, ax=ax[1])
        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path,bbox_inches='tight')
        
    
    def plot_difference_GPy(self,title,save_path):
        assert self.timedependence  == True, "Contourplot of difference only for 2d case"
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(-1,1), self.raw_data[1].reshape(-1,1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
        mean_u, var = self.GPy_models[0].predict(np.hstack((x_star,t_star)))
        mean_f, var = self.GPy_models[1].predict(np.hstack((x_star,t_star)))
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        cont = ax[0].contourf(x_star.reshape(mean_u.shape), t_star.reshape(mean_u.shape), np.abs(mean_u - u_grid.reshape(size)), cmap='viridis', alpha=0.8)
        ax[0].set_title(' |$f_*$ - u(t,x)|')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('t')
        ax[0].scatter(self.X[:,0], self.X[:,1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[0])

        cont2 = ax[1].contourf(x_star.reshape(mean_f.shape), t_star.reshape(mean_f.shape), np.abs(mean_f - f_grid.reshape(size)), cmap="viridis", alpha=0.8)
        ax[1].scatter(self.Y[:,0], self.Y[:,1], c='b', marker='o')
        ax[1].set_title(' |$f_*$ - f(t,x)|')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('t')
        fig.colorbar(cont2, ax=ax[1])
        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path,bbox_inches='tight')
        MSE_u = np.mean((mean_u - u_grid.reshape(size))**2).item()
        MSE_f = np.mean((mean_f - f_grid.reshape(size))**2).item()
        print("MSE_u: ", MSE_u)
        print("MSE_f: ", MSE_f)

    def plot_variance(self,X_star, title:str, save_path:str):
        """plots the variance of the GP at the points X_star"""
        assert self.timedependence == True, "Contourplot of variance only for 2d case"
        assert self.mean_u is not None, "Please predict the mean and variance first"
        assert self.var_u is not None, "Please predict the mean and variance first"
        assert self.mean_f is not None, "Please predict the mean and variance first"
        assert self.var_f is not None, "Please predict the mean and variance first"

        x_star, t_star = X_star[:,0].reshape(-1,1), X_star[:,1].reshape(-1,1) 
        original_size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
        self.var_u = self.var_u.reshape(original_size)
        self.var_f = self.var_f.reshape(original_size)
        fig, ax = plt.subplots(1,2,figsize=(12,5))

        cont_0 = ax[0].contourf(x_star.reshape(self.var_u.shape), t_star.reshape(self.var_u.shape), np.sqrt(self.var_u), cmap='viridis', alpha=0.8)
        ax[0].set_title('Predictive std for u(t,x))')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('t')
        ax[0].scatter(self.X[:,0], self.X[:,1], marker='o')
        fig.colorbar(cont_0, ax=ax[0])

        cont_1 = ax[1].contourf(x_star.reshape(self.var_f.shape), t_star.reshape(self.var_f.shape), np.sqrt(self.var_f), cmap="viridis", alpha=0.8)
        ax[1].scatter(self.Y[:,0], self.Y[:,1], marker='o')
        ax[1].set_title('Predictive std for f(t,x)')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('t')
        fig.colorbar(cont_1, ax=ax[1])

        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path,bbox_inches='tight')

    def plot_variance_GPy(self,title,save_path=None):
        assert self.timedependence  == True, "Contourplot of difference only for 2d case"
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(-1,1), self.raw_data[1].reshape(-1,1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
        mean_u, var_u = self.GPy_models[0].predict(np.hstack((x_star,t_star)))
        mean_f, var_f = self.GPy_models[1].predict(np.hstack((x_star,t_star)))
        var_u, var_f = var_u.reshape(u_grid.shape), var_f.reshape(u_grid.shape)
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        cont = ax[0].contourf(x_star.reshape(u_grid.shape), t_star.reshape(u_grid.shape), var_u, cmap='viridis', alpha=0.8)
        ax[0].set_title('GPy std for u(t,x)')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('t')
        ax[0].scatter(self.X[:,0], self.X[:,1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[0])

        cont2 = ax[1].contourf(x_star.reshape(u_grid.shape), t_star.reshape(u_grid.shape), var_f, cmap="viridis", alpha=0.8)
        ax[1].scatter(self.Y[:,0], self.Y[:,1], c='b', marker='o')
        ax[1].set_title(' GPy std for f(t,x)')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('t')
        fig.colorbar(cont2, ax=ax[1])
        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path,bbox_inches='tight')
    def grad_log_marginal_likelihood(self):
        return grad(self.log_marginal_likelihood_to_optimize())
    
    def plot_raw_data(self,Training_points=False):
        if self.timedependence:
            x_star, t_star = self.raw_data[0].reshape(-1,1), self.raw_data[1].reshape(-1,1)
            u_grid = self.raw_data[2]
            f_grid = self.raw_data[3]
            size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
            fig, ax = plt.subplots(1,2,figsize=(12,5),subplot_kw={"projection": "3d"})
            cont = ax[0].plot_surface(x_star.reshape(size), t_star.reshape(size), u_grid.reshape(size), cmap='viridis', alpha=0.8, edgecolor='none')
            ax[0].set_title('u(t,x)')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('t')
           

            cont2 = ax[1].plot_surface(x_star.reshape(size), t_star.reshape(size), f_grid.reshape(size), cmap="viridis", alpha=0.8)
            ax[1].set_title('f(t,x)')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('t')
            
            if Training_points:
                ax[0].scatter(self.X[:,0], self.X[:,1], self.u_train, c='r', marker='o', label='Training points')
                ax[1].scatter(self.Y[:,0], self.Y[:,1], self.f_train, c='b', marker='o', label='Training points')
            plt.legend()
        else:
            x_star = self.raw_data[0]
            u_grid = self.raw_data[1]
            f_grid = self.raw_data[2]
            fig,ax = plt.subplots(1,2,figsize=(12, 6))
            ax[0].plot(x_star, u_grid, markersize=10, label='Observations')
            ax[0].set_xlabel('$t$')
            ax[0].set_ylabel('$u(t)$')
            ax[0].set_title("u(t) raw data")
            ax[0].grid(alpha = 0.7)

            ax[1].plot(x_star, f_grid, markersize=10, label='Observations')
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$f(t)$')
            ax[1].set_title("f(t) raw data")
            ax[1].grid(alpha = 0.7)
            if Training_points:
                ax[0].scatter(self.X, self.u_train, c='r', marker='o', label='Training points')
                ax[1].scatter(self.Y, self.f_train, c='b', marker='o', label='Training points')
            plt.legend()
    def plot_validation_set(self):
        assert self.validation_set is not None, "Please set the validation set first"
        if self.timedependence:
            x_star, t_star = self.validation_set[0].reshape(-1,1), self.validation_set[1].reshape(-1,1)
            u_values = self.validation_set[4]
            f_values = self.validation_set[5]
            size = (int(np.sqrt(len(x_star))),int(np.sqrt(len(x_star))))
            fig, ax = plt.subplots(1,2,figsize=(12,5),subplot_kw={"projection": "3d"})
            cont = ax[0].plot_surface(x_star.reshape(size), t_star.reshape(size), u_values.reshape(size), cmap='viridis', alpha=0.8, edgecolor='none')
            ax[0].set_title('u(t,x)')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('t')
            fig.colorbar(cont, ax=ax[0])

            x_star, t_star = self.validation_set[2].reshape(-1,1), self.validation_set[3].reshape(-1,1)
            cont2 = ax[1].plot_surface(x_star.reshape(size), t_star.reshape(size), f_values.reshape(size), cmap="viridis", alpha=0.8)
            ax[1].set_title('f(t,x)')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('t')
            fig.colorbar(cont2, ax=ax[1])
        else:
            x_star = self.validation_set[0]
            u_values = self.validation_set[1]
            f_values = self.validation_set[2]
            fig,ax = plt.subplots(1,2,figsize=(12, 6))
            ax[0].plot(x_star, u_values, markersize=10, label='Observations')
            ax[0].set_xlabel('$t$')
            ax[0].set_ylabel('$u(t)$')
            ax[0].set_title("u(t) validation set")
            ax[0].grid(alpha = 0.7)

            ax[1].plot(x_star, f_values, markersize=10, label='Observations')
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$f(t)$')
            ax[1].set_title("f(t) validation set")
            ax[1].grid(alpha = 0.7)
    @staticmethod
    def get_data_set_1d(filename, n_training_points, noise:list,seeds = [50,41] ):
        try:
            df= pd.read_csv("PI_Kernels_class/"+filename)
        except:
            df= pd.read_csv(filename)
        
        t = df['t'].values
        u = df['u'].values
        f = df['f'].values
        

        d = 1 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d,seed=seeds[0],scramble=True)
        sample = engine.random(n_training_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_train_u = t[indices]
        u_train_u = u[indices] + np.random.normal(0,noise[0],u[indices].shape)

        # training data for f(t)
        engine = qmc.Sobol(d,seed=seeds[1])
        sample = engine.random(n_training_points)
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_train_f = t[indices]
        f_train_f = f[indices] + np.random.normal(0,noise[1],f[indices].shape)

        return t_train_u.reshape(-1,1), u_train_u.reshape(-1,1), t_train_f.reshape(-1,1), f_train_f.reshape(-1,1), [t,u,f]
    @staticmethod
    def get_validation_set_1d(filename, n_validation_points, noise:list ):
        try:
            df= pd.read_csv("PI_Kernels_class/"+filename)
        except:
            df= pd.read_csv(filename)
        
        t = df['t'].values
        u = df['u'].values
        f = df['f'].values
        


        d = 1 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d,seed=50)
        sample = engine.random(n_validation_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_val_u = t[indices]
        u_val_u = u[indices] + np.random.normal(0,noise[0],u[indices].shape)

        # training data for f(t)
        engine = qmc.Sobol(d,seed=7)
        sample = engine.random(n_validation_points)
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_val_f = t[indices]
        f_val_f = f[indices] + np.random.normal(0,noise[1],f[indices].shape)

        return [t_val_u, u_val_u, t_val_f, f_val_f]
    @staticmethod
    def get_data_set_2d(filename, n_training_points, noise:list ):
    #load data from mathematica calculation
        try:
            df= pd.read_csv("PI_Kernels_class/"+filename)
        except:
            df= pd.read_csv(filename)
        x = df['space'].values
        t = df['time'].values
        u = df['u'].values
        f = df['f'].values

        x_mesh = x.reshape(int(np.sqrt(len(x))),int(np.sqrt(len(x))))
        t_mesh = t.reshape(x_mesh.shape)
        u_grid = u.reshape(x_mesh.shape)
        f_grid = f.reshape(x_mesh.shape)


        
        # 
        x_axis = np.unique(x_mesh)
        t_axis = np.unique(t_mesh)
        
        
        d = 2 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d,seed=77)
        sample = engine.random(n_training_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array([len(x_axis),len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_train_u = x_axis[indices[:,0]]
        t_train_u = t_axis[indices[:,1]]
        u_train = u_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[0], u_grid[indices[:,1],indices[:,0]].shape)
        
        #same thing for f(t,x)
        engine = qmc.Sobol(d,seed=10)
        sample = engine.random(n_training_points)
        indices = sample * np.array([len(x_axis),len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_train_f = x_axis[indices[:,0]]
        t_train_f = t_axis[indices[:,1]]
        f_train = f_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[1], f_grid[indices[:,1],indices[:,0]].shape)
    
    
        return x_train_u.reshape(-1,1), x_train_f.reshape(-1,1), t_train_u.reshape(-1,1), t_train_f.reshape(-1,1), u_train.reshape(-1,1), f_train.reshape(-1,1), [x_mesh,t_mesh,u_grid,f_grid]
    @staticmethod
    def get_validation_set_2d(filename, n_validation_points, noise:list ):
        try:
            df= pd.read_csv("PI_Kernels_class/"+filename)
        except:
            df= pd.read_csv(filename)
        x = df['space'].values
        t = df['time'].values
        u = df['u'].values
        f = df['f'].values

        x_mesh = x.reshape(int(np.sqrt(len(x))),int(np.sqrt(len(x))))
        t_mesh = t.reshape(x_mesh.shape)
        u_grid = u.reshape(x_mesh.shape)
        f_grid = f.reshape(x_mesh.shape)

        x_axis = np.unique(x_mesh)
        t_axis = np.unique(t_mesh)

        d = 2 # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d)
        sample = engine.random(n_validation_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array([len(x_axis),len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_val_u = x_axis[indices[:,0]]
        t_val_u = t_axis[indices[:,1]]
        u_val = u_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[0], u_grid[indices[:,1],indices[:,0]].shape)
        
        #same thing for f(t,x)
        engine = qmc.Sobol(d)
        sample = engine.random(n_validation_points)
        indices = sample * np.array([len(x_axis),len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_val_f = x_axis[indices[:,0]]
        t_val_f = t_axis[indices[:,1]]
        f_val = f_grid[indices[:,1],indices[:,0]] + np.random.normal(0, noise[1], f_grid[indices[:,1],indices[:,0]].shape)

        return [x_val_u, t_val_u,x_val_f, t_val_f, u_val, f_val]
       