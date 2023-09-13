import GPy

def use_GPy(x_train,y_train,noise):
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(x_train, y_train, kernel)
    m.Gaussian_noise.variance = noise[0]
    m.Gaussian_noise.variance.fix()
    m.optimize(messages=False)
    return m
