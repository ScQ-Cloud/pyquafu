import numpy as np

def adam(func, x, gradsf, args=(), call_back=None, eta=0.01,beta1=0.9, beta2=0.999, epsilon=1e-8, maxiter=1000, tol=1e-8, verbose=False):
    n_para = len(x)
    mt = np.zeros(n_para)
    vt = np.zeros(n_para)

    traj = []
    if verbose:
        print((" "*5).join(["step".ljust(10), "loss".ljust(10), "grad_norm".ljust(10)]))
    
    grads_norm = 0.
    for j in range(maxiter):
        traj.append(func(x, *args))
        if verbose:
            print((" "*5).join([("%d" %j).ljust(10), ("%.5f" %traj[j]).ljust(10), ("%.5f" %grads_norm).ljust(10)]))
        if j > 0 and abs(traj[j]-traj[j-1]) <= tol:
            return x, traj[-1], traj
        
        grads = np.array(gradsf(x, *args))
        grads_norm = np.linalg.norm(grads)
        mt = beta1 * mt + (1.-beta1) * grads
        vt = beta1 * vt + (1.-beta2) * grads**2
        mtt = mt / (1-beta1**(j+2))
        vtt = vt / (1-beta2**(j+2))
        x = x - eta * mtt / (np.sqrt(vtt) + epsilon)
        if call_back:
            call_back(x, *args)
    traj.append(func(x, *args))
    return x, traj[-1], traj


def spsa_grad(func, x, k, args=(), spsa_iter=10, c=0.1, gamma=0.101):
    
    dim = len(x)
    ck = c/(k)**gamma
    gx = 0.0
    for i in range(spsa_iter):
        Delta = 2*np.round(np.random.rand(dim))-1
        x1 = x + ck*Delta
        x2 = x - ck*Delta
        y1 = func(x1, *args) 
        y2 = func(x2, *args)
        gx += (y1 - y2) / (2*ck*Delta)
    gx = gx / spsa_iter
    return gx

def spsa(func, x, args=(), call_back=None, spsa_iter=10, max_iter=1000, a=0.1, c=0.1, A=100, alpha=0.602, gamma=0.101, tol=1e-8, verbose=False):
    '''SPSA minimize
       c: at a level of standard deviation of func
       A: <=10% of max_iter '''
    traj = [func(x, *args)]
    if verbose:
            print((" "*5).join(["step".ljust(10), "loss".ljust(10), "grad_norm".ljust(10)]))
    
    grads_norm = 0.
    for k in range(max_iter):
        if verbose:
            print((" "*5).join([("%d" %k).ljust(10), ("%.5f" %traj[k]).ljust(10), ("%.5f" %grads_norm).ljust(10)]))
        if k > 0 and abs(traj[k]-traj[k-1]) <= tol:
            return x, traj[-1], traj
        
        ak = a/(k+1+A)**alpha
        grads = spsa_grad(func, x, k+1, args, spsa_iter, c=c, gamma=gamma)
        grads_norm = np.linalg.norm(grads)
        x = x - ak * grads
        if call_back:
            call_back(x, *args)
        traj.append(func(x, *args))

    return x, traj[-1], traj

