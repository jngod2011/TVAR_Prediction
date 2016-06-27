#################### estimation.py ########################################
# contient les fonctions utiles pour l'estimation d'un TVAR
# generation_est : genere l'estimateur NLSM 
# aggregation : retourne le prédicteur par agregation a partir d'un certain nombre de predicteurs


from scipy.linalg import norm
from scipy.fftpack import fft
import numpy as np




def generation_est(X, d, T, mu):
    # Info : X est de longueur 2*T pour avoir les temps negatifs
    theta_est = np.zeros((d,T),dtype='complex')
    X_est = np.zeros(T,dtype='complex')  
    for k in (np.arange(T)): # on fait T itérations
        XX = X[T+k-d:T+k][::-1]
        if k==0:
            theta_est[:,k] = (mu * X[T+k] * XX / (1 + mu * norm(XX) ** 2))
            theta_est[:,k] = np.zeros(d,dtype='complex')
        else:
            theta_est[:,k] = (theta_est[:,k-1].T + mu * (X[T+k] - np.dot(XX, theta_est[:,k-1])) * XX / (1 + mu * norm(XX) ** 2)) 
        pred = np.dot(XX, theta_est[:,k])
        X_est[k] = pred
    return X_est, theta_est

def generation_est2(X, d, T, mu):

    theta_est = np.zeros((d,T),dtype='complex')
    X_est = np.zeros(T,dtype='complex')
        
    for k in (np.arange(d,T)):
        XX = X[k-d:k][::-1]
        
        theta_est[:,k] = np.matrix(theta_est[:,k-1] + mu * (X[k] - np.dot(XX, theta_est[:,k-1])) * XX / (1 + mu * norm(XX) ** 2)) 

        X_est[k] = np.dot(XX, theta_est[:,k])
        
    return X_est, theta_est


def aggregation(X, predictions, estimations, d, T, eta, strategy = 1):    
    """Retourne un predicteur par aggregation"""   
    N = len(predictions[:,0])
    pred = np.zeros(T,dtype='complex')
    estim = np.zeros((d,T),dtype='complex')
    alpha = 1./N * np.ones(N)   
    
    for t in np.arange(T): 
        if t > 0:
            if strategy == 1:
                v = alpha * np.exp(-2*eta*(pred[t-1]-X[t-1])*predictions[:,t-1])
            else:
                v = alpha * np.exp(-eta*(predictions[:,t-1]-X[t-1])**2)
        
            alpha = v / np.sum(v)           
        pred[t] = np.dot(alpha, predictions[:,t])  
        estim[:,t] = np.dot(estimations[:,:,t].T,alpha) 
        
    return pred, estim, alpha


def dsp(theta, T, N = 512):
    dsp_array = np.matrix(np.zeros((N, T)))    
    for t in np.arange(T):
        dsp_array[:,t] = np.matrix(1./(2*np.pi*abs(fft(theta[:,t].T, N)) ** 2)).T       
    lambd = np.arange(N, dtype = 'double')/N        
    return lambd, dsp_array

def do_aggregation(X,d,T,eta,mu,strategy=1):
    N = len(mu)
    X_pred = np.zeros((N,T),dtype='complex')
    theta_estim = np.zeros((N,d,T),dtype='complex')


    for k in np.arange(N):
        X_pred[k,:],theta_estim[k,:,:] = generation_est2(X, d, T, mu[k])

    pred_agr, estim_agr, alpha = aggregation(X, X_pred, theta_estim, d,T, 0.1, 1)
        
                  
    return pred_agr, estim_agr, alpha
                           
    