import numpy as np
from scipy.special import erf as _scipy_erf

def sigmoid(d, kappa):
    x = np.asarray(d, dtype=float) * kappa
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def sigmoid_prime(d, kappa):
    s = sigmoid(d, kappa)
    return kappa * s * (1.0 - s)

def erf_smooth(d, kappa):
    return 0.5*(1.0 + _scipy_erf(np.asarray(d,dtype=float)*kappa/np.sqrt(2.0)))

def erf_smooth_prime(d, kappa):
    return (kappa/np.sqrt(2.0*np.pi))*np.exp(-0.5*(np.asarray(d,dtype=float)*kappa)**2)

def smoothstep(d, kappa):
    s = np.clip(0.5 + kappa*np.asarray(d,dtype=float), 0.0, 1.0)
    return 3.0*s**2 - 2.0*s**3

def smoothstep_prime(d, kappa):
    d = np.asarray(d, dtype=float)
    s_raw = 0.5 + kappa*d
    in_support = (s_raw > 0.0) & (s_raw < 1.0)
    s = np.clip(s_raw, 0.0, 1.0)
    return np.where(in_support, 6.0*kappa*s*(1.0-s), 0.0)

def sigmoid_mass(d, kappa, mass):
    return sigmoid(d, kappa*mass)

def sigmoid_mass_prime(d, kappa, mass):
    return sigmoid_prime(d, kappa*mass)

def hard(d, kappa):
    return (np.asarray(d,dtype=float) >= 0).astype(float)

def hard_prime(d, kappa):
    return np.zeros_like(np.asarray(d,dtype=float))

def get_smoothing(name, mass=1.0):
    if name == 'sigmoid':
        return sigmoid, sigmoid_prime
    elif name == 'erf':
        return erf_smooth, erf_smooth_prime
    elif name == 'smoothstep':
        return smoothstep, smoothstep_prime
    elif name == 'sigmoid_mass':
        return (lambda d,k: sigmoid_mass(d,k,mass),
                lambda d,k: sigmoid_mass_prime(d,k,mass))
    elif name == 'hard':
        return hard, hard_prime
    else:
        raise ValueError(f"Unknown smoothing: {name}")

SMOOTHING_NAMES  = ['sigmoid','erf','smoothstep','sigmoid_mass','hard']
SMOOTHING_LABELS = {
    'sigmoid':      r'Sigmoid $\sigma_{\rm sig}$',
    'erf':          r'Erf (Gaussian CDF) $\sigma_{\rm erf}$',
    'smoothstep':   r'Smoothstep $\sigma_{\rm poly}$',
    'sigmoid_mass': r'Mass-scaled $\sigma_{\rm mass}$',
    'hard':         r'Hard contact',
}
SMOOTHING_COLORS = {
    'sigmoid':'#2196F3','erf':'#E91E63','smoothstep':'#4CAF50',
    'sigmoid_mass':'#FF9800','hard':'#9E9E9E',
}
SMOOTHING_LS = {
    'sigmoid':'-','erf':'--','smoothstep':'-.','sigmoid_mass':':',
    'hard':(0,(3,1,1,1)),
}
