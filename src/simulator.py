"""
simulator.py -- Moreau time-stepping with always-in-contact smoothing.
"""
import numpy as np
from smoothing import get_smoothing

class Params:
    def __init__(self):
        self.m=1.0; self.g=9.81; self.dt=1e-3; self.mu=0.5
        self.e=0.0; self.T=1.0; self.z0=1.0
        self.xy_target=np.array([1.0,0.5]); self.z_target=0.0
    @property
    def N(self): return int(self.T/self.dt)
    @property
    def t_c(self): return np.sqrt(2*self.z0/self.g)

def prox_normal(p_N):
    return max(p_N, 0.0)

def prox_friction_disk(p_T, p_N, mu):
    r = mu*max(p_N, 0.0)
    nrm = np.linalg.norm(p_T)
    if nrm <= r: return p_T.copy()
    if nrm == 0.0: return np.zeros(2)
    return p_T*(r/nrm)

def prox_contact(p, mu):
    p_N_proj = prox_normal(p[2])
    p_T_proj = prox_friction_disk(p[:2], p_N_proj, mu)
    return np.array([p_T_proj[0], p_T_proj[1], p_N_proj])

def solve_contact_GS(G, c, mu, n_iter=20):
    try: p = -np.linalg.solve(G, c)
    except: p = -np.linalg.pinv(G)@c
    r = 1.0/(1.0+np.sum(np.abs(G)))
    for _ in range(n_iter):
        p = prox_contact(p - r*(G@p + c), mu)
    return p

def moreau_step_smooth(q_S, v_S, sigma_fn, kappa, params):
    m=params.m; g=params.g; dt=params.dt; mu=params.mu
    q_M = q_S + 0.5*dt*v_S
    d = -q_M[2]
    sigma_val = float(sigma_fn(d, kappa))
    v_free = v_S + dt*np.array([0.,0.,-g])
    G = (1./m)*np.eye(3)
    c = v_free.copy()
    p_raw = solve_contact_GS(G, c, mu)
    p_smooth = p_raw * sigma_val
    v_E = v_free + (1./m)*p_smooth
    q_E = q_M + 0.5*dt*v_E
    return q_E, v_E, p_raw, sigma_val, d

def simulate(theta, sigma_fn, kappa, params):
    q=np.array([0.,0.,params.z0]); v=np.array([theta[0],theta[1],0.])
    N=params.N
    traj_q=np.zeros((N+1,3)); traj_v=np.zeros((N+1,3))
    traj_p=np.zeros((N,3));   traj_s=np.zeros(N); traj_d=np.zeros(N)
    traj_q[0]=q; traj_v[0]=v
    for k in range(N):
        q,v,p_raw,s,d = moreau_step_smooth(q,v,sigma_fn,kappa,params)
        traj_q[k+1]=q; traj_v[k+1]=v; traj_p[k]=p_raw
        traj_s[k]=s;   traj_d[k]=d
    return traj_q, traj_v, traj_p, traj_s, traj_d

def loss(theta, sigma_fn, kappa, params):
    traj_q,_,_,_,_ = simulate(theta, sigma_fn, kappa, params)
    q_T = traj_q[-1]
    diff_xy = q_T[:2] - params.xy_target
    return float(diff_xy@diff_xy + (q_T[2]-params.z_target)**2)

def grad_fd(theta, sigma_fn, kappa, params, h=1e-5):
    g=np.zeros(2)
    for i in range(2):
        tp=theta.copy(); tp[i]+=h
        tm=theta.copy(); tm[i]-=h
        g[i]=(loss(tp,sigma_fn,kappa,params)-loss(tm,sigma_fn,kappa,params))/(2*h)
    return g

def grad_fog(theta, sigma_fn, sigma_prime_fn, kappa, params):
    m=params.m; g=params.g; dt=params.dt; mu=params.mu; N=params.N

    # Forward pass
    q=np.array([0.,0.,params.z0]); v=np.array([theta[0],theta[1],0.])
    states_q=[q.copy()]; states_v=[v.copy()]
    store_p=[]; store_s=[]; store_sp=[]; store_d=[]

    for k in range(N):
        q_M = q+0.5*dt*v; d=-q_M[2]
        s=float(sigma_fn(d,kappa)); sp=float(sigma_prime_fn(d,kappa))
        v_free=v+dt*np.array([0.,0.,-g])
        G=(1./m)*np.eye(3); c=v_free.copy()
        p_raw=solve_contact_GS(G,c,mu)
        v=v_free+(1./m)*p_raw*s
        q=q_M+0.5*dt*v
        states_q.append(q.copy()); states_v.append(v.copy())
        store_p.append(p_raw.copy()); store_s.append(s)
        store_sp.append(sp); store_d.append(d)

    # Loss gradient w.r.t. final state
    q_T=states_q[-1]
    dL_dqT=np.zeros(3)
    dL_dqT[:2]=2.*(q_T[:2]-params.xy_target)
    dL_dqT[2]=2.*(q_T[2]-params.z_target)

    # Propagate Jacobians J_q=dq_k/dtheta (3x2), J_v=dv_k/dtheta (3x2)
    J_q=np.zeros((3,2)); J_v=np.zeros((3,2))
    J_v[0,0]=1.; J_v[1,1]=1.

    for k in range(N):
        p_raw=store_p[k]; s=store_s[k]; sp=store_sp[k]
        p_N=p_raw[2]; p_T=p_raw[:2]; nrm_T=np.linalg.norm(p_T)
        mu_pN=mu*p_N

        # Jacobian of prox_N
        J_pN_xN = 1.0 if p_N > 0 else 0.0

        # Jacobian of prox_T
        if nrm_T <= mu_pN:
            J_pT_xT=np.eye(2); J_pT_xN=np.zeros(2)
        elif nrm_T > 0:
            e_T=p_T/nrm_T
            J_pT_xT=(mu_pN/nrm_T)*(np.eye(2)-np.outer(e_T,e_T))
            J_pT_xN=(mu/nrm_T)*p_T*J_pN_xN
        else:
            J_pT_xT=np.zeros((2,2)); J_pT_xN=np.zeros(2)

        # dp_raw/dv_free (3x3): x = -m*v_free
        J_praw_vfree=np.zeros((3,3))
        J_praw_vfree[2,2]=J_pN_xN*(-m)
        J_praw_vfree[:2,:2]=J_pT_xT*(-m)
        J_praw_vfree[:2,2]=J_pT_xN*(-m)

        # dd/d(q_S, v_S): d=-q_z^M=-q_z^S-dt/2*v_z^S
        dd_dqS=np.zeros(3); dd_dqS[2]=-1.
        dd_dvS=np.zeros(3); dd_dvS[2]=-dt/2.

        # dp_smooth/d(q_S,v_S) = sigma*dp_raw/dv_free@dv_free/d(q_S,v_S)
        #                       + sigma'*outer(p_raw, dd/d(q_S,v_S))
        # dv_free/dq_S=0, dv_free/dv_S=I
        J_psmooth_qS = sp*np.outer(p_raw, dd_dqS)
        J_psmooth_vS = s*J_praw_vfree + sp*np.outer(p_raw, dd_dvS)

        # dv_E/d(q_S,v_S)
        J_vE_qS=(1./m)*J_psmooth_qS
        J_vE_vS=np.eye(3)+(1./m)*J_psmooth_vS

        # dq_E/d(q_S,v_S): q_E=q_M+dt/2*v_E, q_M=q_S+dt/2*v_S
        J_qE_qS=np.eye(3)+0.5*dt*J_vE_qS
        J_qE_vS=0.5*dt*np.eye(3)+0.5*dt*J_vE_vS

        # Chain rule
        J_q_new=J_qE_qS@J_q+J_qE_vS@J_v
        J_v_new=J_vE_qS@J_q+J_vE_vS@J_v
        J_q=J_q_new; J_v=J_v_new

    return dL_dqT@J_q

def grad_zog(theta, sigma_fn, kappa, params, sigma_noise=0.1,
             N_samples=100, rng=None):
    if rng is None: rng=np.random.default_rng(0)
    W=rng.normal(0.,sigma_noise,size=(N_samples,2))
    grad=np.zeros(2)
    for i in range(N_samples):
        w=W[i]; L_i=loss(theta+w, sigma_fn, kappa, params)
        grad+=(w/sigma_noise**2)*L_i
    return grad/N_samples

def analytical_optimum(params):
    t_c=params.t_c; mu=params.mu; g=params.g
    q_st=params.xy_target; d_st=np.linalg.norm(q_st)
    direction=q_st/d_st
    theta_sticking=d_st/t_c
    if theta_sticking <= mu*g*t_c:
        return direction*theta_sticking
    d_star2=d_st-0.5*mu*g*t_c**2
    a=1./(2.*mu*g); b=t_c; c_coef=-d_star2
    disc=b**2-4.*a*c_coef
    if disc < 0: return direction*theta_sticking
    u=(-b+np.sqrt(disc))/(2.*a)
    return direction*(u+mu*g*t_c)
