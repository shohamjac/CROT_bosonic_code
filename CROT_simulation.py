import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib import cm
from qutip import *
from qutip.ipynbtools import plot_animation
from matplotlib import animation
from IPython.display import HTML
from wigner_animation import wigner_animate


def plot_expection(new, tlist, operator, operator2 = None, operator3 = None):
    fig, ax = plt.subplots(figsize=(12,6))
    expectation = [np.trace(new.states[i]*operator) for i in range(len(tlist))]
    ax.plot(tlist, np.real(expectation), 'r')
    if operator2 is not None:
        expectation2 = [np.trace(new.states[i]*operator2) for i in range(len(tlist))]
        ax.plot(tlist, np.real(expectation2), 'b')
    if operator3 is not None:
        expectation3 = [np.trace(new.states[i]*operator3) for i in range(len(tlist))]
        ax.plot(tlist, np.real(expectation3), 'g')
    ax.legend(("123"))
    ax.set_xlabel('Time')
    ax.set_ylabel('expectation value');
    
    
def CROT_simulation(a_start,b_start, #both are strings. can be 0,1,+,-
                    do_CROT = True,
                    N = 10, # number of levels in Hilbert space
                    x_e = 2*np.pi* -50e3,
                    x_f = 2*np.pi* -100e3,
                    x_ab = 0, #Cavity a-b Cross-Kerr
                    K = 0,
                    d = 2*np.pi* -200e6, #Transmon unharmonicity
                    g2 = 2*np.pi * 0.5e6, #drive strength phi_a*phi_q*xi
                    g3 = 2*np.pi * 0.8e6, #drive strength phi_b*phi_q*xi
                    omega_CROT = 2*np.pi * 1e6, #drive strength phi_a*phi_b*phi_q*xi (not squered)
                    delta = 2*np.pi * 50e6,
                    kappa = 0, #1/1e-3 # Cavity decay rate
                    alpha = 2.0
                   ):
    
    N_q = 3 #g,e,f not h
    
    a_rot = 1 #two_legged_cat
    b_rot = 1
    CROT_time = np.pi / (a_rot * b_rot) * (delta / omega_CROT**2)
    time_steps = 10000
    simulation_time = CROT_time
    delta2 = 2 * g2**2 / x_e
    delta3 = 2*g3**2 / x_e
    tlist = np.linspace(0, simulation_time, time_steps)
    
    args = {
        'g': omega_CROT,
        'g2': g2,
        'g3': g3,
        'Delta': delta,
        't_pulse': 1*CROT_time,
        'unhar': -d,
        'Delta2': delta2,
        'Delta3': delta3,
        't_pulse2': CROT_time/40
    }

    g_g = fock_dm(N_q,0) # |g><g|
    e_e = fock_dm(N_q,1) # |e><e|
    f_f = fock_dm(N_q,2) # |f><f|
    
    state_dict = {
        '0': coherent(N,alpha) + coherent(N,-alpha),
        '1': coherent(N,alpha) - coherent(N,-alpha),
        '+': coherent(N,alpha),
        '-': coherent(N,-alpha)
    }
    
    #initialize the system
    rho_cat_a = ket2dm(state_dict[a_start])
    rho_cat_b = ket2dm(state_dict[b_start])
    qubit = fock_dm(N_q,0)
    system = tensor(qubit, rho_cat_a, rho_cat_b).unit()
    
    # steady state Hamiltonian
    H_0 = (0 +
           + d * tensor(f_f, qeye(N), qeye(N)) #Transmon unharmonicity
           
           + x_e * tensor(e_e, num(N), qeye(N)) #Transmon-Cavity_a Cross-Kerr
           + x_e * tensor(e_e,  qeye(N),num(N)) #Transmon-Cavity_b Cross-Kerr
           
           + x_f * tensor(f_f, num(N), qeye(N)) #Transmon-Cavity_a Cross-Kerr
           + x_f * tensor(f_f,  qeye(N),num(N)) #Transmon-Cavity_b Cross-Kerr
           
           + x_ab * tensor(qeye(N_q), num(N), num(N)) #Cavity_a_b Cross-Kerr
           + K * tensor(qeye(N_q), (destroy(N).dag())**2 * (destroy(N))**2, qeye(N)) #Cavity_a self-Kerr
           + K * tensor(qeye(N_q), qeye(N), (destroy(N).dag())**2 * (destroy(N))**2) #Cavity_b self-Kerr
          )
    
    a_b_qd = tensor(destroy(N_q).dag(), destroy(N), destroy(N))
    a_qd = tensor(destroy(N_q).dag(), destroy(N), qeye(N))
    qd = tensor(destroy(N_q).dag(), qeye(N), qeye(N))
    b_qd = tensor(destroy(N_q).dag(), qeye(N), destroy(N))

    H1 = [a_b_qd, 'g*exp(1j*(Delta)*t) * (t<t_pulse)']
    H2 = [a_b_qd.dag(), 'g*exp(-1j*(Delta)*t) * (t<t_pulse)']

    H3 = [a_qd,       'g2*exp(1j*(Delta2+unhar)*t)']
    H4 = [a_qd.dag(), 'g2*exp(-1j*(Delta2+unhar)*t)']

    H5 = [b_qd,       'g3*exp(1j*(Delta3+unhar)*t)']
    H6 = [b_qd.dag(), 'g3*exp(-1j*(Delta3+unhar)*t)']
    
    H = [H_0,H3,H4,H5,H6]
    
    if(do_CROT):
        H.append(H1)
        H.append(H2)
    
    
    #add noise to the simulation
    c_ops = []  # Build collapse operators
    c_ops.append(np.sqrt(kappa) * tensor(qeye(N_q), destroy(N), qeye(N))) #cavity_a decay
    c_ops.append(np.sqrt(kappa) * tensor(qeye(N_q), qeye(N), destroy(N))) #cavity_b decay
    
    #expectation value, for now we don't use this
    e_ops = []# [tensor(num(N_q), qeye(N), qeye(N))]
    
    print("Gate time is: ", args['t_pulse'])
    print("g is:", args['g'])
    print("Delta is:", args['Delta'])
    print("Delta/g is (should be large):", args['Delta']/args['g'])
    print("simulation dt is:", simulation_time/time_steps)
    print("delta2 is:", delta2)
    print("g2 is:", g2)
    print("Delta2/g2 is (should be large):", args['Delta2']/args['g2'])
    print("delta3 is:", delta3)
    print("g3 is:", g3)
    print("Delta3/g3 is (should be large):", args['Delta3']/args['g3'])

    
    options = Options(rhs_reuse=False,store_final_state=True)
    new = mesolve(H, system, tlist, c_ops, e_ops, args = args, options=options, progress_bar=True)
    return new


def plot_cavity_process(state_object):
    rho_a_final = ptrace(state_object.final_state, 1)
    rho_a_start = ptrace(state_object.states[0], 1)
    rho_b_final = ptrace(state_object.final_state, 2)
    rho_b_start = ptrace(state_object.states[0], 2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    plot_wigner(rho_a_start, fig=fig,ax=axes[0][0])
    axes[0][0].set_title("cavity a start")
    plot_wigner(rho_a_final, fig=fig,ax=axes[0][1])
    axes[0][1].set_title("cavity a final")
    plot_wigner(rho_b_start, fig=fig,ax=axes[1][0])
    axes[1][0].set_title("cavity b start")
    plot_wigner(rho_b_final, fig=fig,ax=axes[1][1])
    axes[1][1].set_title("cavity b final")