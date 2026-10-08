"""Single-cell Hodgkin-Huxley models (regular-spiking excitatory and fast-spiking inhibitory).

Adapted from Carlu et al. (2020, J. Neurophysiol.).  The conductance densities and spike
thresholds VT were chosen by a grid search so that the passive membrane properties
(Cm = 200 pF, gL = 10 nS, EL = -70 mV) match the AdEx cells of the rest of the paper; the
M-current (adaptation) is switched off, matching the AdEx networks with b = 0.
Units: Cm in pF, conductances in nS, potentials in mV, tau_max in ms.
"""
import brian2


def get_neuron_params(NAME, name='', number=1):
    """Parameter dictionary of the cell type NAME ('HH_RS_grid' or 'HH_FS_grid')
    for a population of `number` neurons."""
    if NAME == 'HH_RS_grid':
        membrane_size = 200e-6  # cm^2
        # excitatory regular-spiking cell
        params = {
            'name': name, 'N': number,
            'Cm': 1.0*1.e6 * membrane_size,             # 200 pF
            'Gl': 0.05*1.e6 * membrane_size,            # 10 nS
            'GNa': 90*1.e-3 * 1.e9 * membrane_size,     # 18 uS
            'GK': 3.0*1.e-3 * 1.e9 * membrane_size,     # 0.6 uS
            'GM': 0.,                                   # no adaptation
            'El': -70.,
            'EK': -90.,
            'ENa': 50.,
            'VT': -53,
            'tau_max': 4000.,                           # unused (GM = 0)
            'hhtype': 1,
        }
    elif NAME == 'HH_FS_grid':
        membrane_size = 200e-6  # cm^2
        # inhibitory fast-spiking cell
        params = {
            'name': name, 'N': number,
            'Cm': 1.0e6 * membrane_size,                # 200 pF
            'Gl': 0.05e6 * membrane_size,               # 10 nS
            'GNa': 110*1.e-3 * 1.e9 * membrane_size,    # 22 uS
            'GK': 5.0*1.e-3 * 1.e9 * membrane_size,     # 1 uS
            'GM': 0,                                    # no M-current
            'El': -70.,
            'EK': -90.,
            'ENa': 50.,
            'VT': -55,
            'tau_max': 4000.,                           # unused (GM = 0)
            'hhtype': 1,
        }
    else:
        raise ValueError(f"unknown cell type {NAME!r}")
    return params.copy()


def get_membrane_equation(neuron_params, synaptic_array):
    """NeuronGroup of HH cells with conductance-based exponential synapses.

    Sodium, delayed-rectifier potassium and slow M-type potassium currents with Traub-Miles
    kinetics relative to the threshold parameter VT (Carlu et al. 2020).  El and VT are
    per-neuron state variables, set by the caller.  `synaptic_array` lists the presynaptic
    synapse types onto this population (keys name, Erev, Tsyn); each adds a conductance
    G<name> decaying with time constant Tsyn.  A spike is detected when V crosses 0 mV.
    """
    if 'tau_max' not in neuron_params:
        neuron_params['tau_max'] = 400.  # ms

    eqs = """
        dV/dt = (%(Gl)f*nS*(El - V) 
            - %(GNa)f*nS*(m*m*m)*h*(V-%(ENa)f*mV) 
            - %(GK)f*nS*(n*n*n*n)*(V - %(EK)f*mV) 
            - %(GM)f*nS*p*(V - %(EK)f*mV) 
            + I)/(%(Cm)f*pF) : volt
        dm/dt = 0.32*(mV**-1)*(13.*mV-V+VT)/
            (exp((13.*mV-V+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(V-VT-40.*mV)/
            (exp((V-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
        dn/dt = 0.032*(mV**-1)*(15.*mV-V+VT)/
            (exp((15.*mV-V+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-V+VT)/(40.*mV))/ms*n : 1
        dh/dt = 0.128*exp((17.*mV-V+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-V+VT)/(5.*mV)))/ms*h : 1
        dp/dt=((1./
            (exp(-(35.*mV+V)/(10.*mV))+1.))-p)/((%(tau_max)f*ms)/(3.3*( exp((35.*mV+V)/(20.*mV)) )+1.*(exp(-(35.*mV+V)/(20.*mV))) )) :1
        El : volt
        VT : volt
        """ % neuron_params

    # total synaptic current
    eqs += """
        I = I0 """
    for synapse in synaptic_array:
        Gsyn = 'G' + synapse['name']
        eqs += '+' + Gsyn + '*(%(Erev)f*mV - V)' % synapse
    eqs += ' : amp'

    # exponentially decaying synaptic conductances
    for synapse in synaptic_array:
        Gsyn = 'G' + synapse['name']
        eqs += """
        """ + 'd' + Gsyn + '/dt = -' + Gsyn + '*(1./(%(Tsyn)f*ms)) : siemens' % synapse
    eqs += """
        I0 : amp """

    return brian2.NeuronGroup(neuron_params['N'], model=eqs,
                              threshold='V > 0*mV',
                              refractory='V > 0*mV',
                              method='exponential_euler')
