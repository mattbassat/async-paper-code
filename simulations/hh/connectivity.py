"""Synaptic and connectivity parameters of the HH network (adapted from Carlu et al., 2020)."""
import numpy as np
from brian2 import Synapses, nS, seed


def get_connectivity_and_synapses_matrix(NAME='CONFIG1', number=2):
    """2x2 object array M of synapse dictionaries, M[pre, post] (0 = exc, 1 = inh)."""
    if NAME != 'CONFIG1':
        raise ValueError(f"unknown network configuration {NAME!r}")
    M = np.empty((number, number), dtype=object)
    exc_pop = {'p_conn': 0.05, 'Q': 1.5, 'Tsyn': 5., 'Erev': 0.}
    inh_pop = {'p_conn': 0.05, 'Q': 5, 'Tsyn': 5., 'Erev': -80.}
    M[:, 0] = [exc_pop.copy(), inh_pop.copy()]  # postsynaptic: exc
    M[:, 1] = [exc_pop.copy(), inh_pop.copy()]  # postsynaptic: inh
    M[0, 0]['name'], M[1, 0]['name'] = 'ee', 'ie'
    M[0, 1]['name'], M[1, 1]['name'] = 'ei', 'ii'
    M[0, 0]['Ntot'], M[0, 0]['gei'] = 10000, 0.2
    M[0, 0]['ext_drive'] = 4.
    M[0, 0]['afferent_exc_fraction'] = 1.
    return M


def build_up_recurrent_connections_for_2_pop(Pops, syn_conn_matrix, SEED=1):
    """Random recurrent E->E, E->I, I->E and I->I connections (no autapses)."""
    seed(SEED)
    exc_neurons, inh_neurons = Pops
    P = syn_conn_matrix
    exc_exc = Synapses(exc_neurons, exc_neurons, model='w:siemens', on_pre='Gee_post += w')
    exc_exc.connect('i!=j', p=P[0, 0]['p_conn'])
    exc_exc.w = P[0, 0]['Q']*nS
    exc_inh = Synapses(exc_neurons, inh_neurons, model='w:siemens', on_pre='Gei_post += w')
    exc_inh.connect('i!=j', p=P[0, 1]['p_conn'])
    exc_inh.w = P[0, 1]['Q']*nS
    inh_exc = Synapses(inh_neurons, exc_neurons, model='w:siemens', on_pre='Gie_post += w')
    inh_exc.connect('i!=j', p=P[1, 0]['p_conn'])
    inh_exc.w = P[1, 0]['Q']*nS
    inh_inh = Synapses(inh_neurons, inh_neurons, model='w:siemens', on_pre='Gii_post += w')
    inh_inh.connect('i!=j', p=P[1, 1]['p_conn'])
    inh_inh.w = P[1, 1]['Q']*nS
    return exc_exc, exc_inh, inh_exc, inh_inh
