import quantities as pq
import numpy as np
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

center_excit_norm = 0.637718
surround_inhib_norm = 0.0483276

a_rcr_vec = np.linspace(0.1, 3.0, 10) * pq.deg
b_rcr_vec = np.linspace(0.1, 3.0, 10) * pq.deg

center_excit = np.zeros([len(b_rcr_vec), len(a_rcr_vec)])
surround_inhib = np.zeros([len(b_rcr_vec), len(a_rcr_vec)])

for i, b_rc in enumerate(b_rcr_vec):
    for j, a_rc in enumerate(a_rcr_vec):
        network = pylgn.Network()

        # create integrator
        integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
        
        # create spatial kernels
        Wg_s = spl.create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg)      # Stimuli   -> Ganglion, (edog-paper)
        Krg_s = spl.create_gauss_ft(A=1, a=0.1*pq.deg)                           # Gangllion -> Relay     (honda-paper)
        Kcr_s = spl.create_delta_ft()                                            # Relay     -> Cortical_cen_sur 
        Krcr_cen_s = spl.create_gauss_ft(A=1, a=(a_rc))                          # Cortical  -> Relay_cen
        Krcr_sur_s = spl.create_gauss_ft(A=2, a=(b_rc))                          # Cortical  -> Relay_sur
        
         # create temporal kernels
        Wg_t = tpl.create_biphasic_ft(phase_duration =42.5*pq.ms, damping_factor =0.38, delay =0 *pq.ms) # param. values from Norheim, Wyller, Einevoll                                       # Stimnuli -> Ganglion
        Krg_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 0 *pq.ms)              # Ganglion      -> Relay
        Kcr_t = tpl.create_delta_ft()                                            # Relay         -> Cortical_cen_sur
        Krcr_cen_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 0 *pq.ms)         # Cortical_cen  -> Relay
        Krcr_sur_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 0 *pq.ms)         # Cortical_sur  -> Relay
        
        # create neurons                               (Wg_s, delta_t)
        ganglion = network.create_ganglion_cell(kernel=(Wg_s, Wg_t))
        relay = network.create_relay_cell()
        cortical_cen = network.create_cortical_cell()
        cortical_sur = network.create_cortical_cell()
        
        # connect neurons
        network.connect(ganglion, relay, (Krg_s, Krg_t), 1.0)                    # Ganglion      -> Relay
        network.connect(cortical_cen, relay, (Krcr_cen_s, Krcr_cen_t), 0.1)      # Cortical_cen  -> Relay
        network.connect(cortical_sur, relay, (Krcr_sur_s, Krcr_sur_t), 0.1)      # Cortical_sur  -> Relay
        network.connect(relay, cortical_cen, (Kcr_s, Kcr_t), 1.0)                # Relay         -> Cortical_cen
        network.connect(relay, cortical_sur, (Kcr_s, Kcr_t), 1.0)                # Relay         -> Cortical_sur
        

        network.compute_irf(relay)

        center_excit[i, j] = np.real(relay.irf[0].max()) / center_excit_norm
        surround_inhib[i, j] = np.real(relay.irf[0].min()) / surround_inhib_norm


# visualize
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey="row")

# xmin, xmax, ymin, ymax:
extent = [a_rcr_vec.min(), a_rcr_vec.max(), b_rcr_vec.min(), b_rcr_vec.max()]


im1 = ax1.imshow(center_excit, extent=extent, origin="lower", aspect="auto")
ax1.set_title("Center excitation")
ax1.set_ylabel("$b_{rcr}$")
ax1.set_xlabel("$a_{rcr}$")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(surround_inhib, extent=extent, origin="lower", aspect="auto")
ax2.set_title("Surround inhibition")
ax2.set_xlabel("$a_{rcr}$")
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
