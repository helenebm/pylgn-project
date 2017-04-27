import quantities as pq
import numpy as np
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

# fb weights:
fb_weights = [0, -0.6]

# diameters
mask_size = np.linspace(0, 6, 40) * pq.deg

# list to store spatiotemporal summation curves for each weight
responses = []

for w_c in fb_weights:

    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=5, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
    
    # create spatial kernels
         # Originaly params: A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg 
    Wg_s = spl.create_dog_ft(A=1, a=0.25*pq.deg, B=0.85, b=0.5*pq.deg)      # Stimuli   -> Ganglion, (edog-paper)
    Krg_s = spl.create_gauss_ft(A=1, a=0.1*pq.deg)                           # Gangllion -> Relay     (honda-paper)
    Kcr_s = spl.create_delta_ft()                                            # Relay     -> Cortical_cen_sur 
    Krcr_cen_s = spl.create_gauss_ft(A=1, a=(0.1)*pq.deg)                    # Cortical  -> Relay_cen
    Krcr_sur_s = spl.create_gauss_ft(A=2, a=(0.9)*pq.deg)                    # Cortical  -> Relay_sur
    
    # create temporal kernels
    Wg_t = tpl.create_biphasic_ft(phase_duration =42.5*pq.ms, damping_factor =0.38, delay =0 *pq.ms) # param. values from Norheim, Wyller, Einevoll                                       # Stimnuli -> Ganglion
    Krg_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 1 *pq.ms)              # Ganglion      -> Relay
    Kcr_t = tpl.create_delta_ft()                                            # Relay         -> Cortical_cen_sur
    Krcr_cen_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 2 *pq.ms)        # Cortical_cen  -> Relay
    Krcr_sur_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 20 *pq.ms)         # Cortical_sur  -> Relay
    
    # create neurons
    ganglion = network.create_ganglion_cell(kernel=(Wg_s, Wg_t))
    relay = network.create_relay_cell()
    cortical_cen = network.create_cortical_cell()
    cortical_sur = network.create_cortical_cell()
                    
    # connect neurons
    network.connect(ganglion, relay, (Krg_s, Krg_t), 1.0)                   # Ganglion      -> Relay
    network.connect(cortical_cen, relay, (Krcr_cen_s, Krcr_cen_t), w_c)     # Cortical_cen  -> Relay
    network.connect(cortical_sur, relay, (Krcr_sur_s, Krcr_sur_t), w_c)     # Cortical_sur  -> Relay
    network.connect(relay, cortical_cen, (Kcr_s, Kcr_t), 1.0)               # Relay         -> Cortical_cen
    network.connect(relay, cortical_sur, (Kcr_s, Kcr_t), 1.0)               # Relay         -> Cortical_sur

    st_summation_curve = np.zeros([len(mask_size), integrator.Nt]) / pq.s
    for i, d in enumerate(mask_size):
        # create stimulus
        k_pg = integrator.spatial_freqs[3]                                  # spatial_freqs[0] -> flashing splot
        w_pg = integrator.temporal_freqs[1]                                 # temporal_freqs[0] -> static stimulus 
        stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=k_pg,
                                                          angular_freq=w_pg,
                                                          mask_size=d)
        network.set_stimulus(stimulus)

        # compute
        network.compute_response(relay, recompute_ft=True)
        st_summation_curve[i, :] = relay.center_response
    responses.append(st_summation_curve)
    # clear network
    network.clear()

# visualize
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey="row")

# xmin, xmax, ymin, ymax:
extent = [integrator.times.min(), integrator.times.max(), mask_size.min(), mask_size.max()]
vmin = -0.6
vmax = 0.6

im1 = ax1.imshow(responses[0], extent=extent, origin="lower", aspect="auto",
                 vmin=vmin, vmax=vmax)
ax1.set_title("Response (FB weight={})".format(fb_weights[0]))
ax1.set_ylabel("Patch size (deg)")
ax1.set_xlabel("Time (ms)")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(responses[1], extent=extent, origin="lower", aspect="auto",
                 vmin=vmin, vmax=vmax)
ax2.set_title("Resoinse (FB weight={})".format(fb_weights[1]))
ax2.set_xlabel("Time (ms)")
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
