import quantities as pq
import numpy as np
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

# fb weights:
fb_weights = [0.6, 0.9]

# diameters
mask_size = np.linspace(0, 6, 40) * pq.deg

# list to store spatiotemporal summation curves for each weight
responses = []

for w_fb in fb_weights:

    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=10, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
    
    # create spatial kernels
    Wg_s = spl.create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg)
    Krg_s = spl.create_gauss_ft(A=1, a=0.1*pq.deg)  
    Kcr_s = spl.create_delta_ft()
    Krcr_cen_s = spl.create_gauss_ft(A=1, a=(0.1)*pq.deg)
    Krcr_sur_s = spl.create_gauss_ft(A=2, a=(0.9)*pq.deg)
    
    # create temporal kernels
    Wg_t = tpl.create_biphasic_ft(phase_duration =42.5*pq.ms, damping_factor =0.38, delay =0 *pq.ms) 
    Krg_t = tpl.create_exp_decay_ft(18 *pq.ms, delay = 0 *pq.ms) 
    Kcr_t = tpl.create_delta_ft()     
    Krcr_cen_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 0 *pq.ms)  # decay: 10 ms, delay: 30 ms
    Krcr_sur_t = tpl.create_exp_decay_ft(1 *pq.ms, delay = 30 *pq.ms) # decay: 20 ms, delay: 30 ms
    
    # create neurons
    ganglion = network.create_ganglion_cell(kernel=(Wg_s, Wg_t))
    relay = network.create_relay_cell()
    cortical_cen = network.create_cortical_cell()
    cortical_sur = network.create_cortical_cell()
                    
    # connect neurons
    network.connect(ganglion, relay, (Krg_s, Krg_t), 1.0)  
    network.connect(cortical_cen, relay, (Krcr_cen_s, Krcr_cen_t), w_fb)
    network.connect(cortical_sur, relay, (Krcr_sur_s, Krcr_sur_t), -w_fb) 
    network.connect(relay, cortical_cen, (Kcr_s, Kcr_t), 1.0)       
    network.connect(relay, cortical_sur, (Kcr_s, Kcr_t), 1.0)    
    

    st_summation_curve = np.zeros([len(mask_size), integrator.Nt]) / pq.s
    for i, d in enumerate(mask_size):
        # create stimulus
        k_pg = integrator.spatial_freqs[3]  # 0 for flashing spot                         
        w_pg = integrator.temporal_freqs[1] # 0 for static stimuli
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
vmin = -10
vmax = 10

im1 = ax1.imshow(responses[0], extent=extent, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="Blues_r")
ax1.set_title("Response (FB weight={})".format(fb_weights[0]))
ax1.set_ylabel("Patch size (deg)")
ax1.set_xlabel("Time (ms)")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(responses[1], extent=extent, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="Blues_r")
ax2.set_title("Resoinse (FB weight={})".format(fb_weights[1]))
ax2.set_xlabel("Time (ms)")
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
