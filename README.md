# Sequential Latent Variable Agent (NR-SSM)

An implementation of a non-reconstructive forward model based on:

$$\min_{q(z_{t:t+1}|o_{t:t+1})} I(\{Z_{t-1},O_t,A_{t-1}\};Z_t) - I(\{Z_t,A_t\};Z_{t+1)$$

Learned through InfoNCE methods.