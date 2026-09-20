# PPO configuration parity audit

| Parameter | Upstream MJLab/rsl_rl | Genesis runner | Status |
|---|---:|---:|---|
| actor/critic MLP | 512, 256, 128 ELU | 512, 256, 128 ELU | MATCHED |
| actor input/output | 61 / 14 | 61 / 14 | MATCHED |
| critic input | 76 | 76 | MATCHED |
| distribution | scalar raw Gaussian | scalar raw Gaussian | MATCHED |
| initial std | 1.0 | 1.0 | MATCHED |
| learning rate | 1e-3 | 1e-3 | MATCHED |
| gamma | 0.99 | 0.99 | MATCHED |
| GAE lambda | 0.95 | 0.95 | MATCHED |
| PPO clip | 0.2 | 0.2 | MATCHED |
| epochs/minibatches | 5 / 4 | 5 / 4 | MATCHED |
| entropy coefficient | 0.01 | 0.01 | MATCHED |
| value coefficient | 1.0 | 1.0 | MATCHED |
| max gradient norm | 1.0 | 1.0 | MATCHED |
| rollout horizon | 24/env | 24/env | MATCHED |
| adaptive KL schedule | desired_kl=0.01; /1.5 above 2x, *1.5 below 0.5x; bounds [1e-5,1e-2] | same | MATCHED |
| symmetry augmentation | disabled in upstream velocity config | absent | NOT APPLICABLE |
| timeout bootstrap | rsl_rl adds gamma·V on timeouts | Genesis adds gamma·V only for timeouts | MATCHED |

The action distribution itself has not been modified. The current failure is
closed-loop state distribution, not a checkpoint or actor ABI mismatch.
