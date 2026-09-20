# Post-fix training gate

The upstream velocity configuration uses `GaussianDistribution` with scalar
`init_std=1.0`; actions are raw Gaussian joint-position offsets with scale 1.0.
There is no tanh or policy-side action clipping in the port. Genesis now keeps
that behavior and aborts if a finite action exceeds 10 rad. Ten radians is a
diagnostic limit, not a physical clamp: every MicroDuck actuated joint range in
`robot_walk.xml` is below 3 rad.

Training logs include action mean/max/std, BAM torque mean/max/saturation,
losses, gradient norm, parameter norms, and actor `log_std` bounds. Any
non-finite value or action explosion aborts the run.

The fresh post-fix run completed five PPO iterations with finite sampled
rollouts. Iteration-5 deterministic evaluation was not healthy: forward
survived about 1.04 s before falling, while stand evaluation hit the explicit
10-rad action diagnostic on its second control step. Training therefore stopped
before iteration 20 and no long run was started.

Checkpoint parameter parity is exact (maximum difference 0), normalization is
restored, and no-noise/no-delay observations match at reset, steps 1, 2, and 5
with maximum error 0. The deterministic forward actor mean over 1,000 rollout
observations had mean absolute value 0.198, p95 0.501, p99 0.691, and maximum
1.051. The stand failure is therefore a closed-loop physical/state-distribution
instability after the first deterministic action, rather than a checkpoint or
evaluation-loader mismatch.
