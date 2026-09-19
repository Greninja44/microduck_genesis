# PPO backend choice

The upstream trainer is `rsl_rl`, invoked through MJLab's
`VelocityOnPolicyRunner` (`tasks/__init__.py`) and wrapped around a
`ManagerBasedRlEnv`. It is not available in this Genesis virtual environment
(`import rsl_rl` fails), and its wrapper expects MJLab managers, critic
observations, and reset semantics. Reusing it would require implementing that
MJLab adapter, rather than a Genesis environment.

`microduck_genesis.ppo` is therefore a small PyTorch PPO implementation. It
preserves the upstream feed-forward model, ELU widths, Gaussian scalar std,
rollout length, gamma, lambda, clipping, entropy coefficient, value loss,
epochs, minibatches, learning rate, and gradient clipping. Its implementation
does not yet include rsl_rl's adaptive KL schedule, symmetry hook (disabled
upstream), or privileged critic stream.
