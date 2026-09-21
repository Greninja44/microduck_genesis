# Invalid pre-fix training outputs

The checkpoints in the repository-level `checkpoints/` directory were created
before commit `9ecd384`, when upright states were incorrectly terminated by a
projected-gravity sign error. They are retained for debugging history only and
must not be resumed or used as locomotion evidence.

The post-fix validation run uses fresh policy, optimizer, and normalization
state under `checkpoints/postfix/`.
