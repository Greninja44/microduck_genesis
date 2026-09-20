# Domain-randomization parity

Recovered from the active flat velocity task configuration.

| Field | Distribution/range | Frequency | Genesis status |
|---|---|---|---|
| trunk CoM | uniform ±3 mm initially, curriculum to ±8 mm | reset | APPROXIMATED |
| head assembly CoM | uniform ±3 mm initially, curriculum | reset | MISSING |
| mass/inertia | joint pseudo-inertia scale 0.95–1.05 | startup per env | MISSING |
| BAM friction budget | scale 0.9–1.1 | reset per env | APPROXIMATED |
| reflected armature | scale 0.9–1.1 | reset | MISSING |
| encoder bias | uniform [-0.015, 0.015] rad | per env/reset | MATCHED in bookkeeping |
| IMU mounting rotation | random axis, up to 6° | per env/reset | MISSING |
| velocity pushes | x/y uniform [-0.3, 0.3] m/s | every 3–6 s | MISSING |
| motor KP/KD | KP 0.85–1.15, KD 0.9–1.1 | disabled upstream | NOT APPLICABLE |

Genesis currently reproduces command and encoder-bias randomization and keeps
randomization state per environment. Model-field writes and external pushes are
deferred because the corresponding Genesis setters need explicit validation.
No additional randomization was invented.
