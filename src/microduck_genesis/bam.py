"""Vectorized BAM M6 actuator equations used by the upstream task.

The Genesis backend cannot write MuJoCo's per-DOF friction constraint, so this
module exposes the motor and friction budget explicitly.  The motor equations
are exact; applying the budget as a signed torque is a close approximation to
MuJoCo's native static-friction constraint.
"""
from __future__ import annotations
import torch

# xl330/m6.json from better-actuator-models, commit 62bd8ce1.
KT = 0.36601349688984386
RESISTANCE = 2.8113923539223227
ARMATURE = 0.0018077432831600838
FRICTION_BASE = 0.004771183165566
FRICTION_STRIBECK = 0.004676345799486616
LOAD_MOTOR = 0.2667860954283698
LOAD_EXTERNAL = 8.515871897059342e-06
LOAD_MOTOR_STRIBECK = 1.0722918395099123e-05
LOAD_EXTERNAL_STRIBECK = 0.08077928978935671
LOAD_MOTOR_QUAD = 0.009972471242139415
LOAD_EXTERNAL_QUAD = 0.004902565732332559
DTHETA_STRIBECK = 2.890372094130307
ALPHA = 8.683259907618984
FRICTION_VISCOUS = 0.005359668274599504
VIN_MIN = 6.0
ERROR_GAIN = 0.0028773775022263564

def bam_voltage(target: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
                vin: torch.Tensor, kp: float = 200.0) -> torch.Tensor:
    duty = (target - q) * kp * ERROR_GAIN
    # XL330 firmware current limiter: bound duty around the back-EMF center,
    # then apply the physical PWM limit.
    span = RESISTANCE * 1.75 / vin
    center = KT * qd / vin
    duty = torch.maximum(torch.minimum(duty, center + span), center - span)
    return vin * torch.clamp(duty, -1.0, 1.0)

def bam_motor_torque(voltage: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    return KT * voltage / RESISTANCE - (KT * KT) * qd / RESISTANCE

def bam_friction_budget(motor_torque: torch.Tensor, external_torque: torch.Tensor,
                        qd: torch.Tensor, scale: torch.Tensor | float = 1.0) -> torch.Tensor:
    s = torch.exp(-torch.abs(qd / DTHETA_STRIBECK).pow(ALPHA))
    ext, mot = torch.abs(external_torque), torch.abs(motor_torque)
    gearbox = torch.abs(external_torque * LOAD_EXTERNAL - motor_torque * LOAD_MOTOR)
    gearbox_s = torch.abs(external_torque * LOAD_EXTERNAL_STRIBECK - motor_torque * LOAD_MOTOR_STRIBECK)
    drive = (mot > ext).to(motor_torque.dtype)
    quad = drive * LOAD_EXTERNAL_QUAD * ext.square() + (1.0 - drive) * LOAD_MOTOR_QUAD * mot.square()
    budget = FRICTION_BASE + s * FRICTION_STRIBECK + gearbox + s * gearbox_s + s * quad
    return budget * scale

def bam_torque(target: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
               vin: torch.Tensor, external_torque: torch.Tensor | None = None,
               friction_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
    if external_torque is None:
        external_torque = torch.zeros_like(q)
    voltage = bam_voltage(target, q, qd, vin)
    motor = bam_motor_torque(voltage, qd)
    # Genesis force control has no native static-friction solver.  The signed
    # viscous/Coulomb approximation preserves the direction and magnitude.
    budget = bam_friction_budget(motor, external_torque, qd, friction_scale)
    return motor - torch.sign(qd) * budget - FRICTION_VISCOUS * qd
