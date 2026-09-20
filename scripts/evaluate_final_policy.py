#!/usr/bin/env python3
"""Deterministic final-policy evaluation for the BAM Genesis environment.

This script deliberately uses the same actor normalization, 61-D observation
path, BAM torque actuator, and action adapter as training.  It does not update
the policy or environment parameters.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _scalar(x: torch.Tensor) -> float:
    return float(x.detach().reshape(-1)[0].cpu())


def _vec(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().reshape(-1).cpu()]


def _orientation_error(projected_gravity: torch.Tensor) -> float:
    # The Genesis adapter's upright projected gravity is [0, 0, 1].
    z = projected_gravity[..., 2].clamp(-1.0, 1.0)
    return _scalar(torch.acos(z))


def _command(env, name: str) -> float:
    values = {"stand": 0.0, "forward": 0.2}
    if name not in values:
        raise ValueError(name)
    env.commands.command[:, :] = 0.0
    env.commands.command[:, 0] = values[name]
    return values[name]


def _csv_header() -> list[str]:
    names = ["mode", "step", "time", "commanded_vx", "actual_vx", "tracking_error",
             "base_x", "base_y", "base_z", "quat_w", "quat_x", "quat_y", "quat_z",
             "orientation_error", "reward", "done", "fell", "foot_left_contact",
             "foot_right_contact", "self_collision"]
    names += [f"action_{i}" for i in range(14)]
    names += [f"bam_torque_{i}" for i in range(14)]
    names += [f"joint_pos_{i}" for i in range(14)]
    names += [f"joint_vel_{i}" for i in range(14)]
    return names


@torch.no_grad()
def run_mode(env, model, norm, mode: str, steps: int, writer) -> dict[str, float | int | bool | str]:
    command_vx = _command(env, mode)
    obs = env.reset()
    # reset() resamples the command; set the evaluation command after reset.
    command_vx = _command(env, mode)
    initial_pos = None
    rows: list[dict[str, object]] = []
    actual_vx: list[float] = []
    track_err: list[float] = []
    heights: list[float] = []
    orient: list[float] = []
    rewards: list[float] = []
    abs_actions: list[float] = []
    abs_torque: list[float] = []
    max_torque = 0.0
    fell = False
    done_step = steps
    numerical_failure = ""
    pos = quat = None
    start = time.perf_counter()

    for step in range(steps):
        actor_input = norm(obs)
        action = model.actor(actor_input)  # deterministic mean action
        if not torch.isfinite(action).all():
            raise FloatingPointError("deterministic policy produced NaN/Inf action")
        try:
            obs, reward, done, info = env.step(action)
        except (FloatingPointError, RuntimeError) as exc:
            # Preserve the trajectory accumulated before a policy destabilizes
            # the simulator.  Evaluation must report this rather than masking
            # it with clipping or an implicit reset.
            numerical_failure = f"{type(exc).__name__}: {exc}"
            done_step = step
            break
        q, qd, pos, quat, vel, ang = env._state()
        gravity = env._projected_gravity(quat)
        foot_pos, foot_vel, foot_contact, self_collision, _ = env._contact_state()
        done_bool = bool(done[0].item())
        terms = info.get("termination_terms", {})
        fell_bool = bool((terms.get("bad_orientation", done)[0] | terms.get("nan_state", torch.zeros_like(done))[0]).item())
        fell = fell or fell_bool
        if initial_pos is None:
            initial_pos = pos[0].detach().clone()
        vx = _scalar(vel[:, 0])
        height = _scalar(pos[:, 2])
        ori = _orientation_error(gravity)
        torques = env.last_bam_torque[0]
        action_row = _vec(action[0])
        torque_row = _vec(torques)
        row: dict[str, object] = {
            "mode": mode, "step": step + 1, "time": (step + 1) * env.control_dt,
            "commanded_vx": command_vx, "actual_vx": vx, "tracking_error": vx - command_vx,
            "base_x": _scalar(pos[:, 0]), "base_y": _scalar(pos[:, 1]), "base_z": height,
            "quat_w": _scalar(quat[:, 0]), "quat_x": _scalar(quat[:, 1]),
            "quat_y": _scalar(quat[:, 2]), "quat_z": _scalar(quat[:, 3]),
            "orientation_error": ori, "reward": _scalar(reward), "done": int(done_bool),
            "fell": int(fell_bool), "foot_left_contact": int(foot_contact[0, 0].item()),
            "foot_right_contact": int(foot_contact[0, 1].item()),
            "self_collision": int(self_collision[0].item()),
        }
        row.update({f"action_{i}": v for i, v in enumerate(action_row)})
        row.update({f"bam_torque_{i}": v for i, v in enumerate(torque_row)})
        row.update({f"joint_pos_{i}": v for i, v in enumerate(_vec(q[0]))})
        row.update({f"joint_vel_{i}": v for i, v in enumerate(_vec(qd[0]))})
        writer.writerow(row)
        actual_vx.append(vx); track_err.append(vx - command_vx); heights.append(height)
        orient.append(ori); rewards.append(_scalar(reward)); abs_actions.append(sum(abs(v) for v in action_row) / 14.0)
        abs_torque.append(sum(abs(v) for v in torque_row) / 14.0); max_torque = max(max_torque, max(abs(v) for v in torque_row))
        if done_bool:
            done_step = step + 1
            break

    if initial_pos is None or pos is None:
        raise RuntimeError(f"{mode} produced no valid simulation step: {numerical_failure}")
    final_pos = pos[0].detach()
    elapsed = done_step * env.control_dt
    return {
        "mode": mode, "commanded_vx": command_vx, "steps": done_step,
        "survival_time": elapsed, "mean_actual_vx": sum(actual_vx) / len(actual_vx),
        "tracking_rmse": math.sqrt(sum(e * e for e in track_err) / len(track_err)),
        "forward_distance": _scalar(final_pos[0] - initial_pos[0]),
        "translation_drift": _scalar(torch.linalg.vector_norm((final_pos - initial_pos)[:2])),
        "mean_base_height": sum(heights) / len(heights), "final_base_height": heights[-1],
        "mean_orientation_error": sum(orient) / len(orient), "fall": fell,
        "mean_abs_action": sum(abs_actions) / len(abs_actions),
        "mean_abs_bam_torque": sum(abs_torque) / len(abs_torque), "max_abs_bam_torque": max_torque,
        "numerical_failure": numerical_failure,
        "elapsed_wall": time.perf_counter() - start,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/model_0099.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--output", default="runs/final_validation/final_policy_rollout.csv")
    args = p.parse_args()

    from microduck_genesis.env import MicroDuckGenesisEnv
    from microduck_genesis.ppo import ActorCritic, RunningNorm

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    iteration = int(state.get("iteration", -1))
    env = MicroDuckGenesisEnv(num_envs=1, headless=not args.viewer, device=args.device,
                             seed=0, evaluation_command="forward", randomization=False,
                             domain_randomization=False, sensor_noise=True, sensor_delay=True)
    model = ActorCritic().to(env.device); model.load_state_dict(state["model"]); model.eval()
    norm = RunningNorm(61).to(env.device); norm.load_state_dict(state["norm"]); norm.eval()
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_header()); writer.writeheader()
        stand = run_mode(env, model, norm, "stand", args.steps, writer)
        forward = run_mode(env, model, norm, "forward", args.steps, writer)

    print(f"checkpoint: {checkpoint}")
    print(f"training_iteration: {iteration}")
    for result in (stand, forward):
        print(f"\n{result['mode'].upper()}:")
        print(f"survival_time: {result['survival_time']:.3f} s")
        if result["mode"] == "stand":
            print(f"base_height_mean: {result['mean_base_height']:.6f} m")
            print(f"translation_drift: {result['translation_drift']:.6f} m")
        else:
            print(f"commanded_vx: {result['commanded_vx']:.6f} m/s")
            print(f"mean_actual_vx: {result['mean_actual_vx']:.6f} m/s")
            print(f"tracking_rmse: {result['tracking_rmse']:.6f} m/s")
            print(f"forward_distance: {result['forward_distance']:.6f} m")
            print(f"base_height_mean: {result['mean_base_height']:.6f} m")
        print(f"orientation_deviation_mean: {result['mean_orientation_error']:.6f} rad")
        print(f"fall: {'YES' if result['fall'] else 'NO'}")
        if result["numerical_failure"]:
            print(f"numerical_failure: {result['numerical_failure']}")
        print(f"mean_abs_action: {result['mean_abs_action']:.6f}")
        print(f"mean_abs_bam_torque: {result['mean_abs_bam_torque']:.6f} Nm")
        print(f"max_abs_bam_torque: {result['max_abs_bam_torque']:.6f} Nm")
        print(f"wall_time: {result['elapsed_wall']:.2f} s")
    print(f"trajectory_csv: {out}")


if __name__ == "__main__":
    main()
