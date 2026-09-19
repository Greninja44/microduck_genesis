#!/usr/bin/env python3
"""Concise static report of the exact upstream walking model used for locomotion."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "upstream/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml"
CFG = ROOT / "upstream/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py"


def inherited_default(root: ET.Element, class_name: str, tag: str) -> dict[str, str]:
    for default in root.findall("default"):
        for nested in default.findall(f".//default[@class='{class_name}']"):
            child = nested.find(tag)
            if child is not None:
                return child.attrib
    return {}


def main() -> int:
    if not MODEL.is_file():
        print(f"FAIL: missing model: {MODEL}")
        return 1
    root = ET.parse(MODEL).getroot()
    compiler = root.find("compiler")
    joints = [j for j in root.findall(".//joint") if j.get("name")]
    actuators = list(root.findall(".//actuator/*"))
    inertials = list(root.findall(".//inertial"))
    meshes = list(root.findall(".//mesh"))
    collision_geoms = [g for g in root.findall(".//geom") if g.get("name", "").endswith("_collision")]
    print("MODEL: MJCF/XML")
    print(f"path: {MODEL.relative_to(ROOT)}")
    print(f"meshdir: {compiler.get('meshdir') if compiler is not None else 'MJCF default'}")
    print(f"mesh_assets: {len(meshes)}  inertial_bodies: {len(inertials)}  named_collision_geoms: {len(collision_geoms)}")
    print(f"free_base: {root.find('.//freejoint') is not None}")
    print(f"servo_dofs: {len(joints)}  actuators: {len(actuators)}")
    print("joint order (name | axis | range radians):")
    for joint in joints:
        print(f"  {joint.get('name')} | {joint.get('axis')} | {joint.get('range')}")
    print("actuator order:")
    for actuator in actuators:
        print(f"  {actuator.get('name')} -> {actuator.get('joint')} ({actuator.tag}, class={actuator.get('class')})")
    print("body hierarchy (parent -> body):")
    worldbody = root.find("worldbody")
    def show_bodies(parent: ET.Element, parent_name: str) -> None:
        for body in parent.findall("body"):
            name = body.get("name", "unnamed")
            inertial = body.find("inertial")
            mass = inertial.get("mass") if inertial is not None else "0"
            print(f"  {parent_name} -> {name} (mass={mass})")
            show_bodies(body, name)
    if worldbody is not None:
        show_bodies(worldbody, "world")
    print("named floor-contact geometry:")
    for geom in collision_geoms:
        print(f"  {geom.get('name')}: mesh={geom.get('mesh')} friction={geom.get('friction', 'MJCF default')}")
    pd = inherited_default(root, "chosen_actuator", "position")
    joint_defaults = inherited_default(root, "chosen_actuator", "joint")
    print(f"XML position actuator defaults: kp={pd.get('kp')} kv={pd.get('kv')} force={pd.get('forcerange')} ctrl={pd.get('ctrlrange')}")
    print(f"XML joint defaults: damping={joint_defaults.get('damping')} frictionloss={joint_defaults.get('frictionloss')} armature={joint_defaults.get('armature')}")
    print("RL override: BAM M6 voltage actuator, kp_fw=200, delay=3..6 simulation ticks; action scale=1.0.")
    print("RL timing: 50 Hz policy control (0.02 s); source comments state velocity simulation dt=0.005 s.")
    print("RL observations: actor 61D = 48 proprioception + [twist(3), head_pose(4), body_pose(6)].")
    print(f"locomotion config: {CFG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
