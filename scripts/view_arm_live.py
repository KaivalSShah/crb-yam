"""Mirror a real YAM arm's joint angles in the MuJoCo viewer, read-only.

Reads each arm motor's output-shaft position register (``xout``) over CAN and poses the sim model to match,
so you can move the arm by hand and watch it in simulation. Only register reads are sent: no motor is
enabled and nothing is commanded.

macOS (gs_usb adapter; sudo is needed to claim the USB device, mjpython to open the viewer):
    sudo .venv/bin/mjpython scripts/view_arm_live.py --channel <adapter serial>

List adapter serials with:
    .venv/bin/python -c "from gs_usb.gs_usb import GsUsb; print([d.gs_usb.serial_number for d in GsUsb.scan()])"
"""

import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from i2rt.motor_config_tool.dm_motor_registers import BUS_ERRORS, DMRegAddr, Interface, _open_bus, read_register
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config, combine_arm_and_gripper_xml


@dataclass
class Args:
    channel: str
    """gs_usb: adapter serial number (or index 0, 1, ...). socketcan: interface name, e.g. can0."""
    interface: Interface = "gs_usb"
    """socketcan (Linux) or gs_usb (candleLight adapter without SocketCAN, e.g. macOS)."""
    arm: str = "yam"
    """Arm variant (yam, yam_pro, yam_ultra, yam_ultra_2, big_yam)."""
    gripper: str = "linear_4310"
    """Gripper shown on the model; visual only, its motor is not read."""


def main(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    hw = _load_arm_config(arm_type)
    motor_ids = [motor_id for motor_id, _ in hw.motor_list]
    directions = np.array(hw.directions, dtype=float)

    model = mujoco.MjModel.from_xml_path(
        combine_arm_and_gripper_xml(arm_type, GripperType.from_string_name(args.gripper))
    )
    data = mujoco.MjData(model)
    joint_pos = np.zeros(len(motor_ids))
    answered = [False] * len(motor_ids)

    with _open_bus(args.channel, args.interface) as iface, mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i, motor_id in enumerate(motor_ids):
                try:
                    raw = float(read_register(iface, motor_id, DMRegAddr.XOUT))
                except BUS_ERRORS:
                    answered[i] = False  # keep the last good angle, but show the motor as not answering
                    continue
                answered[i] = True
                # Same mapping as the real driver for a zero motor offset: direction, then wrap into (-pi, pi].
                joint_pos[i] = (directions[i] * raw + np.pi) % (2 * np.pi) - np.pi
            cells = [
                f"j{i + 1}: {q:+.3f}" if ok else f"j{i + 1}: no reply"
                for i, (q, ok) in enumerate(zip(joint_pos, answered, strict=True))
            ]
            print("\r" + "  ".join(cells), end="", flush=True)
            data.qpos[: len(joint_pos)] = joint_pos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main(tyro.cli(Args))
