"""The teleop dataset schema stays in the shape LeRobot policies train on."""

from i2rt.robot_learning.lerobot_recorder import joint_names, teleop_features


def test_joint_names_include_the_gripper_when_present() -> None:
    assert joint_names(7, 6) == ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    assert joint_names(6, 6) == ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def test_features_name_both_cameras_and_the_joints() -> None:
    features = teleop_features(joint_names(7, 6))
    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["shape"] == (7,)
    assert features["action"]["names"][-1] == "gripper"
    for camera in ("overhead", "right_wrist"):
        image = features[f"observation.images.{camera}"]
        assert image["dtype"] == "video"
        assert image["shape"] == (480, 640, 3)
        assert image["names"] == ["height", "width", "channels"]
