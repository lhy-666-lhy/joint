from types import SimpleNamespace

import run_a6_o126c_zero_contact_live_probe as live


class FakeJoint:
    def get_limits(self):
        return [[0.0, 1.0]]


class FakeWorld:
    def __init__(self):
        self.robot = object()
        self.object = object()
        self.friction_calls = []

    def configure_contact_friction(self, *args, **kwargs):
        self.friction_calls.append((args, kwargs))


class FakeCapturer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.world = FakeWorld()
        self.closed = False
        self.instances.append(self)

    def _get_world(self, urdf, size):
        self.urdf = urdf
        self.size = size
        return self.world

    def close(self):
        self.closed = True


def test_independent_world_constructs_configures_and_closes_per_arm(monkeypatch):
    FakeCapturer.instances = []
    drive_calls = []
    monkeypatch.setattr(live, "ViewPcdCapturer", FakeCapturer)
    monkeypatch.setattr(live, "resolve_urdf", lambda *args, **kwargs: "object.urdf")
    monkeypatch.setattr(
        live,
        "set_joint_drive_properties",
        lambda robot, **kwargs: drive_calls.append((robot, kwargs)),
    )
    monkeypatch.setattr(live, "find_target_joint", lambda obj, link: FakeJoint())
    monkeypatch.setattr(live, "target_joint_index", lambda obj, joint: 2)
    init = {"object_urdf": "ignored", "size": 0.75}

    worlds = []
    for _ in range(2):
        with live.independent_world(init, "target_link") as (_, world, idx, span):
            worlds.append(world)
            assert idx == 2
            assert span == 1.0

    assert worlds[0] is not worlds[1]
    assert len(FakeCapturer.instances) == 2
    assert all(instance.closed for instance in FakeCapturer.instances)
    assert len(drive_calls) == 2
    assert all(len(world.friction_calls) == 1 for world in worlds)
