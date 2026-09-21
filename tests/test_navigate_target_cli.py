import json
from unittest.mock import patch
import pytest
import navigate_target


@pytest.mark.parametrize(
    "option,value,field",
    [
        ("--class-name", "vehicle", "class_name"),
        ("--instance-id", "vehicle_01", "instance_id"),
    ],
)
def test_explicit_cli_target_overrides_null_file(tmp_path, option, value, field):
    session = tmp_path / "session.json"
    session.write_text("{}")
    target = tmp_path / "target.json"
    target.write_text('{"target": null}')
    responses = [
        {"job_id": "job"},
        {"job": {"id": "job", "state": "succeeded", "result": {"success": True}}},
    ]
    with patch(
        "sys.argv",
        [
            "navigate_target.py",
            "--session",
            str(session),
            "--target",
            str(target),
            option,
            value,
        ],
    ), patch.object(navigate_target, "send_command", side_effect=responses) as send:
        navigate_target.main()
    assert getattr(send.call_args_list[0].args[2], field) == value
    assert json.loads(target.read_text()) == {"target": None}


def test_null_target_explains_how_to_select_and_never_submits(tmp_path, capsys):
    session = tmp_path / "session.json"
    session.write_text("{}")
    target = tmp_path / "target.json"
    target.write_text('{"target": null}')
    with patch(
        "sys.argv",
        ["navigate_target.py", "--session", str(session), "--target", str(target)],
    ), patch.object(navigate_target, "send_command") as send:
        with pytest.raises(SystemExit) as exc:
            navigate_target.main()
    assert exc.value.code == 2
    assert "--class-name vehicle" in capsys.readouterr().err
    send.assert_not_called()
