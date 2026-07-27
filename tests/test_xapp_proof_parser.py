"""Witness-3 rigor for scripts/xapp_e2e_proof.py.

The mediator's receive log embeds the xApp's A1_POLICY_RESP JSON *escaped
inside* its own JSON log lines. The parser must (a) properly unescape the
embedded payload at any depth and (b) require BOTH the policy_instance_id
AND ``handler_id == "hw-python"`` in the SAME payload object — mere
co-occurrence of the two markers anywhere in the file must NOT count
(the old weak fallback did exactly that and is gone).
"""

from __future__ import annotations

import json

from scripts.xapp_e2e_proof import _embedded_json_objects, _line_acks_policy

PID = "8f7a1b2c-0000-4000-8000-123456789abc"


def _resp_payload(pid: str = PID, handler: str = "hw-python") -> dict:
    return {
        "policy_type_id": 20001,
        "policy_instance_id": pid,
        "handler_id": handler,
        "status": "OK",
    }


def _mediator_line(payload: dict) -> str:
    """A realistic Go-mediator log line: the RMR payload JSON is embedded
    (escaped) inside the ``msg`` field of the mediator's own JSON line."""
    return json.dumps(
        {
            "ts": 1753500000000,
            "crit": "INFO",
            "id": "a1mediator",
            "msg": "Message received: type=20011 payload=" + json.dumps(payload),
        }
    )


def test_single_escaped_payload_acks():
    line = _mediator_line(_resp_payload())
    assert _line_acks_policy(line, PID) is True


def test_double_escaped_payload_acks():
    # Some log shippers wrap the mediator line in another JSON envelope —
    # one more escaping level. The parser must still find the payload.
    outer = json.dumps({"log": _mediator_line(_resp_payload()), "stream": "stdout"})
    assert _line_acks_policy(outer, PID) is True


def test_wrong_handler_id_does_not_ack():
    line = _mediator_line(_resp_payload(handler="rogue-xapp"))
    assert _line_acks_policy(line, PID) is False


def test_wrong_policy_id_does_not_ack():
    line = _mediator_line(_resp_payload(pid="other-policy-id"))
    assert _line_acks_policy(line, PID) is False


def test_markers_in_different_payloads_do_not_ack():
    # The old weak fallback accepted `pid in log and "hw-python" in log`
    # file-wide. Both markers present in the SAME LINE but in DIFFERENT
    # payload objects must still be rejected.
    line = json.dumps(
        {
            "msg": "payload=" + json.dumps({"policy_instance_id": PID, "handler_id": "other"}),
            "msg2": "payload=" + json.dumps(
                {"policy_instance_id": "different", "handler_id": "hw-python"}
            ),
        }
    )
    assert _line_acks_policy(line, PID) is False


def test_plain_text_co_occurrence_does_not_ack():
    # Non-JSON line mentioning both markers — no embedded payload → no ack.
    line = f"note: {PID} routed via hw-python endpoint"
    assert _line_acks_policy(line, PID) is False


def test_nested_policy_id_in_same_payload_acks():
    payload = {
        "handler_id": "hw-python",
        "ack": {"policy_instance_id": PID},  # nested variant, same payload
    }
    assert _line_acks_policy(_mediator_line(payload), PID) is True


def test_embedded_object_extraction_depth():
    objs = _embedded_json_objects(_mediator_line(_resp_payload()))
    assert any(o.get("handler_id") == "hw-python" for o in objs)
    # The outer log-line object is found too.
    assert any("msg" in o for o in objs)


def test_non_json_line_yields_no_objects():
    assert _embedded_json_objects("plain text, no braces") == []
    assert _embedded_json_objects("{ broken json }") == []
