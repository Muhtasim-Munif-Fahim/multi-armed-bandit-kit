"""Tests for the 'best' CLI subcommand."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from bandit_kit.cli import main


def test_cli_best_text_output() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["best", "--arms", "bern:0.1,bern:0.2,bern:0.05"])
    assert rc == 0
    text = buf.getvalue()
    assert "oracle arm: bern:0.2" in text
    assert "EV=0.2000" in text


def test_cli_best_json_output() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["best", "--arms", "bern:0.1,bern:0.4", "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["oracle_arm"] == "bern:0.4"
    assert payload["oracle_payoff"] == 0.4


def test_cli_best_rejects_empty_arm_list(capsys) -> None:
    rc = main(["best", "--arms", ""])
    assert rc == 2
    assert "at least one arm spec" in capsys.readouterr().err


def test_cli_best_supports_gaussian_arms() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["best", "--arms", "gauss:1.0:0.5,gauss:2.0:0.25"])
    assert rc == 0
    assert "gauss:2.0:0.25" in buf.getvalue()