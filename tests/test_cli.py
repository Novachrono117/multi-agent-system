"""The command line, exercised through ``main()`` with no network and no model.

The single most important test here is
``test_the_readme_demo_command_works``: the README promises one command, and a
README that promises a command which does not run is the exact failure this
repository was built to stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobfit.cli import main

POSTING = """Backend Engineer (Python) - Remote

What we are looking for
- 2+ years of Python in production
- Comfortable with PostgreSQL and writing real SQL
- Experience with Docker and GitHub Actions
- Testing discipline with pytest

Nice to have
- Kubernetes for our staging cluster
- Terraform and infrastructure as code
"""


@pytest.fixture
def posting_file(tmp_path: Path) -> Path:
    path = tmp_path / "posting.txt"
    path.write_text(POSTING, encoding="utf-8")
    return path


class TestRunOffline:
    def test_the_readme_demo_command_works(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The one command the README promises, run exactly as written."""
        code = main(["run", "--offline", "--posting", "examples/posting_sample.txt"])
        assert code == 0
        assert "Draft for human review" in capsys.readouterr().out

    def test_inline_text_is_accepted(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["run", "--offline", "--text", POSTING]) == 0
        assert "coverage" in capsys.readouterr().out.casefold()

    def test_a_posting_file_is_accepted(
        self, posting_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["run", "--offline", "--posting", str(posting_file)]) == 0
        assert capsys.readouterr().out.strip()

    def test_the_brief_can_be_written_to_a_file(self, posting_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "brief.md"
        assert main(["run", "--offline", "--posting", str(posting_file), "--out", str(out)]) == 0
        assert out.exists()
        assert "Draft for human review" in out.read_text(encoding="utf-8")

    def test_json_output_is_the_full_run_state(
        self, posting_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["run", "--offline", "--posting", str(posting_file), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["run_id"]
        assert payload["outcome"]["status"] == "completed"
        assert payload["trace"]

    def test_the_trace_is_written_as_json_lines(self, posting_file: Path, tmp_path: Path) -> None:
        trace = tmp_path / "trace.jsonl"
        main(["run", "--offline", "--posting", str(posting_file), "--trace", str(trace)])
        lines = trace.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 5
        events = [json.loads(line)["event"] for line in lines]
        assert "run_started" in events
        assert "run_finished" in events

    def test_every_tool_call_in_the_offline_run_succeeds(
        self, posting_file: Path, tmp_path: Path
    ) -> None:
        """A regression guard for a bug the trace caught and the output hid.

        The offline stub used to extract the posting id by finding the first
        token containing a colon, which matched the word "candidate:" in the
        prompt. Both tool calls then failed against a nonexistent posting - and
        the printed brief still looked correct, because the score is recomputed
        in the node.
        """
        trace = tmp_path / "trace.jsonl"
        main(["run", "--offline", "--posting", str(posting_file), "--trace", str(trace)])
        tool_events = [
            json.loads(line)
            for line in trace.read_text(encoding="utf-8").strip().splitlines()
            if json.loads(line)["event"] == "tool_call"
        ]
        assert tool_events
        assert all(event["ok"] for event in tool_events)

    def test_offline_output_is_never_labelled_as_model_output(
        self, posting_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No model ran, so nothing may claim one did.

        The first offline demo printed "Output source: llm". That is the kind of
        false label this whole project exists to remove, so it is now asserted.
        """
        main(["run", "--offline", "--posting", str(posting_file)])
        out = capsys.readouterr().out
        assert "Output source: deterministic" in out
        assert "Output source: llm" not in out

    def test_the_deterministic_coverage_is_reported_as_measured(
        self, posting_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["run", "--offline", "--posting", str(posting_file)])
        out = capsys.readouterr().out
        assert "measured deterministically" in out

    def test_never_claim_items_reach_the_output(
        self, posting_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The posting asks for Kubernetes; the example profile forbids claiming it."""
        main(["run", "--offline", "--posting", str(posting_file)])
        out = capsys.readouterr().out
        assert "Do NOT claim these in an interview" in out
        assert "Kubernetes" in out

    def test_the_fictional_profile_warns_on_stderr(
        self, posting_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Assessing against a made-up candidate must never happen silently."""
        main(["run", "--offline", "--posting", str(posting_file)])
        assert "fictional example profile" in capsys.readouterr().err


class TestRunErrors:
    def test_a_missing_profile_explains_how_to_fix_it(
        self, posting_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "run",
                "--offline",
                "--posting",
                str(posting_file),
                "--profile",
                str(tmp_path / "nope.toml"),
            ]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "no profile found" in err
        assert "profile.example.toml" in err

    def test_a_source_is_required(self) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--offline"])

    def test_sources_are_mutually_exclusive(self, posting_file: Path) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--offline", "--posting", str(posting_file), "--text", "x"])

    def test_an_unknown_board_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--offline", "--search", "python", "--sources", "linkedin"])


class TestGraphCommand:
    def test_it_prints_a_mermaid_diagram_of_the_real_graph(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The README diagram comes from here, so it cannot drift from the code."""
        assert main(["graph"]) == 0
        out = capsys.readouterr().out
        assert "graph TD" in out
        for node in ("intake", "supervisor", "fit_screener", "brief_writer", "finalize"):
            assert node in out


class TestDoctorCommand:
    def test_it_reports_the_configuration(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "transport" in out
        assert "max tool steps" in out

    def test_it_says_the_anthropic_key_is_absent(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The first question a reader has is why it did not talk to a model."""
        main(["doctor"])
        assert "NOT set" in capsys.readouterr().out

    def test_it_lists_the_allowed_hosts_and_omits_the_forbidden_ones(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["doctor"])
        out = capsys.readouterr().out
        assert "remotive.com" in out
        assert "linkedin" not in out.casefold()
        assert "indeed" not in out.casefold()

    def test_it_names_the_free_alternative_when_ollama_is_selected(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jobfit.config import get_settings

        monkeypatch.setenv("JOBFIT_TRANSPORT", "ollama")
        get_settings.cache_clear()
        main(["doctor"])
        out = capsys.readouterr().out
        assert "ollama pull" in out
        assert "--offline" in out
