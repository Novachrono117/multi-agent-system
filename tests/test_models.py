"""Domain model invariants - the ones that encode project rules."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jobfit.models.assessment import FitAssessment, FitVerdict, MatchScore
from jobfit.models.brief import DISCLAIMER, RoleBrief
from jobfit.models.posting import JobPosting, PostingSource, dedup_postings
from jobfit.models.profile import load_profile
from jobfit.models.trace import OutputSource, TokenUsage, sum_usage

EXAMPLE_PROFILE = Path("configs/profile.example.toml")

API_SOURCES = [
    PostingSource.REMOTIVE,
    PostingSource.ARBEITNOW,
    PostingSource.GREENHOUSE,
]


class TestPostingProvenance:
    """Attribution is structural: an API posting cannot exist without its url."""

    @pytest.mark.parametrize("source", API_SOURCES)
    def test_api_sourced_posting_requires_a_url(self, source: PostingSource) -> None:
        with pytest.raises(ValidationError, match="url it came from"):
            JobPosting(source=source, external_id="1", title="Engineer")

    def test_pasted_posting_needs_no_url(self) -> None:
        posting = JobPosting(source=PostingSource.PASTED, external_id="local", title="Engineer")
        assert posting.attribution() is None

    def test_attribution_line_carries_the_link_back(self) -> None:
        posting = JobPosting(
            source=PostingSource.REMOTIVE,
            external_id="2091045",
            title="Engineer",
            url="https://remotive.com/remote-jobs/x-2091045",
        )
        credit = posting.attribution()
        assert credit is not None
        assert "remotive" in credit
        assert posting.url in credit

    def test_posting_id_is_unique_across_sources(self) -> None:
        a = JobPosting(source=PostingSource.PASTED, external_id="1", title="A")
        b = JobPosting(source=PostingSource.REMOTIVE, external_id="1", title="B", url="u")
        assert a.posting_id != b.posting_id


class TestDedupReducer:
    def test_same_posting_from_two_batches_collapses(self) -> None:
        a = JobPosting(source=PostingSource.REMOTIVE, external_id="1", title="A", url="u")
        b = JobPosting(source=PostingSource.REMOTIVE, external_id="1", title="A", url="u")
        assert len(dedup_postings([a], [b])) == 1

    def test_later_posting_wins_because_it_may_carry_the_body(self) -> None:
        thin = JobPosting(source=PostingSource.GREENHOUSE, external_id="1", title="A", url="u")
        fat = thin.model_copy(update={"body_text": "the full description"})
        assert dedup_postings([thin], [fat])[0].body_text == "the full description"

    def test_distinct_postings_are_both_kept_in_order(self) -> None:
        a = JobPosting(source=PostingSource.REMOTIVE, external_id="1", title="A", url="u")
        b = JobPosting(source=PostingSource.REMOTIVE, external_id="2", title="B", url="u")
        assert [p.external_id for p in dedup_postings([a], [b])] == ["1", "2"]

    def test_handles_empty_and_none_sides(self) -> None:
        assert dedup_postings(None, None) == []


class TestUsageAccounting:
    def test_usage_adds(self) -> None:
        left = TokenUsage(input_tokens=10, output_tokens=2)
        right = TokenUsage(input_tokens=5, output_tokens=3, cache_read_tokens=7)
        total = left + right
        assert total.input_tokens == 15
        assert total.output_tokens == 5
        assert total.cache_read_tokens == 7
        assert total.total == 20

    def test_reducer_tolerates_missing_sides(self) -> None:
        assert sum_usage(None, None).total == 0
        assert sum_usage(None, TokenUsage(input_tokens=4)).input_tokens == 4

    def test_cost_uses_supplied_rates_not_hardcoded_ones(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cost_usd(2.0, 10.0) == pytest.approx(12.0)


class TestBriefRules:
    def test_disclaimer_cannot_be_overridden(self) -> None:
        with pytest.raises(ValidationError):
            RoleBrief(posting_id="x:1", role_summary="s", disclaimer="Auto-applied for you")

    def test_disclaimer_is_present_by_default(self) -> None:
        assert RoleBrief(posting_id="x:1", role_summary="s").disclaimer == DISCLAIMER

    def test_brief_defaults_to_declaring_its_source(self) -> None:
        assert RoleBrief(posting_id="x:1", role_summary="s").source == OutputSource.LLM


class TestAssessmentGating:
    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            (FitVerdict.STRONG_FIT, True),
            (FitVerdict.WORTH_APPLYING, True),
            (FitVerdict.STRETCH, False),
            (FitVerdict.NO_FIT, False),
        ],
    )
    def test_only_promising_postings_earn_the_writer(
        self, verdict: FitVerdict, expected: bool
    ) -> None:
        assert FitAssessment(posting_id="x:1", verdict=verdict).worth_a_brief is expected

    def test_coverage_outside_zero_to_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MatchScore(coverage=1.5)

    def test_deal_breaker_marks_the_score_blocked(self) -> None:
        assert MatchScore(coverage=0.9, deal_breakers_hit=["senior title"]).blocked is True


class TestProfile:
    def test_example_profile_loads_and_is_usable(self) -> None:
        profile = load_profile(EXAMPLE_PROFILE)
        assert profile.skills
        assert "python" in profile.skill_set()

    def test_example_profile_declares_what_must_never_be_claimed(self) -> None:
        assert load_profile(EXAMPLE_PROFILE).never_claim

    def test_profile_is_immutable_once_loaded(self) -> None:
        profile = load_profile(EXAMPLE_PROFILE)
        with pytest.raises(ValidationError):
            profile.headline = "Senior Staff Everything Engineer"

    def test_missing_profile_fails_loudly(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_profile(Path("configs/does-not-exist.toml"))

    def test_profile_requires_at_least_one_skill(self, tmp_path: Path) -> None:
        bad = tmp_path / "p.toml"
        lines = [
            "[profile]",
            'headline = "h"',
            'location = "l"',
            'seniority = "mid"',
            "years_experience = 1",
            "skills = []",
        ]
        bad.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_profile(bad)
