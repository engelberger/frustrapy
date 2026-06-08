"""Boundary test for the tmol parameter license-tier contract (mission #36).

This guards the contract authored in ``docs/tmol/param_inventory.json`` (the machine-checkable
form of ``docs/tmol/PARAM_SOURCING.md``). It is a pure-JSON contract check: it does NOT import
frustrapy and does NOT run any energy code, so it stays in the fast lane.

It locks in two invariants from the approved gate (docs/tmol/TMOL_LANE_DECISION.md):

  (a) every parameter group is tagged with a tier (academic + permissive) and a sourcing decision;
  (b) the permissive (redistributable) build profile excludes every LOADER / NO-PUBLIC-ORIGIN
      parameter file. It does NOT forbid reuse in the academic profile.
"""

import json
import os

import pytest

_VALID_TIERS = {"REUSE", "LOADER", "REDERIVE"}

_INVENTORY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "tmol", "param_inventory.json"
)


@pytest.fixture(scope="module")
def inventory():
    with open(os.path.abspath(_INVENTORY_PATH), "r") as handle:
        return json.load(handle)


def test_inventory_present_and_nonempty(inventory):
    assert inventory["groups"], "the catalog must list at least one parameter group"
    assert "build_profiles" in inventory
    assert {"academic", "permissive"} <= set(inventory["build_profiles"])


def test_every_group_tagged_with_tier_and_decision(inventory):
    """(a) every group carries a valid tier per profile and a non-empty decision."""
    for group in inventory["groups"]:
        gid = group.get("id", "<unnamed>")
        tier = group.get("tier", {})
        assert tier.get("academic") in _VALID_TIERS, f"{gid}: bad/missing academic tier"
        assert tier.get("permissive") in _VALID_TIERS, f"{gid}: bad/missing permissive tier"
        assert group.get("decision"), f"{gid}: missing sourcing decision"
        assert group.get("file"), f"{gid}: missing file evidence"
        assert isinstance(group.get("public_origin"), bool), f"{gid}: public_origin must be bool"


def test_permissive_profile_excludes_loader_and_no_public_origin_files(inventory):
    """(b) the permissive build ships no LOADER / NO-PUBLIC-ORIGIN parameter file."""
    permissive = inventory["build_profiles"]["permissive"]
    assert permissive.get("includes_params") is False, (
        "the permissive/redistributable profile must vendor no parameter file"
    )
    excluded = set(permissive.get("excluded_param_files", []))

    # Every group that is LOADER in the permissive tier, or has no clean public origin,
    # must appear in the permissive profile's excluded file list.
    for group in inventory["groups"]:
        gid = group["id"]
        must_exclude = (
            group["tier"]["permissive"] == "LOADER" or group["public_origin"] is False
        )
        if must_exclude:
            assert group["file"] in excluded, (
                f"{gid}: LOADER/NO-PUBLIC-ORIGIN file {group['file']!r} must be excluded "
                f"from the permissive build"
            )


def test_academic_profile_reuse_not_forbidden(inventory):
    """The contract must NOT forbid reuse in the academic profile."""
    academic = inventory["build_profiles"]["academic"]
    assert academic.get("includes_params") is True, (
        "the academic profile reuses the bundled tmol params directly"
    )
    academic_files = set(academic.get("param_files", []))
    # Every group's file is reusable in the academic profile (covered by the bundled YAMLs).
    for group in inventory["groups"]:
        assert group["file"] in academic_files, (
            f"{group['id']}: file {group['file']!r} must be reusable in the academic profile"
        )


def test_permissive_excluded_files_are_not_in_a_vendored_set(inventory):
    """No excluded param file may also be listed as vendored in the permissive build."""
    permissive = inventory["build_profiles"]["permissive"]
    excluded = set(permissive.get("excluded_param_files", []))
    vendored = set(permissive.get("param_files", []))  # absent by design -> empty
    assert not (excluded & vendored), "an excluded param file must never be vendored"
