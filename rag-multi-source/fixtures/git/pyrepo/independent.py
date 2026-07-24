"""Looks similar to feature_a/b/c but deliberately does NOT import
shared.utils — the distractor case (mirrors fixtures/sql/06_reports.sql's
usp_BuildReport_CustomerChurnRisk, which queries Orders directly instead
of going through the shared staging procedure)."""

import os


def helper(value: int) -> int:
    # Same name as shared.utils.helper, but a local, independent definition.
    return value * 3


def run(value: int) -> int:
    return helper(value) + len(os.getcwd())
