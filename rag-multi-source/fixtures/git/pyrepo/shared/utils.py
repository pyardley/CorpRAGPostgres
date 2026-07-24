"""Shared helper used by several feature modules (fixture for the git
import-dependency graph — mirrors fixtures/sql/05_shared_procs.sql's
"one shared object fanned into several callers" shape)."""


def helper(value: int) -> int:
    return value * 2


def other_helper(value: int) -> int:
    return value + 1
