"""Lexical file identity for receipts carrying Windows or POSIX paths.

Explicit drive/UNC/backslash paths use Windows semantics. POSIX paths retain
case, including relative paths on POSIX hosts. This does not grant access or
replace the executor's resolution, scope check, and fresh byte verification.
"""
from __future__ import annotations

import ntpath
import os
import posixpath
import re


def windows_path(value: str) -> bool:
    if re.match(r"^[A-Za-z]:", value) or value.startswith(("\\", "//")):
        return True
    if value.startswith("/"):
        return False
    return "\\" in value or os.name == "nt"


def absolute_file_path(value: str) -> bool:
    return (ntpath if windows_path(value) else posixpath).isabs(value)


def normalize_file_path(value: str) -> str:
    if windows_path(value):
        return ntpath.normpath(value.replace("/", "\\")).casefold()
    return posixpath.normpath(value)


def file_basename(value: str) -> str:
    return (ntpath if windows_path(value) else posixpath).basename(value)


def file_targets_match(expected: str, actual: str) -> bool:
    if not expected or not actual:
        return False
    expected_path = normalize_file_path(expected)
    actual_path = normalize_file_path(actual)
    if expected_path == actual_path:
        return True
    # A basename is a selector; explicit directories never disappear.
    if any(c in expected for c in "/\\") or ntpath.splitdrive(expected)[0]:
        return False
    name = file_basename(actual_path)
    return expected.casefold() == name.casefold() if windows_path(actual_path) else expected == name


def file_selector_matches(selector, path: str) -> bool:
    name = file_basename(path)
    value = selector.get("value")
    if not isinstance(value, str) or not value or any(c in value for c in '/\\:*?"<>|'):
        return False
    fold = str.casefold if windows_path(path) else str
    suffixes = selector.get("suffixes")
    matched = []
    if "suffixes" in selector:
        if not isinstance(suffixes, list) or not suffixes or any(not isinstance(s, str) or not s for s in suffixes):
            return False
        # Suffixes filter a file type, while the selected name retains case.
        matched = [s for s in suffixes if name.casefold().endswith(s.casefold())]
        if not matched:
            return False
    if selector.get("kind") == "file_name":
        return fold(name) == fold(value)
    if selector.get("kind") == "file_stem":
        if matched:
            return any(fold(name[:-len(s)]) == fold(value) for s in matched)
        return fold(posixpath.splitext(name)[0]) == fold(value)
    return False
