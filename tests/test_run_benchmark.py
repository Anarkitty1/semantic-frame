"""Tests for benchmark CLI entry point."""

from __future__ import annotations

import os
import subprocess
import sys


class TestBenchmarkCLI:
    """Tests for run_benchmark.py CLI."""

    def test_help_exits_zero(self) -> None:
        """Test --help flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.run_benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_mock_mode_runs(self) -> None:
        """Test --mock mode completes without API key."""
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.run_benchmark",
                "--mock",
                "--task",
                "statistical",
                "--quick",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        # Mock mode should complete successfully
        assert result.returncode == 0, f"Mock mode failed: {result.stderr}"

    def test_missing_api_key_error(self) -> None:
        """Test clear error when API key missing (non-mock mode)."""
        # Minimal environment without API key
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.run_benchmark", "--task", "statistical", "--quick"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode != 0
        output = (result.stdout + result.stderr).lower()
        assert "api" in output or "key" in output or "anthropic" in output

    def test_invalid_task_error(self) -> None:
        """Test error for invalid task name."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.run_benchmark",
                "--task",
                "nonexistent_task",
                "--mock",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should fail with invalid task
        assert result.returncode != 0 or "invalid" in result.stderr.lower()
