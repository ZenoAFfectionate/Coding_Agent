"""TestRunnerTool - Test Execution Tool

Provides test discovery and execution capabilities for a Coding Agent, including:
- Discover tests in a project (pytest / unittest)
- Run all tests or specific test files/functions
- Parse and summarize test results (pass/fail/error counts)
- Collect coverage information (when pytest-cov is available)

Safety features:
- Sandboxed to configured project path
- Timeout control (default 120s for test suites)
- Output size limits

Usage:
    ```python
    test_tool = TestRunnerTool(project_path="./my-project")

    # Discover tests
    result = test_tool.run({"action": "discover"})

    # Run all tests
    result = test_tool.run({"action": "run"})

    # Run specific test file
    result = test_tool.run({"action": "run", "target": "tests/test_utils.py"})

    # Run a single test function
    result = test_tool.run({"action": "run", "target": "tests/test_utils.py::test_add"})

    # Run with coverage
    result = test_tool.run({"action": "coverage"})
    ```
"""

import subprocess
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from ...utils.subprocess_utils import safe_run
from ..base import Tool, ToolParameter, tool_action


class TestRunnerTool(Tool):
    """Test Runner Tool

    Discovers and executes tests using pytest or unittest, parses results,
    and provides structured summaries for Coding Agents.

    Supports:
    - pytest (preferred, auto-detected)
    - unittest (fallback)
    - Coverage reporting via pytest-cov
    """

    def __init__(
        self,
        project_path: str = ".",
        timeout: int = 120,
        max_output_size: int = 5 * 1024 * 1024,  # 5 MB
        framework: str = "auto",
        expandable: bool = False,
    ):
        """Initialize TestRunnerTool.

        Args:
            project_path: Root path of the project to test.
            timeout: Maximum seconds for test execution.
            max_output_size: Maximum output size in bytes before truncation.
            framework: Test framework - 'auto', 'pytest', or 'unittest'.
            expandable: Whether to expand into sub-tools via @tool_action.
        """
        super().__init__(
            name="test_runner",
            description=(
                "Test runner tool - discover tests, run test suites or individual tests, "
                "and collect coverage reports (supports pytest and unittest)"
            ),
            expandable=expandable,
        )
        self.project_path = Path(project_path).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.framework = framework

    # ------------------------------------------------------------------ #
    #  Framework detection
    # ------------------------------------------------------------------ #

    def _detect_framework(self) -> str:
        """Detect which test framework is available.

        Returns:
            'pytest' or 'unittest'.
        """
        if self.framework != "auto":
            return self.framework

        # Check if pytest is installed
        try:
            result = safe_run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.project_path),
            )
            if result.returncode == 0:
                return "pytest"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return "unittest"

    # ------------------------------------------------------------------ #
    #  Command execution helper
    # ------------------------------------------------------------------ #

    def _run_cmd(self, args: List[str], timeout: Optional[int] = None) -> str:
        """Execute a command and return output.

        Args:
            args: Full command as a list of strings.
            timeout: Override default timeout.

        Returns:
            Combined stdout/stderr.
        """
        effective_timeout = timeout or self.timeout
        try:
            result = safe_run(
                args,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            if len(output) > self.max_output_size:
                output = output[: self.max_output_size]
                output += f"\n\n... output truncated (exceeded {self.max_output_size} bytes)"

            return output.strip() if output.strip() else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: Test execution timed out after {effective_timeout}s"
        except FileNotFoundError:
            return "Error: Python executable not found."
        except Exception as e:
            return f"Error running tests: {e}"

    # ------------------------------------------------------------------ #
    #  Result parsing
    # ------------------------------------------------------------------ #

    def _parse_pytest_summary(self, output: str) -> str:
        """Extract a structured summary from pytest output.

        Args:
            output: Raw pytest output.

        Returns:
            Formatted summary string.
        """
        lines = output.strip().split("\n")

        # Look for the summary line, e.g. "=== 5 passed, 2 failed in 1.23s ==="
        summary_line = ""
        for line in reversed(lines):
            if "passed" in line or "failed" in line or "error" in line:
                summary_line = line.strip()
                break

        # Look for FAILED test names
        failed_tests = []
        for line in lines:
            if line.startswith("FAILED "):
                failed_tests.append(line.strip())

        # Look for ERROR entries
        error_tests = []
        for line in lines:
            if line.startswith("ERROR "):
                error_tests.append(line.strip())

        # Build structured summary
        parts = ["=== Test Results Summary ==="]
        if summary_line:
            parts.append(f"Result: {summary_line}")

        if failed_tests:
            parts.append(f"\nFailed Tests ({len(failed_tests)}):")
            for ft in failed_tests[:20]:  # limit display
                parts.append(f"  - {ft}")

        if error_tests:
            parts.append(f"\nErrors ({len(error_tests)}):")
            for et in error_tests[:20]:
                parts.append(f"  - {et}")

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Action dispatch
    # ------------------------------------------------------------------ #

    def run(self, parameters: Dict[str, Any]) -> str:
        """Execute a test action."""
        action = parameters.get("action", "run")

        dispatch = {
            "discover": self._discover,
            "run": self._run_tests,
            "coverage": self._coverage,
        }

        handler = dispatch.get(action)
        if handler is None:
            supported = ", ".join(sorted(dispatch.keys()))
            return f"Unsupported action: '{action}'. Supported actions: {supported}"

        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        """Return tool parameter definitions."""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Test action: discover (find tests), run (execute tests), coverage (run with coverage)",
                required=True,
                enum=["discover", "run", "coverage"],
            ),
            ToolParameter(
                name="target",
                type="string",
                description=(
                    "Test target - file path, directory, or specific test. "
                    "Examples: 'tests/', 'tests/test_utils.py', 'tests/test_utils.py::test_add'. "
                    "Default: run all tests."
                ),
                required=False,
            ),
            ToolParameter(
                name="verbose",
                type="boolean",
                description="Enable verbose output (default: True).",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="keyword",
                type="string",
                description="Filter tests by keyword expression (pytest -k). Example: 'test_add or test_sub'.",
                required=False,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Individual actions
    # ------------------------------------------------------------------ #

    @tool_action("test_discover", "Discover and list available tests in the project")
    def _discover(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Discover tests.

        Args:
            parameters: Optional dict with 'target' key.

        Returns:
            List of discovered tests.
        """
        parameters = parameters or {}
        framework = self._detect_framework()

        if framework == "pytest":
            args = ["python", "-m", "pytest", "--collect-only", "-q"]
            target = parameters.get("target")
            if target:
                args.append(target)
            output = self._run_cmd(args, timeout=30)
            return f"=== Test Discovery (pytest) ===\n{output}"
        else:
            args = ["python", "-m", "unittest", "discover", "-v"]
            target = parameters.get("target")
            if target:
                args.extend(["-s", target])
            output = self._run_cmd(args, timeout=30)
            return f"=== Test Discovery (unittest) ===\n{output}"

    @tool_action("test_run", "Run tests and report results")
    def _run_tests(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Run tests.

        Args:
            parameters: Optional dict with 'target', 'verbose', 'keyword' keys.

        Returns:
            Test results with structured summary.
        """
        parameters = parameters or {}
        framework = self._detect_framework()

        if framework == "pytest":
            args = ["python", "-m", "pytest"]

            verbose = parameters.get("verbose", True)
            if verbose:
                args.append("-v")

            keyword = parameters.get("keyword")
            if keyword:
                args.extend(["-k", keyword])

            target = parameters.get("target")
            if target:
                args.append(target)

            output = self._run_cmd(args)
            summary = self._parse_pytest_summary(output)
            return f"{output}\n\n{summary}"
        else:
            args = ["python", "-m", "unittest"]
            target = parameters.get("target")
            if target:
                args.append(target)
            else:
                args.extend(["discover", "-v"])

            verbose = parameters.get("verbose", True)
            if verbose and "discover" in args:
                args.append("-v")

            output = self._run_cmd(args)
            return f"=== Test Results (unittest) ===\n{output}"

    @tool_action("test_coverage", "Run tests with coverage analysis")
    def _coverage(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Run tests with coverage.

        Args:
            parameters: Optional dict with 'target' key.

        Returns:
            Test results with coverage report.
        """
        parameters = parameters or {}
        framework = self._detect_framework()

        if framework == "pytest":
            # Try pytest-cov first
            args = [
                "python", "-m", "pytest",
                "--cov=.",
                "--cov-report=term-missing",
                "-v",
            ]

            target = parameters.get("target")
            if target:
                args.append(target)

            output = self._run_cmd(args)

            # Check if pytest-cov is not installed
            if "no module named" in output.lower() or "unrecognized arguments: --cov" in output.lower():
                return (
                    "Coverage plugin not available. Install it with:\n"
                    "  pip install pytest-cov\n\n"
                    "Falling back to regular test run...\n\n"
                    + self._run_tests(parameters)
                )

            summary = self._parse_pytest_summary(output)
            return f"{output}\n\n{summary}"
        else:
            # Use coverage.py with unittest
            args = ["python", "-m", "coverage", "run", "-m", "unittest", "discover"]
            output = self._run_cmd(args)

            # Generate report
            report = self._run_cmd(["python", "-m", "coverage", "report", "-m"])

            if "no module named" in output.lower():
                return (
                    "coverage module not available. Install it with:\n"
                    "  pip install coverage\n\n"
                    "Falling back to regular test run...\n\n"
                    + self._run_tests(parameters)
                )

            return f"=== Test Run ===\n{output}\n\n=== Coverage Report ===\n{report}"
