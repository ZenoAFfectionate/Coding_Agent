"""ProfilerTool - Performance Profiling

Provides profiling capabilities for a Coding Agent:
- CPU profiling of Python files (top-N hotspots)
- Benchmarking code snippets with timeit
- Memory profiling snapshots with tracemalloc

All profiling uses Python stdlib (cProfile, timeit, tracemalloc) — no
external dependencies required.

Safety:
- All code execution happens in isolated subprocesses
- Paths are sandboxed to a configurable workspace root
- Timeout and output-size limits
"""

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.subprocess_utils import safe_run
from ..base import Tool, ToolParameter, tool_action


class ProfilerTool(Tool):
    """Performance profiling tool for Coding Agents.

    Supports profile (CPU), timeit (benchmarking), and memory
    (tracemalloc) actions — all using Python stdlib.
    """

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 60,
        max_output_size: int = 512 * 1024,  # 512 KB
        python_executable: str = "python3",
        expandable: bool = False,
    ):
        super().__init__(
            name="profiler",
            description=(
                "Profile Python code - CPU profiling (cProfile), benchmarking (timeit), "
                "and memory profiling (tracemalloc). All stdlib, no external deps."
            ),
            expandable=expandable,
        )
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.python_executable = python_executable
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Path safety
    # ------------------------------------------------------------------ #

    def _safe_path(self, rel_path: str) -> Optional[Path]:
        """Resolve *rel_path* inside the workspace, blocking escapes."""
        resolved = (self.workspace / rel_path).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            return None
        return resolved

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _truncate(self, text: str) -> str:
        """Truncate output if it exceeds the limit."""
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + f"\n... truncated ({len(text)} bytes total)"
        return text

    def _run_python_code(self, code: str, timeout: Optional[int] = None) -> str:
        """Write *code* to a temp file, run it in a subprocess, return combined output."""
        effective_timeout = timeout or self.timeout
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=str(self.workspace),
                delete=False, encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = safe_run(
                [self.python_executable, tmp_path],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            return self._truncate(output.strip()) if output.strip() else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {effective_timeout}s"
        except Exception as e:
            return f"Error: {e}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def run(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action", "profile")
        dispatch = {
            "profile": self._profile,
            "timeit": self._timeit,
            "memory": self._memory,
        }
        handler = dispatch.get(action)
        if handler is None:
            return f"Unsupported action '{action}'. Supported: {', '.join(dispatch)}"
        return handler(parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action", type="string",
                description="Action: profile (CPU), timeit (benchmark), memory (tracemalloc)",
                required=True,
            ),
            ToolParameter(
                name="path", type="string",
                description="Relative file path to profile (required for 'profile' action)",
                required=False,
            ),
            ToolParameter(
                name="code", type="string",
                description="Code snippet to benchmark or memory-profile (required for 'timeit' and 'memory')",
                required=False,
            ),
            ToolParameter(
                name="top_n", type="integer",
                description="Number of top results to display (default 15)",
                required=False, default=15,
            ),
            ToolParameter(
                name="number", type="integer",
                description="Number of iterations for timeit (default 1000)",
                required=False, default=1000,
            ),
            ToolParameter(
                name="repeat", type="integer",
                description="Number of repetitions for timeit (default 5)",
                required=False, default=5,
            ),
            ToolParameter(
                name="setup", type="string",
                description="Setup code for timeit (executed once before each repetition)",
                required=False,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

    @tool_action("profiler_profile", "CPU profile a Python file and show top hotspots")
    def _profile(self, parameters: Dict[str, Any]) -> str:
        """CPU-profile a Python file using cProfile.

        Args:
            parameters: Dict with path, top_n (optional).
        Returns:
            Profiling results with top-N hotspots.
        """
        rel_path = parameters.get("path", "")
        if not rel_path:
            return "Error: 'path' is required for profile action."

        path = self._safe_path(rel_path)
        if path is None:
            return "Error: path escapes workspace."
        if not path.exists():
            return f"Error: file not found: {rel_path}"
        if not path.is_file():
            return f"Error: not a file: {rel_path}"

        top_n = parameters.get("top_n", 15) or 15

        # Build a profiling wrapper script
        wrapper = textwrap.dedent(f"""\
            import cProfile
            import pstats
            import io

            profiler = cProfile.Profile()
            profiler.enable()

            # Execute the target file
            with open({str(path)!r}, 'r') as _f:
                _code = _f.read()
            exec(compile(_code, {str(path)!r}, 'exec'))

            profiler.disable()

            # Format results
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumulative')
            stats.print_stats({top_n})
            print(stream.getvalue())

            # Also show callers for context
            stream2 = io.StringIO()
            stats2 = pstats.Stats(profiler, stream=stream2)
            stats2.sort_stats('tottime')
            stats2.print_stats({top_n})
            print("--- Sorted by total time ---")
            print(stream2.getvalue())
        """)

        output = self._run_python_code(wrapper)
        return f"=== CPU Profile ({rel_path}, top {top_n}) ===\n{output}"

    @tool_action("profiler_timeit", "Benchmark a code snippet")
    def _timeit(self, parameters: Dict[str, Any]) -> str:
        """Benchmark a code snippet using timeit.

        Args:
            parameters: Dict with code, number (optional), repeat (optional), setup (optional).
        Returns:
            Timing results.
        """
        code = parameters.get("code", "")
        if not code.strip():
            return "Error: 'code' is required for timeit action."

        number = parameters.get("number", 1000) or 1000
        repeat = parameters.get("repeat", 5) or 5
        setup = parameters.get("setup", "pass") or "pass"

        # Build the timeit wrapper
        wrapper = textwrap.dedent(f"""\
            import timeit

            stmt = {code!r}
            setup_code = {setup!r}
            number = {number}
            repeat = {repeat}

            times = timeit.repeat(stmt=stmt, setup=setup_code, number=number, repeat=repeat)

            best = min(times)
            worst = max(times)
            avg = sum(times) / len(times)

            # Per-iteration stats
            best_per = best / number
            avg_per = avg / number

            print(f"=== Benchmark Results ===")
            print(f"Statement: {{stmt}}")
            if setup_code != "pass":
                print(f"Setup: {{setup_code}}")
            print(f"Iterations: {{number}} x {{repeat}} repetitions")
            print()
            print(f"Total time (best of {{repeat}}): {{best:.6f}}s")
            print(f"Total time (worst):            {{worst:.6f}}s")
            print(f"Total time (average):          {{avg:.6f}}s")
            print()

            # Format per-iteration time with appropriate unit
            def format_time(t):
                if t >= 1:
                    return f"{{t:.4f}} s"
                elif t >= 1e-3:
                    return f"{{t*1e3:.4f}} ms"
                elif t >= 1e-6:
                    return f"{{t*1e6:.4f}} μs"
                else:
                    return f"{{t*1e9:.4f}} ns"

            print(f"Per iteration (best):    {{format_time(best_per)}}")
            print(f"Per iteration (average): {{format_time(avg_per)}}")
        """)

        output = self._run_python_code(wrapper)
        return output

    @tool_action("profiler_memory", "Memory profiling snapshot using tracemalloc")
    def _memory(self, parameters: Dict[str, Any]) -> str:
        """Profile memory usage of a code snippet using tracemalloc.

        Args:
            parameters: Dict with code, top_n (optional).
        Returns:
            Memory profiling results.
        """
        code = parameters.get("code", "")
        if not code.strip():
            return "Error: 'code' is required for memory action."

        top_n = parameters.get("top_n", 15) or 15

        # Build the tracemalloc wrapper
        # The user code is embedded as a string and exec'd so tracemalloc
        # captures allocations from it.
        wrapper = textwrap.dedent(f"""\
            import tracemalloc
            import linecache

            tracemalloc.start()

            # Snapshot before
            snapshot_before = tracemalloc.take_snapshot()

            # Execute user code
            exec({code!r})

            # Snapshot after
            snapshot_after = tracemalloc.take_snapshot()

            # Compare snapshots
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')

            print("=== Memory Profile (tracemalloc) ===")
            print()

            # Overall stats
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {{current / 1024:.1f}} KB")
            print(f"Peak memory usage:    {{peak / 1024:.1f}} KB")
            print()

            # Top allocations
            top_stats = snapshot_after.statistics('lineno')
            print(f"Top {top_n} memory allocations:")
            print(f"{{'-' * 60}}")
            for i, stat in enumerate(top_stats[:{top_n}], 1):
                print(f"  #{{i}}: {{stat}}")

            # Also show top allocations by traceback
            print()
            print(f"Top {top_n} memory differences (new - old):")
            print(f"{{'-' * 60}}")
            for i, stat in enumerate(stats[:{top_n}], 1):
                print(f"  #{{i}}: {{stat}}")

            tracemalloc.stop()
        """)

        output = self._run_python_code(wrapper)
        return output
