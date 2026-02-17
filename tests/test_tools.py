"""Tests for built-in tools: FileTool, CodeExecutionTool, CodeSearchTool, GitTool, TestRunnerTool."""

import sys
import os
import types
import importlib
import importlib.util
import tempfile
import shutil
import unittest
from pathlib import Path

# ------------------------------------------------------------------ #
#  Bootstrap minimal package hierarchy for relative imports
#  This avoids triggering code/__init__.py (which pulls in optional
#  deps like hello_agents that may not be installed).
# ------------------------------------------------------------------ #

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_code_dir = os.path.join(_project, "code")
_tools_dir = os.path.join(_code_dir, "tools")
_builtin_dir = os.path.join(_tools_dir, "builtin")

# Ensure project root is on sys.path
if _project not in sys.path:
    sys.path.insert(0, _project)


def _ensure_pkg(name, path):
    """Register a stub package in sys.modules if not already present."""
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        pkg.__package__ = name
        sys.modules[name] = pkg


def _load_module(fqn, filepath):
    """Load a single module by fully-qualified name and file path."""
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


# Set up stub packages
_ensure_pkg("code", _code_dir)
_ensure_pkg("code.tools", _tools_dir)
_ensure_pkg("code.tools.builtin", _builtin_dir)

# Load the base module (needed by all tools via relative import)
_load_module("code.tools.base", os.path.join(_tools_dir, "base.py"))

# Load each tool module
_file_mod = _load_module("code.tools.builtin.file_tool", os.path.join(_builtin_dir, "file_tool.py"))
_exec_mod = _load_module("code.tools.builtin.code_execution_tool", os.path.join(_builtin_dir, "code_execution_tool.py"))
_search_mod = _load_module("code.tools.builtin.code_search_tool", os.path.join(_builtin_dir, "code_search_tool.py"))
_git_mod = _load_module("code.tools.builtin.git_tool", os.path.join(_builtin_dir, "git_tool.py"))
_test_mod = _load_module("code.tools.builtin.test_runner_tool", os.path.join(_builtin_dir, "test_runner_tool.py"))

FileTool = _file_mod.FileTool
CodeExecutionTool = _exec_mod.CodeExecutionTool
CodeSearchTool = _search_mod.CodeSearchTool
GitTool = _git_mod.GitTool
TestRunnerTool = _test_mod.TestRunnerTool


class FileToolTest(unittest.TestCase):
    """Test FileTool operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = FileTool(workspace=self.tmpdir)
        # Create a sample file
        (Path(self.tmpdir) / "hello.txt").write_text("line1\nline2\nline3\nline4\nline5\n")
        (Path(self.tmpdir) / "subdir").mkdir()
        (Path(self.tmpdir) / "subdir" / "nested.py").write_text("print('hi')\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_file(self):
        result = self.tool.run({"action": "read", "path": "hello.txt"})
        self.assertIn("line1", result)
        self.assertIn("line5", result)
        self.assertIn("lines 1-5", result)

    def test_read_with_range(self):
        result = self.tool.run({"action": "read", "path": "hello.txt", "start_line": 2, "end_line": 3})
        self.assertIn("line2", result)
        self.assertIn("line3", result)
        self.assertNotIn("line1", result)

    def test_read_nonexistent(self):
        result = self.tool.run({"action": "read", "path": "nope.txt"})
        self.assertIn("Error", result)

    def test_write_file(self):
        result = self.tool.run({"action": "write", "path": "new.txt", "content": "hello world"})
        self.assertIn("Successfully", result)
        self.assertEqual((Path(self.tmpdir) / "new.txt").read_text(), "hello world")

    def test_write_creates_dirs(self):
        result = self.tool.run({"action": "write", "path": "a/b/c.txt", "content": "deep"})
        self.assertIn("Successfully", result)
        self.assertTrue((Path(self.tmpdir) / "a" / "b" / "c.txt").exists())

    def test_edit_file(self):
        result = self.tool.run({
            "action": "edit", "path": "hello.txt",
            "old_string": "line3", "new_string": "LINE_THREE"
        })
        self.assertIn("Successfully", result)
        content = (Path(self.tmpdir) / "hello.txt").read_text()
        self.assertIn("LINE_THREE", content)
        self.assertNotIn("line3", content)

    def test_edit_not_found(self):
        result = self.tool.run({
            "action": "edit", "path": "hello.txt",
            "old_string": "nonexistent_string", "new_string": "xxx"
        })
        self.assertIn("not found", result)

    def test_list_dir(self):
        result = self.tool.run({"action": "list_dir", "path": "."})
        self.assertIn("hello.txt", result)
        self.assertIn("subdir", result)

    def test_file_info(self):
        result = self.tool.run({"action": "file_info", "path": "hello.txt"})
        self.assertIn("line_count", result)

    def test_path_escape_blocked(self):
        result = self.tool.run({"action": "read", "path": "../../etc/passwd"})
        self.assertIn("Error", result)

    def test_get_parameters(self):
        params = self.tool.get_parameters()
        names = [p.name for p in params]
        self.assertIn("action", names)
        self.assertIn("path", names)


class CodeExecutionToolTest(unittest.TestCase):
    """Test CodeExecutionTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = CodeExecutionTool(workspace=self.tmpdir, timeout=10)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_python_hello(self):
        result = self.tool.run({"action": "python", "code": "print('hello from sandbox')"})
        self.assertIn("hello from sandbox", result)
        self.assertIn("Exit code: 0", result)

    def test_python_error(self):
        result = self.tool.run({"action": "python", "code": "raise ValueError('oops')"})
        self.assertIn("oops", result)
        self.assertNotIn("Exit code: 0", result)

    def test_python_timeout(self):
        result = self.tool.run({"action": "python", "code": "import time; time.sleep(20)", "timeout": 2})
        self.assertIn("timed out", result)

    def test_shell_echo(self):
        result = self.tool.run({"action": "shell", "code": "echo 'shell works'"})
        self.assertIn("shell works", result)

    def test_empty_code(self):
        result = self.tool.run({"action": "python", "code": ""})
        self.assertIn("Error", result)

    def test_get_parameters(self):
        params = self.tool.get_parameters()
        names = [p.name for p in params]
        self.assertIn("action", names)
        self.assertIn("code", names)


class CodeSearchToolTest(unittest.TestCase):
    """Test CodeSearchTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = CodeSearchTool(workspace=self.tmpdir)
        # Create sample files
        (Path(self.tmpdir) / "main.py").write_text("def hello():\n    print('hello world')\n\nhello()\n")
        (Path(self.tmpdir) / "utils.py").write_text("def add(a, b):\n    return a + b\n")
        sub = Path(self.tmpdir) / "src"
        sub.mkdir()
        (sub / "app.js").write_text("console.log('hello');\nfunction greet() { return 'hi'; }\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_pattern(self):
        result = self.tool.run({"action": "search", "pattern": "hello"})
        self.assertIn("hello", result)
        # Should find matches in multiple files
        self.assertNotIn("No matches", result)

    def test_search_with_file_filter(self):
        result = self.tool.run({"action": "search", "pattern": "hello", "file_pattern": "*.py"})
        self.assertIn("hello", result)

    def test_search_no_match(self):
        result = self.tool.run({"action": "search", "pattern": "zzzznonexistent"})
        self.assertIn("No matches", result)

    def test_find_files(self):
        result = self.tool.run({"action": "find_files", "pattern": "*.py"})
        self.assertIn("main.py", result)
        self.assertIn("utils.py", result)

    def test_find_files_js(self):
        result = self.tool.run({"action": "find_files", "pattern": "*.js"})
        self.assertIn("app.js", result)

    def test_get_parameters(self):
        params = self.tool.get_parameters()
        names = [p.name for p in params]
        self.assertIn("pattern", names)
        self.assertIn("file_pattern", names)


class GitToolTest(unittest.TestCase):
    """Test GitTool (on the actual CodingAgent repo)."""

    def setUp(self):
        # Use the CodingAgent repo itself for testing
        self.repo_path = os.path.join(os.path.dirname(__file__), "..")
        self.tool = GitTool(repo_path=self.repo_path)

    def test_status(self):
        result = self.tool.run({"action": "status"})
        self.assertIn("branch", result.lower())

    def test_log(self):
        result = self.tool.run({"action": "log", "limit": 3})
        # Should contain at least one commit hash
        self.assertNotIn("Error", result)

    def test_branch_list(self):
        result = self.tool.run({"action": "branch", "sub_action": "list"})
        self.assertNotIn("Error", result)

    def test_destructive_blocked(self):
        tool = GitTool(repo_path=self.repo_path, allow_destructive=False)
        result = tool._run_git(["push", "--force", "origin", "main"])
        self.assertIn("Blocked", result)

    def test_get_parameters(self):
        params = self.tool.get_parameters()
        names = [p.name for p in params]
        self.assertIn("action", names)
        self.assertIn("message", names)


class TestRunnerToolTest(unittest.TestCase):
    """Test TestRunnerTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tool = TestRunnerTool(project_path=self.tmpdir, timeout=30)
        # Create a simple test file
        (Path(self.tmpdir) / "test_sample.py").write_text(
            "import unittest\n\n"
            "class TestSample(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(1 + 1, 2)\n"
            "    def test_sub(self):\n"
            "        self.assertEqual(3 - 1, 2)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_discover(self):
        result = self.tool.run({"action": "discover"})
        self.assertIn("Test Discovery", result)

    def test_run_tests(self):
        result = self.tool.run({"action": "run"})
        # Should show that tests ran (either pytest or unittest output)
        lower = result.lower()
        self.assertTrue("passed" in lower or "ok" in lower or "test" in lower)

    def test_get_parameters(self):
        params = self.tool.get_parameters()
        names = [p.name for p in params]
        self.assertIn("action", names)
        self.assertIn("target", names)


if __name__ == "__main__":
    unittest.main()
