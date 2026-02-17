"""Tests for OutputParser - robust structured output parsing."""

import sys
import os
import importlib.util
import unittest

# Direct import to avoid triggering the full package __init__.py chain
_mod_path = os.path.join(os.path.dirname(__file__), "..", "code", "core", "output_parser.py")
_spec = importlib.util.spec_from_file_location("output_parser", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OutputParser = _mod.OutputParser
ParseError = _mod.ParseError


class TestParseJSON(unittest.TestCase):
    """Test JSON extraction from LLM output."""

    def test_direct_json(self):
        result = OutputParser.parse_json('{"name": "test", "value": 42}')
        self.assertEqual(result, {"name": "test", "value": 42})

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "val"}\n```\nDone.'
        result = OutputParser.parse_json(text)
        self.assertEqual(result, {"key": "val"})

    def test_json_embedded_in_text(self):
        text = 'The answer is {"x": 1, "y": 2} and more text.'
        result = OutputParser.parse_json(text)
        self.assertEqual(result, {"x": 1, "y": 2})

    def test_json_array(self):
        result = OutputParser.parse_json('[1, 2, 3]')
        self.assertEqual(result, [1, 2, 3])

    def test_repair_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        result = OutputParser.parse_json(text)
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_repair_python_booleans(self):
        text = '{"flag": True, "other": False, "empty": None}'
        result = OutputParser.parse_json(text)
        self.assertEqual(result, {"flag": True, "other": False, "empty": None})

    def test_repair_single_quotes(self):
        text = "{'key': 'value', 'num': 42}"
        result = OutputParser.parse_json(text)
        self.assertEqual(result, {"key": "value", "num": 42})

    def test_no_json_returns_none(self):
        result = OutputParser.parse_json("This is plain text with no JSON.")
        self.assertIsNone(result)

    def test_strict_raises(self):
        with self.assertRaises(ParseError):
            OutputParser.parse_json("no json here", strict=True)

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2]}, "ok": true}'
        result = OutputParser.parse_json(text)
        self.assertEqual(result["outer"]["inner"], [1, 2])


class TestParseReact(unittest.TestCase):
    """Test ReAct-style Thought/Action extraction."""

    def test_standard_format(self):
        text = "Thought: I need to search for info\nAction: search[python tutorial]"
        thought, action = OutputParser.parse_react(text)
        self.assertEqual(thought, "I need to search for info")
        self.assertEqual(action, "search[python tutorial]")

    def test_finish_action(self):
        text = "Thought: I have enough info\nAction: Finish[The answer is 42]"
        thought, action = OutputParser.parse_react(text)
        self.assertIn("enough info", thought)
        self.assertEqual(action, "Finish[The answer is 42]")

    def test_case_insensitive(self):
        text = "thought: reasoning here\naction: tool[input]"
        thought, action = OutputParser.parse_react(text)
        self.assertEqual(thought, "reasoning here")
        self.assertEqual(action, "tool[input]")

    def test_extra_text_around(self):
        text = "Some preamble\nThought: analyze the problem\nAction: calc[2+2]\nExtra stuff"
        thought, action = OutputParser.parse_react(text)
        self.assertEqual(thought, "analyze the problem")
        self.assertIn("calc[2+2]", action)

    def test_no_thought(self):
        text = "Action: search[query]"
        thought, action = OutputParser.parse_react(text)
        self.assertIsNone(thought)
        self.assertEqual(action, "search[query]")

    def test_no_action(self):
        text = "Thought: I'm thinking..."
        thought, action = OutputParser.parse_react(text)
        self.assertEqual(thought, "I'm thinking...")
        self.assertIsNone(action)


class TestParseToolCall(unittest.TestCase):
    """Test tool call parsing."""

    def test_standard(self):
        name, inp = OutputParser.parse_tool_call("search[python docs]")
        self.assertEqual(name, "search")
        self.assertEqual(inp, "python docs")

    def test_with_backticks(self):
        name, inp = OutputParser.parse_tool_call("`calculator[2+3]`")
        self.assertEqual(name, "calculator")
        self.assertEqual(inp, "2+3")

    def test_finish(self):
        name, inp = OutputParser.parse_tool_call("Finish[The answer is 42]")
        self.assertEqual(name, "Finish")
        self.assertEqual(inp, "The answer is 42")

    def test_no_match(self):
        name, inp = OutputParser.parse_tool_call("just text")
        self.assertIsNone(name)
        self.assertIsNone(inp)


class TestExtractField(unittest.TestCase):
    """Test generic field extraction."""

    def test_colon_format(self):
        text = "Name: John\nAge: 30\nCity: NYC"
        self.assertEqual(OutputParser.extract_field(text, "Name"), "John")
        self.assertEqual(OutputParser.extract_field(text, "Age"), "30")

    def test_bold_format(self):
        text = "**Result:** success\n**Score:** 95"
        self.assertEqual(OutputParser.extract_field(text, "Result"), "success")

    def test_default(self):
        self.assertEqual(OutputParser.extract_field("no fields", "Missing", "default"), "default")


class TestExtractCodeBlock(unittest.TestCase):
    """Test code block extraction."""

    def test_python_block(self):
        text = 'Here:\n```python\nprint("hello")\n```\n'
        code = OutputParser.extract_code_block(text, "python")
        self.assertEqual(code, 'print("hello")')

    def test_any_block(self):
        text = '```js\nconsole.log("hi")\n```'
        code = OutputParser.extract_code_block(text)
        self.assertEqual(code, 'console.log("hi")')

    def test_no_block(self):
        self.assertIsNone(OutputParser.extract_code_block("no code here"))


class TestRetryPrompt(unittest.TestCase):
    """Test retry prompt building."""

    def test_builds_prompt(self):
        prompt = OutputParser.build_retry_prompt(
            "bad output", "missing Action field", "Thought: ...\nAction: ..."
        )
        self.assertIn("bad output", prompt)
        self.assertIn("missing Action field", prompt)
        self.assertIn("Expected format", prompt)


if __name__ == "__main__":
    unittest.main()
