"""OutputParser - Robust Structured Output Parsing

Provides reliable extraction of structured data from LLM text outputs:
- JSON block extraction with auto-repair (trailing commas, missing braces)
- Regex-based field extraction (Thought / Action / Observation)
- Pydantic model validation (optional)
- Retry-with-feedback: when parsing fails, build an error message the LLM
  can use to correct its output
- Multiple fallback strategies

Usage:
    ```python
    from code.core.output_parser import OutputParser

    parser = OutputParser()

    # Extract JSON from LLM text
    data = parser.parse_json(llm_output)

    # Extract ReAct fields
    thought, action = parser.parse_react(llm_output)

    # Build retry prompt when parsing fails
    retry_prompt = parser.build_retry_prompt(llm_output, error_msg)
    ```
"""

import json
import re
from typing import Any, Optional, Tuple

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None        # type: ignore
    ValidationError = None  # type: ignore


class ParseError(Exception):
    """Raised when output parsing fails after all fallback strategies."""
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class OutputParser:
    """Robust output parser with multiple extraction strategies and auto-repair."""

    @staticmethod
    def parse_json(text: str, strict: bool = False) -> Any:
        """Extract and parse JSON from LLM output.

        Tries multiple strategies:
        1. Parse the whole text as JSON
        2. Extract ```json ... ``` code blocks
        3. Find the first { ... } or [ ... ] region
        4. Auto-repair common issues (trailing commas, single quotes)

        Args:
            text: Raw LLM output.
            strict: If True, raise ParseError on failure instead of returning None.

        Returns:
            Parsed JSON object (dict or list), or None if all strategies fail.

        Raises:
            ParseError: If strict=True and parsing fails.
        """
        # Strategy 1: direct parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: extract code block
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: find first JSON-like region
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            # Find matching close, accounting for nesting
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx: i + 1]
                        try:
                            return json.loads(candidate)
                        except (json.JSONDecodeError, ValueError):
                            # Try repair
                            repaired = OutputParser._repair_json(candidate)
                            try:
                                return json.loads(repaired)
                            except (json.JSONDecodeError, ValueError):
                                pass
                        break

        if strict:
            raise ParseError("Failed to extract JSON from output.", raw_output=text)
        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """Attempt to repair common JSON issues.

        Fixes:
        - Trailing commas before } or ]
        - Single quotes -> double quotes (outside strings)
        - Python True/False/None -> JSON equivalents
        """
        # Replace Python booleans and None
        repaired = text
        repaired = re.sub(r'\bTrue\b', 'true', repaired)
        repaired = re.sub(r'\bFalse\b', 'false', repaired)
        repaired = re.sub(r'\bNone\b', 'null', repaired)

        # Remove trailing commas
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)

        # Single quotes to double quotes (naive - won't handle all edge cases)
        # Only do this if there are no double quotes at all
        if '"' not in repaired and "'" in repaired:
            repaired = repaired.replace("'", '"')

        return repaired


    @staticmethod
    def parse_json_model(text: str, model_class: type) -> Any:
        """Parse JSON from text and validate against a Pydantic model.

        Args:
            text: Raw LLM output.
            model_class: A Pydantic BaseModel subclass.

        Returns:
            Validated model instance.

        Raises:
            ParseError: If parsing or validation fails.
        """
        if not HAS_PYDANTIC:
            raise ParseError("Pydantic is required for model validation but is not installed.")

        data = OutputParser.parse_json(text, strict=True)
        if not isinstance(data, dict):
            raise ParseError(f"Expected a JSON object, got {type(data).__name__}.", raw_output=text)

        try:
            return model_class(**data)
        except ValidationError as e:
            raise ParseError(f"Validation failed: {e}", raw_output=text) from e


    @staticmethod
    def parse_react(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract Thought and Action from ReAct-style output.

        Supports multiple formats:
        - "Thought: ..." and "Action: ..."
        - "Thought:" on one line, content on the next
        - Case-insensitive matching
        - Handles Qwen3-style <think>...</think> blocks

        Args:
            text: Raw LLM output.

        Returns:
            (thought, action) tuple, either may be None.
        """
        # Strip <think>...</think> blocks (Qwen3 reasoning output)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        thought = None
        action = None

        # Try multi-line match first (more robust)
        thought_match = re.search(
            r"(?:^|\n)\s*[Tt]hought\s*:\s*(.*?)(?=\n\s*[Aa]ction\s*:|$)",
            cleaned, re.DOTALL,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Single-line fallback
        if not thought:
            m = re.search(r"[Tt]hought\s*:\s*(.+)", cleaned)
            if m:
                thought = m.group(1).strip()

        # Action extraction — capture only the ToolName[...] part
        action_match = re.search(
            r"(?:^|\n)\s*[Aa]ction\s*:\s*(\w+\[)",
            cleaned,
        )
        if action_match:
            # Find the balanced closing bracket
            start_pos = action_match.end() - 1  # position of '['
            tool_prefix = action_match.group(1)  # e.g. "file["
            bracket_content = OutputParser._extract_balanced_brackets(
                cleaned[start_pos:]
            )
            if bracket_content is not None:
                action = tool_prefix + bracket_content + "]"

        # Fallback: try Finish[...] or plain text action
        if not action:
            m = re.search(r"[Aa]ction\s*:\s*(.+?)(?:\n|$)", cleaned)
            if m:
                action = m.group(1).strip()

        return thought, action

    @staticmethod
    def _extract_balanced_brackets(text: str) -> Optional[str]:
        """Extract content between balanced [ ] while respecting JSON strings.

        Args:
            text: Text starting with '['.

        Returns:
            Content between the balanced brackets (excluding the brackets),
            or None if no balanced match is found.
        """
        if not text or text[0] != '[':
            return None

        depth = 0
        in_string = False
        escape = False

        for i, c in enumerate(text):
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    return text[1:i]  # content between [ and ]

        return None

    @staticmethod
    def parse_tool_call(action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse a tool call from action text.

        Supports formats:
        - tool_name[input]
        - `tool_name[input]`

        Uses balanced bracket matching to correctly handle JSON arguments
        that may contain nested brackets.

        Args:
            action_text: Action string like "file[{\"action\": \"read\"}]".

        Returns:
            (tool_name, tool_input) tuple, either may be None.
        """
        # Remove backticks
        clean = action_text.strip().strip("`")
        match = re.match(r"(\w+)\[", clean)
        if not match:
            return None, None

        tool_name = match.group(1)
        content = OutputParser._extract_balanced_brackets(clean[match.end() - 1:])
        if content is not None:
            return tool_name, content
        return None, None


    @staticmethod
    def extract_field(text: str, field_name: str, default: Optional[str] = None) -> Optional[str]:
        """Extract a named field from structured text.

        Matches patterns like:
        - "FieldName: value"
        - "**FieldName:** value"
        - "## FieldName\nvalue"

        Args:
            text: Input text.
            field_name: Name of the field to extract.
            default: Default value if not found.

        Returns:
            Extracted value string, or default.
        """
        patterns = [
            rf"(?:^|\n)\s*\*{{0,2}}{re.escape(field_name)}\*{{0,2}}\s*:\*{{0,2}}\s*(.+?)(?=\n\s*\*{{0,2}}\w+\s*:|$)",
            rf"(?:^|\n)#+\s*{re.escape(field_name)}\s*\n(.*?)(?=\n#+\s|\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return default

    @staticmethod
    def extract_code_block(text: str, language: Optional[str] = None) -> Optional[str]:
        """Extract the first code block from text.

        Args:
            text: Input text.
            language: Optional language tag filter (e.g. 'python', 'json').

        Returns:
            Code block content, or None.
        """
        if language:
            pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
        else:
            pattern = r"```\w*\s*\n(.*?)```"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None


    @staticmethod
    def build_retry_prompt(
        original_output: str,
        error_message: str,
        expected_format: str = "",
    ) -> str:
        """Build a retry prompt that helps the LLM correct its output.

        Args:
            original_output: The LLM's previous output that failed parsing.
            error_message: Description of what went wrong.
            expected_format: Description of the expected format.

        Returns:
            A prompt string to send back to the LLM.
        """
        prompt = (
            "Your previous response could not be parsed correctly.\n\n"
            f"**Error:** {error_message}\n\n"
        )
        if expected_format:
            prompt += f"**Expected format:**\n{expected_format}\n\n"
        prompt += (
            f"**Your previous output (first 500 chars):**\n"
            f"```\n{original_output[:500]}\n```\n\n"
            "Please try again, making sure your output strictly follows the expected format."
        )
        return prompt
