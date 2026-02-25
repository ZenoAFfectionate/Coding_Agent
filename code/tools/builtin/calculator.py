"""Calculator tool - safe math expression evaluator."""

import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool


class CalculatorTool(Tool):
    """Python calculator tool - evaluates math expressions safely via AST."""

    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.pow,  # ^ treated as exponentiation (like math notation)
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
    }

    # Supported functions and constants
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }

    def __init__(self):
        super().__init__(
            name="python_calculator",
            description="Evaluate math expressions. Supports arithmetic (+, -, *, /, //, %, **), "
                        "functions (sqrt, sin, cos, tan, log, exp, abs, round, max, min), "
                        "and constants (pi, e). Examples: 2+3*4, sqrt(16), sin(pi/2)."
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "Error: expression cannot be empty."

        try:
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            return str(result)
        except Exception as e:
            return f"Calculation error: {str(e)}"

    def _eval_node(self, node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left),
                self._eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    def get_parameters(self):
        from ..base import ToolParameter
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Math expression to evaluate (supports +, -, *, /, **, sqrt, sin, cos, log, etc.)",
                required=True
            )
        ]


def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    tool = CalculatorTool()
    return tool.run({"input": expression})
