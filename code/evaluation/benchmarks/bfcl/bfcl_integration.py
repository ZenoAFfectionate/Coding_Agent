"""
BFCL Official Evaluation Tool Integration Module

Wraps the BFCL official evaluation tool calls, providing a convenient interface.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os

from ....utils.subprocess_utils import safe_run


class BFCLIntegration:
    """BFCL Official Evaluation Tool Integration Class

    Provides the following features:
    1. Check if the BFCL evaluation tool is installed
    2. Install the BFCL evaluation tool
    3. Run the BFCL official evaluation
    4. Parse evaluation results

    Usage example:
        integration = BFCLIntegration()

        # Check and install
        if not integration.is_installed():
            integration.install()

        # Run evaluation
        integration.run_evaluation(
            model_name="HelloAgents",
            category="simple_python",
            result_file="result/HelloAgents/BFCL_v3_simple_python_result.json"
        )

        # Parse results
        scores = integration.parse_results(
            model_name="HelloAgents",
            category="simple_python"
        )
    """

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """Initialize BFCL integration.

        Args:
            project_root: BFCL project root directory. If None, uses the current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.result_dir = self.project_root / "result"
        self.score_dir = self.project_root / "score"

    def is_installed(self) -> bool:
        """Check if the BFCL evaluation tool is installed.

        Returns:
            True if installed, False otherwise.
        """
        try:
            result = safe_run(
                ["bfcl", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install(self) -> bool:
        """Install the BFCL evaluation tool.

        Returns:
            True if installation succeeds, False otherwise.
        """
        print("📦 Installing BFCL evaluation tool...")
        print("   Running: pip install bfcl-eval")

        try:
            result = safe_run(
                ["pip", "install", "bfcl-eval"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print("✅ BFCL evaluation tool installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Installation timed out")
            return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False

    def prepare_result_file(
        self,
        source_file: Union[str, Path],
        model_name: str,
        category: str
    ) -> Path:
        """Prepare the result file required for BFCL evaluation.

        BFCL expected file path format:
        result/{model_name}/BFCL_v3_{category}_result.json

        Args:
            source_file: Source result file path.
            model_name: Model name.
            category: Evaluation category.

        Returns:
            Target file path.
        """
        source_file = Path(source_file)

        # Create target directory
        target_dir = self.result_dir / model_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Determine target file name
        target_file = target_dir / f"BFCL_v3_{category}_result.json"

        # Copy file
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, target_file)
            print(f"✅ Result file prepared")
            print(f"   Source file: {source_file}")
            print(f"   Target file: {target_file}")
        else:
            print(f"⚠️ Source file does not exist: {source_file}")

        return target_file

    def run_evaluation(
        self,
        model_name: str,
        category: str,
        result_file: Optional[Union[str, Path]] = None
    ) -> bool:
        """Run the BFCL official evaluation.

        Args:
            model_name: Model name.
            category: Evaluation category.
            result_file: Result file path (optional; if provided, the file will be prepared first).

        Returns:
            True if evaluation succeeds, False otherwise.
        """
        # If a result file is provided, prepare it first
        if result_file:
            self.prepare_result_file(result_file, model_name, category)

        # Set environment variables
        env = os.environ.copy()
        env["BFCL_PROJECT_ROOT"] = str(self.project_root)

        print(f"\n🔧 Running BFCL official evaluation...")
        print(f"   Model: {model_name}")
        print(f"   Category: {category}")
        print(f"   Project root: {self.project_root}")

        # Build command
        cmd = [
            "bfcl", "evaluate",
            "--model", model_name,
            "--test-category", category
        ]

        print(f"   Command: {' '.join(cmd)}")

        try:
            result = safe_run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env
            )

            if result.returncode == 0:
                print("✅ BFCL evaluation completed")
                print(result.stdout)
                return True
            else:
                print(f"❌ Evaluation failed")
                print(f"   Error message: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Evaluation timed out")
            return False
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return False

    def parse_results(
        self,
        model_name: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Parse BFCL evaluation results.

        Args:
            model_name: Model name.
            category: Evaluation category.

        Returns:
            Evaluation result dictionary, or None if the file does not exist.
        """
        # BFCL evaluation result path
        score_file = self.score_dir / model_name / f"BFCL_v3_{category}_score.json"

        if not score_file.exists():
            print(f"⚠️ Evaluation result file does not exist: {score_file}")
            return None

        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            print(f"\n📊 BFCL Evaluation Results")
            print(f"   Model: {model_name}")
            print(f"   Category: {category}")

            # Extract key metrics
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")

            return results

        except Exception as e:
            print(f"❌ Failed to parse results: {e}")
            return None

    def get_summary_csv(self) -> Optional[Path]:
        """Get the summary CSV file path.

        BFCL generates the following CSV files:
        - data_overall.csv: Overall scores
        - data_live.csv: Live dataset scores
        - data_non_live.csv: Non-live dataset scores
        - data_multi_turn.csv: Multi-turn conversation scores

        Returns:
            Path to data_overall.csv, or None if it does not exist.
        """
        csv_file = self.score_dir / "data_overall.csv"

        if csv_file.exists():
            print(f"\n📄 Summary CSV file: {csv_file}")
            return csv_file
        else:
            print(f"⚠️ Summary CSV file does not exist: {csv_file}")
            return None

    def print_usage_guide(self):
        """Print usage guide."""
        print("\n" + "="*60)
        print("BFCL Official Evaluation Tool Usage Guide")
        print("="*60)
        print("\n1. Install the BFCL evaluation tool:")
        print("   pip install bfcl-eval")
        print("\n2. Set environment variable:")
        print(f"   export BFCL_PROJECT_ROOT={self.project_root}")
        print("\n3. Prepare the result file:")
        print("   Place evaluation results in: result/{model_name}/BFCL_v3_{category}_result.json")
        print("\n4. Run evaluation:")
        print("   bfcl evaluate --model {model_name} --test-category {category}")
        print("\n5. View results:")
        print("   Evaluation results at: score/{model_name}/BFCL_v3_{category}_score.json")
        print("   Summary results at: score/data_overall.csv")
        print("\n" + "="*60)
