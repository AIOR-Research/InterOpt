from pathlib import Path
import sys

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "shared_tools"
sys.path.insert(0, str(TOOLS_ROOT))

from baseline_runner_common import main


if __name__ == "__main__":
    main(
        method_name="orpilot",
        method_label="ORPilot-style Interview Agent",
        default_prompt_path=Path(__file__).resolve().parent / "prompts" / "agent_prompt.md",
    )
