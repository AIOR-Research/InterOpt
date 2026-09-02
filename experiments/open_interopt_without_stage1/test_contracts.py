from __future__ import annotations

import unittest
from pathlib import Path

import run_pipeline


PIPELINE_DIR = Path(__file__).resolve().parent


class NoStage1ContractTests(unittest.TestCase):
    def test_prompt_bundle_excludes_gap_search_prompt(self) -> None:
        prompts = run_pipeline.load_prompt_files(PIPELINE_DIR / "prompts")
        self.assertEqual(len(prompts), 5)
        self.assertFalse((PIPELINE_DIR / "prompts" / "gap_search_prompt.md").exists())

    def test_stage1_call_path_is_absent(self) -> None:
        source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
        forbidden = [
            "gap_search_client.complete",
            "gap_search_prompt.md",
            "Internal gap-search guidance",
            "render_gap_search_user_message",
            "normalize_gap_search_result",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_statistics_record_disabled_stage1(self) -> None:
        source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
        required = [
            '"ablation_arm": ABLATION_ARM',
            '"gap_search_enabled": False',
            '"gap_search_call_count": 0',
            '"gap_search_model": None',
            '"gap_search_estimated_cost_usd": 0.0',
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_formal_defaults_are_k5_maxturn20(self) -> None:
        import run_pipeline_parallel

        parser = run_pipeline_parallel.build_argument_parser()
        args = parser.parse_args(
            [
                "--toml_dir",
                str(PIPELINE_DIR),
                "--output_dir",
                str(PIPELINE_DIR / "_unused_contract_output"),
            ]
        )
        self.assertEqual(args.k, 5)
        self.assertEqual(args.max_turns, 20)


if __name__ == "__main__":
    unittest.main()
