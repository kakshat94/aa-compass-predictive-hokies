"""Tests for the configurable secondary-qualification proportions."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.phase2_training.qualification_mapper import (
    build_prop_table,
    sprinkle_qualifications,
)
from pipelines.phase2_training.training_config import (
    load_secondary_proportion_scales,
)

PARAMS = Path("params.yaml")
QUAL_DATA = Path("inputs/Supply_sample.csv")
SEED = 42
FLAG_COLUMNS = ["DOMESTIC", "ATC", "ROC", "CHF", "OC", "LATIN", "EUROPE", "PACIFIC"]


def _roster() -> pd.DataFrame:
    rows = []
    for work_center, count in (("Domestic", 60), ("ATC", 200), ("ROC", 80)):
        for index in range(count):
            rows.append(
                {
                    "TECH_ID": f"{work_center}-{index}",
                    "EMPLOYEE_WORK_CENTER_ID": work_center,
                }
            )
    return pd.DataFrame(rows)


def _write_params(text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


class SecondaryProportionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qual_data = pd.read_csv(QUAL_DATA)
        cls.prop_table = build_prop_table(cls.qual_data)
        cls.roster = _roster()

    def _sprinkle(self, scales=None) -> pd.DataFrame:
        return sprinkle_qualifications(
            self.roster, self.prop_table, seed=SEED, prop_scales=scales
        )

    def test_params_file_has_one_scale_for_every_pair(self) -> None:
        scales = load_secondary_proportion_scales(PARAMS)
        self.assertEqual(len(scales), 42)
        self.assertEqual(set(scales.values()), {1.0})

    def test_default_scales_match_current_behaviour(self) -> None:
        baseline = self._sprinkle()
        with_defaults = self._sprinkle(load_secondary_proportion_scales(PARAMS))
        pd.testing.assert_frame_equal(
            baseline[FLAG_COLUMNS], with_defaults[FLAG_COLUMNS]
        )
        pd.testing.assert_series_equal(
            baseline["FINAL_QUALS"], with_defaults["FINAL_QUALS"]
        )

    def test_zero_scale_removes_pair(self) -> None:
        scales = load_secondary_proportion_scales(PARAMS)
        scales[("ATC", "ROC")] = 0.0
        result = self._sprinkle(scales)
        atc = result[result["EMPLOYEE_WORK_CENTER_ID"] == "ATC"]
        self.assertEqual(int(atc["ROC"].sum()), 0)

    def test_half_scale_halves_the_count(self) -> None:
        scales = load_secondary_proportion_scales(PARAMS)
        scales[("ATC", "ROC")] = 0.5
        result = self._sprinkle(scales)
        atc = result[result["EMPLOYEE_WORK_CENTER_ID"] == "ATC"]
        expected = int(math.floor(200 * self.prop_table.loc["ATC", "ROC"] * 0.5)) + 1
        self.assertEqual(int(atc["ROC"].sum()), expected)

    def test_missing_section_defaults_to_one(self) -> None:
        path = _write_params("shifts:\n  morning_equals_evening: true\n")
        scales = load_secondary_proportion_scales(path)
        self.assertEqual(set(scales.values()), {1.0})
        Path(path).unlink()

    def test_missing_pair_defaults_to_one(self) -> None:
        path = _write_params(
            "training:\n"
            "  secondary_proportion_scales:\n"
            "    ATC:\n"
            "      ROC: 0.25\n"
        )
        scales = load_secondary_proportion_scales(path)
        self.assertEqual(scales[("ATC", "ROC")], 0.25)
        self.assertEqual(scales[("ATC", "CHF")], 1.0)
        Path(path).unlink()

    def test_invalid_inputs_raise(self) -> None:
        cases = {
            "above one": "ATC:\n  ROC: 1.5\n",
            "negative": "ATC:\n  ROC: -0.1\n",
            "unknown secondary": "ATC:\n  BOGUS: 1.0\n",
            "same primary": "ATC:\n  ATC: 1.0\n",
            "domestic primary": "Domestic:\n  ATC: 1.0\n",
            "non numeric": "ATC:\n  ROC: high\n",
            "boolean": "ATC:\n  ROC: true\n",
        }
        for label, body in cases.items():
            with self.subTest(label=label):
                path = _write_params(
                    "training:\n  secondary_proportion_scales:\n"
                    + "".join(f"    {line}\n" for line in body.splitlines())
                )
                with self.assertRaises(ValueError):
                    load_secondary_proportion_scales(path)
                Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
