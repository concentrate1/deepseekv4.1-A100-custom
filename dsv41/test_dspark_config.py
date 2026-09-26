"""CPU-only checks for the reloadable DSpark runtime config."""

from pathlib import Path
import tempfile
import unittest

from .dspark_config import (
    DEFAULT_DSPARK_CONFIG_PATH,
    DSparkConfig,
    DSparkConfigError,
    load_dspark_config,
)


class DSparkConfigTests(unittest.TestCase):
    def test_repo_default_is_greedy(self):
        self.assertEqual(load_dspark_config(DEFAULT_DSPARK_CONFIG_PATH), DSparkConfig(draft_temperature=0.0))
        self.assertTrue(DEFAULT_DSPARK_CONFIG_PATH.is_file())

    def test_non_negative_finite_numbers(self):
        for value, expected in (("0", 0.0), ("0.75", 0.75), ("3", 3.0)):
            with self.subTest(value=value):
                self.assertEqual(self._load('{"draft_temperature": ' + value + '}')
                                 .draft_temperature, expected)

    def test_rejects_schema_and_numeric_errors(self):
        invalid = (
            "[]", "{}", '{"draft_temperature": 0, "unknown": true}',
            '{"draft_temperature": null}', '{"draft_temperature": true}',
            '{"draft_temperature": "0"}', '{"draft_temperature": -0.1}',
            '{"draft_temperature": NaN}', '{"draft_temperature": Infinity}',
            '{"draft_temperature": 1e999}',
            '{"draft_temperature": ' + '9' * 400 + '}',
            '{"draft_temperature": 0, "draft_temperature": 1}',
            '{"draft_temperature": 0',
        )
        for content in invalid:
            with self.subTest(content=content):
                with self.assertRaises(DSparkConfigError):
                    self._load(content)

    def test_missing_file_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_dspark_config(Path(directory) / "missing.json")

    @staticmethod
    def _load(content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dspark.json"
            path.write_text(content, encoding="utf-8")
            return load_dspark_config(path)


if __name__ == "__main__":
    unittest.main()
