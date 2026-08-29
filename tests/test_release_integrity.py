"""Fast, data-independent checks for the publication release boundary."""
import json
import unittest

from tools import release_manifest, verify_environment


class ReleaseIntegrityTests(unittest.TestCase):
    def test_manifest_matches_distributed_files(self):
        expected = json.loads(release_manifest.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(expected, release_manifest.snapshot())

    def test_manifest_covers_canonical_code_and_evidence(self):
        current = release_manifest.snapshot()
        self.assertEqual(set(current["canonical_code"]), set(release_manifest.CANONICAL_FILES))
        self.assertEqual(set(current["manuscript_evidence"]), set(release_manifest.RESULT_FILES))
        self.assertEqual(
            set(current["reproduction_environment"]),
            set(release_manifest.ENVIRONMENT_FILES),
        )

    def test_runtime_matches_locked_environment(self):
        self.assertEqual([], verify_environment.mismatches())


if __name__ == "__main__":
    unittest.main()
