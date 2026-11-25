import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app import lookup, detect_faces

class ProjectTests(unittest.TestCase):
    def test_known_profile(self):
        path = Path(__file__).resolve().parents[1] / "data/demo_profiles.csv"
        self.assertEqual(lookup(path, "DEMO001")["display_name"], "Alex Example")

    def test_unknown_profile(self):
        path = Path(__file__).resolve().parents[1] / "data/demo_profiles.csv"
        self.assertIsNone(lookup(path, "UNKNOWN"))

    def test_bad_image(self):
        with self.assertRaises(ValueError):
            detect_faces("/not/a/real/file.jpg")

if __name__ == "__main__":
    unittest.main()
