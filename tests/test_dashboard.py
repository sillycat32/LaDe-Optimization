import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardWidgetTests(unittest.TestCase):
    def test_single_hour_horizon_omits_invalid_allocation_slider(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=90).run()
        horizon_slider = next(
            slider
            for slider in app.sidebar.slider
            if slider.label == "Planning horizon (hours)"
        )

        horizon_slider.set_value(1)
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn(
            "Allocate fleet for forecast hour",
            [slider.label for slider in app.sidebar.slider],
        )
        self.assertIn(
            "Fleet allocation uses the only forecast hour available.",
            [caption.value for caption in app.sidebar.caption],
        )
