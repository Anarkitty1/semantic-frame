"""Tests for advanced analyzers."""

import numpy as np

from semantic_frame.core.analyzers import detect_step_changes
from semantic_frame.core.enums import StructuralChange


class TestAdvancedAnalyzers:
    def test_detect_step_up(self):
        """Test detection of a step up."""
        # Create data with a clear step up
        # 20 points at 10, then 20 points at 20
        data = np.concatenate([np.full(20, 10.0), np.full(20, 20.0)])

        change_type, index = detect_step_changes(data, window_size=5)

        assert change_type == StructuralChange.STEP_UP
        # The change happens at index 20
        # The detector might find it slightly around 20 depending on window
        assert 18 <= index <= 22

    def test_detect_step_down(self):
        """Test detection of a step down."""
        # Create data with a clear step down
        data = np.concatenate([np.full(20, 20.0), np.full(20, 10.0)])

        change_type, index = detect_step_changes(data, window_size=5)

        assert change_type == StructuralChange.STEP_DOWN
        assert 18 <= index <= 22

    def test_no_step_change(self):
        """Test that no change is detected in stable data."""
        data = np.full(50, 10.0)
        # Add some noise
        data += np.random.normal(0, 0.1, 50)

        change_type, index = detect_step_changes(data)

        assert change_type == StructuralChange.NONE
        assert index is None

    def test_gradual_change_ignored(self):
        """Test that gradual trends are NOT detected as step changes."""
        # Linear trend from 10 to 20
        data = np.linspace(10, 20, 50)

        change_type, index = detect_step_changes(data, threshold=3.0)

        assert change_type == StructuralChange.NONE
