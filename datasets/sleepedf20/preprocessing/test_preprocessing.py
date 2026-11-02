"""Tests for timing, exclusion and preservation of gaps in the epoch grid."""
import unittest

import numpy as np

from preprocess_sleepedf20 import epoch_labels, select_epochs


class PreprocessingTests(unittest.TestCase):
    def test_stage_mapping_and_tail_clipping(self):
        labels = epoch_labels(
            [0, 30, 60, 90, 120, 150, 180, 210, 240], [30] * 8 + [300],
            ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
             "Sleep stage 4", "Sleep stage R", "Movement time", "Sleep stage ?", "Sleep stage W"], 9)
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 3, 4, -2, -1, 0])

    def test_window_uses_true_sleep_and_original_time(self):
        labels = np.zeros(400, dtype=np.int8)
        labels[0] = -1  # Unknown must not count as sleep onset.
        labels[100:201] = 2
        labels[125] = 0  # Intra-night wake is retained.
        labels[150] = -2
        labels[160] = -1
        indices, start, stop, first, last = select_epochs(labels)
        self.assertEqual((start, stop, first, last), (40, 261, 100, 200))
        self.assertEqual(len(indices), 219)
        self.assertIn(125, indices)
        self.assertNotIn(150, indices)
        self.assertNotIn(160, indices)
        self.assertEqual(int(np.sum(np.diff(indices) != 1)), 2)

    def test_margin_clips_at_recording_edges(self):
        indices, start, stop, _, _ = select_epochs(np.array([0, 1, 0, 4, 0]))
        self.assertEqual((start, stop), (0, 5))
        np.testing.assert_array_equal(indices, np.arange(5))

    def test_annotation_offset_and_missing_labels(self):
        np.testing.assert_array_equal(
            epoch_labels([0], [30], ["Sleep stage 2"], 3, offset=30), [-3, 2, -3])

    def test_reject_ambiguous_annotations(self):
        with self.assertRaises(ValueError):
            epoch_labels([1], [30], ["Sleep stage W"], 2)
        with self.assertRaises(ValueError):
            epoch_labels([0, 0], [30, 30], ["Sleep stage W", "Sleep stage 2"], 2)
        with self.assertRaises(ValueError):
            select_epochs(np.array([0, -1, -2]))


if __name__ == "__main__":
    unittest.main()
