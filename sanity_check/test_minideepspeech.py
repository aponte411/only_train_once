import os
import unittest

import torch
import torch.nn as nn

from only_train_once import OTO

OUT_DIR = "./cache"


class MiniDeepSpeech(nn.Module):
    """A simplified version of DeepSpeech with Conv1d layers"""

    def __init__(self):
        super().__init__()
        self.input_channels = 1
        self.hidden_channels = 32
        self.output_channels = 16
        self.conv = nn.Sequential(
            nn.Conv1d(
                self.input_channels,
                self.hidden_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
            nn.Conv1d(
                self.hidden_channels,
                self.output_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm1d(self.output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class TestDeepSpeech(unittest.TestCase):
    def test_sanity(self):
        """Main OTO test"""

        model = MiniDeepSpeech()
        batch_size = 2
        sequence_length = 100
        dummy_input = torch.randn(batch_size, 1, sequence_length)
        oto = OTO(model, dummy_input)
        print("Testing visualization...")
        oto.visualize(view=False, out_dir=OUT_DIR)
        full_flops = oto.compute_flops(in_million=False, in_billion=False)["total"]
        full_num_params = oto.compute_num_params()
        print("Full FLOPs:", full_flops)
        print("Full number of parameters:", full_num_params)
        print("Creating compressed model...")
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        print("Loading models...")
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)
        print("Comparing outputs...")
        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference:", max_output_diff.item())
        self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model:", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compressed model:",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )
        print("Computing compression metrics...")
        oto_compressed = OTO(compressed_model, dummy_input)
        compressed_flops = oto_compressed.compute_flops(in_million=False)["total"]
        compressed_num_params = oto_compressed.compute_num_params()
        print(f"Compressed_FLOPs: {compressed_flops}")
        print(f"Compressed number of parameters: {compressed_num_params}")
        self.assertNotEqual(compressed_flops, 0)
        # flop_reduction = 1.0 - compressed_flops / full_flops
        # param_reduction = 1.0 - compressed_num_params / full_num_params
        # print(f"FLOP reduction (%): {flop_reduction * 100:.2f}%")
        # print(f"Param reduction (%): {param_reduction * 100:.2f}%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
