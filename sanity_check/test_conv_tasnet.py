import os
import unittest

import torch
from torchaudio.models import conv_tasnet_base

from only_train_once import OTO

OUT_DIR = "./cache"


class TestConvTasNet(unittest.TestCase):
    def test_sanity(self):
        batch_size = 2
        sample_length = 32000

        dummy_input = torch.randn(batch_size, 1, sample_length)
        model = conv_tasnet_base()
        model
        oto = OTO(model, dummy_input)
        print("Testing visualization...")
        oto.visualize(view=False, out_dir=OUT_DIR)
        full_flops = oto.compute_flops(in_million=True)["total"]
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
        compressed_flops = oto_compressed.compute_flops(in_million=True)["total"]
        compressed_num_params = oto_compressed.compute_num_params()
        print(f"Compressed_FLOPs: {compressed_flops}")
        flop_reduction = 1.0 - compressed_flops / full_flops
        param_reduction = 1.0 - compressed_num_params / full_num_params
        print(f"FLOP reduction (%): {flop_reduction * 100:.2f}%")
        print(f"Param reduction (%): {param_reduction * 100:.2f}%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
