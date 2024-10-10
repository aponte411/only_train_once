import unittest
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

"""
LLM Test Cases
"""
from test_llama_embed_lora import TestLLAMAEmbedLoRA
from test_llamav2 import TestLLAMAv2 # full 32 layers
from test_llamav1 import TestLLAMAv1
from test_llamav1_lora import TestLLAMAv1LoRA
from test_bert import TestBert
from test_phi2 import TestPhi2

"""
CNN Test Cases
"""
from test_resnet18 import TestResNet18
from test_resnet50 import TestResNet50
from test_vgg16bn import TestVGG16BN
from test_convnexttiny import TestConvNextTiny
from test_carn import TestCARN
from test_densenet121 import TestDenseNet121

from test_in_case3 import TestINCase3
from test_convtranspose_in_case1 import TestConvTransposeInCase1
from test_convtranspose_in_case2 import TestConvTransposeInCase2
from test_weight_share_case1 import TestDemoNetWeighShareCase1
from test_weight_share_case2 import TestDemoNetWeighShareCase2

from test_concat_case1 import TestDemoNetConcatCase1
from test_concat_case2 import TestDemoNetConcatCase2

from test_groupconv_case1 import TestGroupConvCase1

from test_group_norm_case1 import TestGroupNormCase1
from test_group_norm_case2 import TestGroupNormCase2
from test_group_norm_case3 import TestGroupNormCase3
from test_group_norm_case4 import TestGroupNormCase4

from test_yolov5 import TestYolov5
# # from test_yolov8 import TestYolov8

from test_diffmodel_cifar import TestDiffModelCIFAR
from test_diffmodel_bedroom import TestDiffModelBedroom
from test_diffmodel_celeba import TestDiffModelCeleba
from test_diffmodel_church import TestDiffModelChurch


OUT_DIR = './cache'

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == '__main__':
    unittest.main()