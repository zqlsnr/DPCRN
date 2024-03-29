"""
#!/usr/bin/python3
@FileName: convert_weights_to_tf_lite.py.py
@Time: 2022/10/18 13:53        
@Author: zql
"""

import argparse
from main import DPCRN_model
import yaml


if __name__ == '__main__':
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description='data evaluation')
    parser.add_argument('--weights_file', '-m', default='./pretrained_weights/DPCRN_base/models_base_20_0.023676.h5',
                        help='path to .h5 weights file')
    parser.add_argument('--target_folder', '-t', default='./pretrained_weights/model_tflite/',
                        help='target folder for saved model')
    parser.add_argument('--quantization', '-q',
                        help='use quantization (True/False)',
                        default='True')

    args = parser.parse_args()
    f = open('./configuration/DPCRN-base.yaml', 'r', encoding='utf-8')
    result = f.read()
    print(result)

    config_dict = yaml.safe_load(result)

    converter = DPCRN_model(batch_size=1, length_in_s=5, lr=1e-3, config=config_dict)

    converter.create_tf_lite_model(args.weights_file,
                                   args.target_folder,
                                   use_dynamic_range_quant=bool(args.quantization))