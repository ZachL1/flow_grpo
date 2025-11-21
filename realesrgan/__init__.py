import os
import yaml

__all__ = ['data_config']

with open(os.path.join(os.path.dirname(__file__), 'data_config.yaml'), 'r', encoding='utf-8') as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)