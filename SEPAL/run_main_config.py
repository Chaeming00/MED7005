import json
import subprocess
import argparse
import os
import sys

print("Command-line arguments:", sys.argv)

# Get parsed the path of the config file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_config', type=str, default='config_dataset.json', help='Path to the .json file with the configs of the dataset.')
parser.add_argument('--model_config', type=str, default='config_model.json', help='Path to the .json file with the configs of the model.')
parser.add_argument('--train_config', type=str, default='config_train.json', help='Path to the .json file with the configs of the training.')
parser.add_argument('--model_directory', type=str, default=None, help='Path to the directory of a model.')
# Add the new arguments to the parser
parser.add_argument('--train_path', type=str, required=False, help='Path to the training data CSV file.')
parser.add_argument('--gene_files', type=json.loads, required=False, help='JSON string with gene files paths.')

args = parser.parse_args()

# Read the dataset, model, and train configs
with open(args.dataset_config, 'rb') as f:
    config_params = json.load(f)

with open(args.model_config, 'rb') as f:
    config_params.update(json.load(f))

with open(args.train_config, 'rb') as f:
    config_params.update(json.load(f))

# gene_files 추가 처리
if 'gene_files' in config_params and isinstance(config_params['gene_files'], str):
    import json
    try:
        config_params['gene_files'] = json.loads(config_params['gene_files'])
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for --gene_files. Use a proper JSON string.")

# 디버깅: gene_files 값 확인
print("Gene Files:", config_params['gene_files'])


# Add train_path and gene_files if provided
if args.train_path:
    config_params['train_path'] = args.train_path

if args.gene_files:
    config_params['gene_files'] = args.gene_files
    
# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = config_params['cuda']

# If model_directory is specified
if args.model_directory:
    params_path = os.path.join(args.model_directory, 'script_params.json')
    with open(params_path, 'rb') as f:
        params = json.load(f)
    config_params.update(params)
    config_params['model_directory'] = args.model_directory

# Determine which script to call
if config_params['sota'] == 'None':
    command_list = ['python', 'main.py']
elif config_params['sota'] == 'pretrain':
    command_list = ['python', 'pretrain_backbone.py']
else:
    raise ValueError(f"Unexpected sota value: {config_params['sota']}")

for key, val in config_params.items():
    command_list.append(f'--{key}')
    # val이 dict 타입이면 json.dumps로 처리
    if isinstance(val, (dict, list)):
        command_list.append(json.dumps(val))
    else:
        command_list.append(str(val))

# Execute the command
subprocess.call(command_list)

