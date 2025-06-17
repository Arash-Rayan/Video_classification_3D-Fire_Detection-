import argparse 
parser = argparse.ArgumentParser() 

import torch 

default_args = {
    'learning_rate': 0.9, 
    'epochs': 1,
    'batch_size': 32, 
    'root':'/media/ai/External/datasets/firesmoke_dataset',
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu'
}

parser.add_argument('--learning_rate', type=int ,default= default_args['learning_rate'])
parser.add_argument('--epochs', type=int, default = default_args['epochs'])
parser.add_argument('--batch_size', type=int, default=default_args['batch_size'])
parser.add_argument('--root' , type=str, default=default_args['root'])
parser.add_argument('--device', type=str , default=default_args['device'])

args = parser.parse_args()
