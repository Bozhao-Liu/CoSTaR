import os
import shutil
from Evaluation.getmatrix import save_matric_from_path
from tqdm import tqdm
import numpy as np
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
       
    v = v.replace('\r', '')
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description='batch wise result analysis')
parser.add_argument('--load', default = True, type=str2bool, 
			help="specify whether load result from previously acquired matrics (default: True)")

def create_table(path = '.'):
	args,unknown = parser.parse_known_args()
	title = 'net & loss'
	table = ' \\\\ \n'
	
	keys = []
	path = os.path.join(path, 'Result', 'prediction')
	networks = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder != '__pycache__']
	priority_order = {"Segformer": 0, "MedT": 1, "MSegnet": 2, "unet": 3, "UnetT":4, "Jin":5, "JinPlus":6, "JinPP":7, "JinPPViT":8, "UJin":9}
	networks = sorted(networks, key=lambda item: priority_order.get(item, float('inf')))
	with tqdm(total = len(networks), desc = 'networks') as t:
		for network in networks:
			t.set_description("network = {}".format(network))
			line = network
			network = os.path.join(path, network)
			losses = [folder for folder in os.listdir(network) if os.path.isdir(os.path.join(network, folder)) and folder != '__pycache__']
			for loss in tqdm(losses, desc = 'loss', leave = 0):
				result = line + ' & ' + loss
				loss = os.path.join(network, loss)
				shutil.copyfile('getmatrix.py', os.path.join(loss, 'getmatrix.py'))	
				shutil.copyfile('printtable.py', os.path.join(loss, 'printtable.py'))
				metric = save_matric_from_path(loss, read_from_history = args.load)
				for key in keys:
					result = result + ' & ' + '{}±{} '.format(np.round(np.mean(metric[key])*100, decimals=1), np.round(np.std(metric[key])*100, decimals=1))
					
				for key in metric:
					if key not in keys:
						keys.append(key)
						title = title + ' & ' + key
						result = result + ' & ' + '{}±{} '.format(np.round(np.mean(metric[key])*100, decimals=1), np.round(np.std(metric[key])*100, decimals=1))
						
				with open(os.path.join(loss, 'table.txt'), 'w') as f:
					f.write(result)
					
				table = table + result + ' \\\\ \n'	
			t.update()

	with open(os.path.join(path, 'table.txt'), 'w') as f:
				f.write(title + table)
			

if __name__ == '__main__':
	create_table('.')
