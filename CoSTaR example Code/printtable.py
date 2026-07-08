import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

		
with open('matrix.json', 'r') as file:
	matrix = json.load(file)

table = ''
for key in ['IOU', 'tp', 'fp', 'lp', 'pp', 'AR', 'AP', 'AR50', 'AP50', 'AR30', 'AP30', 'AR10', 'AP10']:
	table = table + '&' + matrix[key]

with open('table.txt', 'w') as f:
        f.write(table)