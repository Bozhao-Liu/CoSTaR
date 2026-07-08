import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import json
import shutil

eps=1e-8
def mask_iou(mask1, mask2):
	"""
	Compute IoU between two batches of binary masks.
	mask1, mask2: torch tensors of shape (N, H, W)
	Returns: tensor of shape (N,) with IoU per mask
	"""
	mask1 = mask1.bool()
	mask2 = mask2.bool()
	inter = torch.logical_and(mask1, mask2).flatten(1).sum(dim=1).float()  # (N,)
	union = torch.logical_or(mask1, mask2).flatten(1).sum(dim=1).float()  # (N,)

	return inter / (union + eps)

def precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold=0.0, eps=1e-8):
	"""
	Compute precision and recall for segmentation masks at a given IoU threshold,
	using precomputed IoUs from mask_iou function.

	Args:
		gt_masks (torch.BoolTensor): Ground truth masks, shape [N, H*W]
		pred_masks (torch.BoolTensor): Predicted masks, shape [N, H*W]
		IOU_threshold (float): IoU threshold to count a prediction as true positive
		eps (float): Small epsilon to avoid division by zero

	Returns:
		precision (float), recall (float)
	"""
	assert gt_masks.shape == pred_masks.shape, "GT and prediction masks must have the same shape"

	# Use precomputed IoUs
	ious = mask_iou(pred_masks, gt_masks)  # shape [N,]

	pred_nonempty = pred_masks.sum(dim=1) > 0
	gt_nonempty = gt_masks.sum(dim=1) > 0

	# True positives: prediction and GT exist, IoU >= threshold
	tp = (ious > IOU_threshold).sum().float()

	# False negatives: GT exists but prediction IoU < threshold or prediction empty
	fn = torch.logical_and(gt_nonempty, ious <= IOU_threshold ).sum().float()

	# False positives: prediction exists but GT does NOT exist
	fp = torch.logical_and(pred_nonempty, ~gt_nonempty).sum().float()

	precision = tp / (tp + fp + eps)
	recall = tp / (tp + fn + eps)

	return precision.item(), recall.item()

def compute_mPR(gt_masks, pred_masks, device, thresholds:list = [50, 95]):
	if thresholds[1] > 95:
		thresholds[1] = 95
		
	if thresholds[0] < 0:
		thresholds[0] = 0
		
	iou_thresholds = torch.arange(thresholds[0]/100, thresholds[1]/100 + 0.05, 0.05, device=device)
	APs, ARs = [], []
	for thr in iou_thresholds:
		p, r = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = thr.item())
		APs.append(p)
		ARs.append(r)
		
	return float(torch.tensor(APs, device=device).mean()), float(torch.tensor(ARs, device=device).mean())
	
def get_best_threshold(y_true, y_score, thresholds):
	best_iou = 0
	best_threshold = 0.5
	for t in thresholds:
		y_pred = (y_score >= t).bool()
		y_true = y_true.bool()
		inter = torch.logical_and(y_true, y_pred).sum().float()
		union = torch.logical_or(y_true, y_pred).sum().float()
		iou = inter/union
		if iou > best_iou:
			best_iou = iou
			best_threshold = t.item()
			
	return best_iou, best_threshold

def compute_coverage(y_true, y_pred):
	
	TP = torch.logical_and(y_true, y_pred).sum(dim=1).float()
	PredP = y_pred.sum(dim=1).float()
	LabelP = y_true.sum(dim=1).float()
	haveP = PredP>0
	haveGT = LabelP>0

	miss = torch.logical_and(y_true, ~y_pred).sum(dim=1).float()
	miss[haveGT] = miss[haveGT] / (LabelP[haveGT] + eps)
	overflow = torch.logical_and(~y_true, y_pred).sum(dim=1).float() * haveGT
	overflow[haveGT] = overflow[haveGT] / (LabelP[haveGT] + eps)

	precisions = torch.zeros_like(TP)
	recalls = torch.zeros_like(TP)

	# Compute only for non-empty images
	precisions[haveP] = TP[haveP] / (PredP[haveP] + eps)
	recalls[haveGT] = TP[haveGT] / (LabelP[haveGT] + eps)

	# Mean over valid images
	mean_precision = precisions[haveP].mean().item() if haveP.sum() > 0 else 0.0
	mean_recall = recalls[haveGT].mean().item() if haveGT.sum() > 0 else 0.0
	mean_miss = miss[haveGT].mean().item() if haveGT.sum() > 0 else 0.0
	mean_overflow = overflow[haveGT].mean().item() if haveGT.sum() > 0 else 0.0

	return mean_precision, mean_recall, mean_miss, mean_overflow
	
def categorize_size_by_bbox(mask):
	"""
	Categorize mask size based on bounding box area.
	mask: 2D tensor (H,W), binary
	Returns: 'small', 'medium', or 'large'
	"""
	mask_bool = mask.bool()
	
	
	if mask_bool.sum() == 0:
		return 'empty'  # fallback for empty mask
		
	if mask_bool.size()[0] == 1:
		mask_bool = mask_bool.view((mask_bool.size()[1], mask_bool.size()[2]))
	
	# Find bounding box
	rows = torch.any(mask_bool, dim=1).nonzero(as_tuple=False).squeeze()
	cols = torch.any(mask_bool, dim=0).nonzero(as_tuple=False).squeeze()
	
	# If single-pixel object, nonzero returns 0D tensor, handle that
	if rows.ndim == 0:
		min_row, max_row = rows.item(), rows.item()
	else:
		min_row, max_row = rows[0].item(), rows[-1].item()
		
	if cols.ndim == 0:
		min_col, max_col = cols.item(), cols.item()
	else:
		min_col, max_col = cols[0].item(), cols[-1].item()
	
	bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
	
	# Categorize based on COCO-style thresholds
	if bbox_area < 32**2:
		return 'small'
	elif bbox_area < 96**2:
		return 'medium'
	else:
		return 'large'
		
def compute_sml(gt_masks, pred_masks):
	# -----------------------------
	# Compute size-based APs/APm/APl & ARs/ARm/ARl
	# -----------------------------
	size_categories = defaultdict(lambda: defaultdict(lambda: 0))
	
	# For each image
	
	N = gt_masks.size()[0]
	for i in range(N):
		gt_mask = gt_masks[i]
		pred_mask = pred_masks[i]
		size = categorize_size_by_bbox(gt_mask)
		if size == 'empty':
			continue
		
		size_categories[size]['TP'] += float(torch.logical_and(pred_mask, gt_mask).sum())
		size_categories[size]['FP'] += float(torch.logical_and(pred_mask, ~gt_mask).sum())
		size_categories[size]['FN'] += float(torch.logical_and(~pred_mask, gt_mask).sum())
	
	# Aggregate size metrics
	def mean_prec_rec(metrics):
		precision = metrics['TP']/(metrics['TP']+metrics['FP']+eps)
		recall = metrics['TP']/(metrics['TP']+metrics['FP']+eps)
		return precision, recall
	
	APs, ARs = mean_prec_rec(size_categories['small'])
	APm, ARm = mean_prec_rec(size_categories['medium'])
	APl, ARl = mean_prec_rec(size_categories['large'])
	
	return {
		'APs': APs,
		'APm': APm,
		'APl': APl,
		'ARs': ARs,
		'ARm': ARm,
		'ARl': ARl,
	}
	
	
def compute_segmentation_metrics(gt_masks, pred_masks, thresholds=None):
	"""
	Pixel-level segmentation metrics (GPU-only).
	gt_masks: torch tensor (N,H,W), binary ground truth
	pred_masks: torch tensor (N,H,W), probabilities in [0,1]
	"""
	device = gt_masks.device
	N = gt_masks.shape[0]
	
	# Best threshold by IoU
	if thresholds is None:
		thresholds = torch.round(torch.linspace(0, 1.0, 51, device=device), decimals = 2)
	
	best_iou, best_threshold = get_best_threshold(gt_masks, pred_masks, thresholds)
	#convert probability mask to segmentation mask
	pred_masks = (pred_masks > best_threshold).bool()
	APR = compute_sml(gt_masks.bool(), pred_masks)
	gt_masks = gt_masks.view(N, -1).bool()
	pred_masks = pred_masks.view(N, -1).bool()
	# Fixed thresholds
	AP30, AR30 = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = 0.3)
	AP, AR = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = 0)
	
	mAP, mAR = compute_mPR(gt_masks, pred_masks, device, thresholds = [0, 100])

	precision, recall, miss, overflow = compute_coverage(gt_masks, pred_masks)
	
	result = {
		'best-threshold': best_threshold,
		'IoU': best_iou,
		'Dice-Coeff': 2 * best_iou/(best_iou + 1),
		'mAP': mAP,
		'mAR': mAR,
		'AP30': AP30,
		'AR30': AR30,
		'AP': AP,
		'AR': AR,
		'PixRecall':recall,
		'PixPrecision':precision,
		'miss': miss,
		'overflow': overflow
	}
	
	#APR = compute_sml(gt_masks, pred_masks)
	for key in APR:
		result[key] = APR[key]
	
	return result


# ======================================================
# Example usage
# ======================================================

class imageDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, cv):
		"""
		initialize DatasetWrapper
		"""
		super(imageDataset, self).__init__()
		self.f_name = []
		mbs = [os.path.join(cv, mb) for mb in os.listdir(cv) if os.path.isdir(os.path.join(cv, mb))]
		for mb in mbs:
			imgs = [img for img in os.listdir(mb) if img.count('_') == 1]
			for img in imgs:
				files = [os.path.join(mb, file) for file in os.listdir(mb) if img.replace('.png','_') in file and ('label' in file or 'prop' in file)]
				files.sort()
				self.f_name.append(files)
		self.transformer = transforms.ToTensor()


	def __len__(self):
		# return size of dataset
		return len(self.f_name)



	def __getitem__(self, idx):

		try:
			mask_name, pred_mask = self.f_name[idx]
		except:
			print(self.f_name, idx, self.f_name[idx])
		mask = self.transformer(Image.open(mask_name))
		prop = self.transformer(Image.open(pred_mask))

		return mask, prop

def save_matric_from_path(path, read_from_history = True):

	
	if os.path.exists(os.path.join(path, 'matrix.json')) and read_from_history:
		with open(os.path.join(path, 'matrix.json'), 'r') as file:
			metrics = json.load(file)
		return metrics
	
	if os.path.isdir(os.path.join(path, '__pycache__')):
		shutil.rmtree(os.path.join(path, '__pycache__'))
	
	matrices = defaultdict(list)
	cv_iters = [os.path.join(path, cv) for cv in os.listdir(path) if os.path.isdir(os.path.join(path, cv))]
	for cv in tqdm(cv_iters, desc = path, leave = 0):
		gt_masks = torch.Tensor([])
		prob_masks = torch.Tensor([])
		dataloader = DataLoader(imageDataset(cv), 
						batch_size=400, 
						shuffle=True,
						num_workers=6,
						pin_memory=torch.cuda.is_available())
		for mask, prop in dataloader:
			gt_masks = torch.cat((gt_masks, mask), dim=0)
			prob_masks = torch.cat((prob_masks, prop), dim=0)
		
		# Step2: Object-level metrics (CPU)
		metrics = compute_segmentation_metrics(gt_masks, prob_masks)
		for key in metrics:
			matrices[key].append(float(metrics[key]))

	'''	
	matrix = {}	
	for key in matrices:
		matrix[key] = '{}±{} '.format(np.round(np.mean(matrices[key])*100, decimals=1), np.round(np.std(matrices[key])*100, decimals=1))
	'''
			
	with open(os.path.join(path, 'matrix.json'), 'w') as file:
		json.dump(matrices, file, indent=4)	
		
	return matrices


if __name__ == "__main__":
	path = '.'

	save_matric_from_path(path)
