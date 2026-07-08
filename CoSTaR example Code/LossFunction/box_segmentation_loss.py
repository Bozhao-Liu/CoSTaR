import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class BackgroundLoss(nn.Module):
	def forward(self, preds, targets):
		pred_prob = torch.sigmoid(preds)
		outside = (targets == 0).float()
		return (pred_prob * outside).mean()


class ForegroundLoss(nn.Module):
	def forward(self, preds, targets):
		pred_prob = torch.sigmoid(preds)
		inside = (targets > 0).float()
		return ((pred_prob - 1)**2 * inside).mean()


class EntropyLoss(nn.Module):
	def forward(self, preds, targets=None):
		pred_prob = torch.sigmoid(preds)
		entropy = - (pred_prob * torch.log(pred_prob + 1e-6) +
					 (1 - pred_prob) * torch.log(1 - pred_prob + 1e-6))
		return entropy.mean()
		
def dump_tensor(tensor, name_prefix):
	import os
	import uuid
	os.makedirs("dumped", exist_ok=True)

	filename = f"dumped/{name_prefix}_{uuid.uuid4().hex[:8]}.txt"

	# move to CPU and flatten
	data = tensor.detach().cpu().view(-1)

	with open(filename, "w") as f:
		for v in data:
			f.write(f"{v.item()}\n")

	print(f"[DUMPED] {name_prefix} -> {filename}")
	
def dump_image(tensor, name_prefix, uid=None):
	import os
	import uuid
	import torch
	import torchvision.utils as vutils

	os.makedirs("dumped", exist_ok=True)

	if uid is None:
		uid = uuid.uuid4().hex[:8]

	filename = f"dumped/{uid}_{name_prefix}.png"

	img = tensor.detach().cpu()

	# shape handling
	if img.dim() == 4:
		img = img[0]
	if img.dim() == 3 and img.shape[0] != 1:
		img = img[0:1]

	# normalize
	img = img.float()
	img_min, img_max = img.min(), img.max()
	if img_max > img_min:
		img = (img - img_min) / (img_max - img_min)
	else:
		img = torch.zeros_like(img)

	vutils.save_image(img, filename)

	print(f"[DUMPED IMG] {name_prefix} -> {filename}")
	
def dump_patch(tensor, name_prefix, size=10):
	import os
	import uuid
	os.makedirs("dumped", exist_ok=True)

	filename = f"dumped/{name_prefix}_patch_{uuid.uuid4().hex[:8]}.txt"

	t = tensor.detach().cpu()[0, 0, :size, :size]

	with open(filename, "w") as f:
		for row in t:
			f.write(" ".join([f"{v.item():.4f}" for v in row]) + "\n")

	print(f"[PATCH] {name_prefix} -> {filename}")
	
def log_txt(path, data_dict, step=None):
	import os
	os.makedirs(os.path.dirname(path), exist_ok=True)

	with open(path, "a") as f:
		if step is not None:
			f.write(f"[step {step}] ")

		line = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
						   for k, v in data_dict.items()])
		f.write(line + "\n")


class SoftBoxLoss:
	def __init__(self, loss_functions: dict, ema_decay = 0.95):
		self.loss_functions = loss_functions
		self.loss_functions.update({"BG": BackgroundLoss(), "FG": ForegroundLoss(),"Entropy": EntropyLoss()})
		self.loss_weights = {
								"BCE": 0.5,
								"Dice": 2.0,
								"BG": 0.3,
								"FG": 0.5,
								"Entropy": 0,
							}
		self.ema_decay = ema_decay
		
	@staticmethod
	def _to_logits(p, eps=1e-6): 
		#Reverse Sigmoid
		p = torch.clamp(p, eps, 1 - eps)
		return torch.log(p) - torch.log(1 - p)
	
	def __generate_teacher__(self, avg_preds, boxes_mask, current_epoch, weight_map): 
		#create teacher base on Exponential Moving Average (EMA) 
		assert 0 <= self.ema_decay <= 1, \
			f'EMA decay out of range: {self.ema_decay}' 
		#confidence of the current prediction is the same as the ground truth, start with 1 - 0.99**0 = 0, meaning 0 confident the first prediction is the same as ground truth 
		#confidence ramps up over time 
		confidence = min(0.75, max(0.1, 1 - self.ema_decay**current_epoch))
		
		assert boxes_mask.shape == avg_preds.shape, \
			'Pseudo_labels shape does not match prediction, it should be {}, but got {} instead'.format(boxes_mask.shape, avg_preds.shape) 

		teacher = (1 - confidence) * boxes_mask + confidence * ( weight_map * avg_preds + (1 - weight_map) * boxes_mask )
		teacher = teacher.clamp(0, 1)
		
		return teacher

	def __create_pseudo_labels__(self, preds, boxes_mask, current_epoch, avg_preds=None, uncertainty=None):
		"""
		Create pseudo-labels and per-pixel weight maps using a teacher–student
		update mechanism with epoch-based confidence ramp-up.

		Parameters
		----------
		preds : torch.Tensor
			Model logits of shape (B, C, H, W). Used for shape consistency checks.

		boxes_mask : torch.Tensor
			Binary bounding-box mask of shape (B, C, H, W). Defines the region
			where labels are valid. Pixels outside the bounding box are forced to 0.

		current_epoch : int
			Current epoch index used in the exponential moving average (EMA)
			confidence update rule.

		avg_preds : torch.Tensor
			Averaged predictions from teacher model or augmentation ensemble,
			of shape (B, C, H, W). Provides soft supervision to guide pseudo-labels.

		uncertainty : torch.Tensor, optional
			Pixel-wise uncertainty map of shape (B, C, H, W).
			Values are normalized to [0, 1] after masking outside bounding boxes.
			Higher uncertainty leads to lower weighting in `weight_map`.

		Returns
		-------
		pseudo_labels : torch.Tensor
			Pseudo-label tensor of shape (B, C, H, W), values in [0, 1].
			Computed as the confidence-weighted blend of teacher and box mask:
				teacher = confidence * avg_preds + (1 - confidence) * boxes_mask
				pseudo_labels = teacher * boxes_mask

		weight_map : torch.Tensor
			Per-pixel weighting map of shape (B, C, H, W), values in [0, 1].
			If `uncertainty` is provided, the weights are computed as
				weight_map = 1 - normalized(uncertainty)
			otherwise set to all ones.

		Notes
		-----
		- All tensors must have identical shapes (B, C, H, W) and be finite.
		- The function ensures that all intermediate results are clamped to [0, 1]
		  for stability and interpretability.
		- Normalization of uncertainty prevents division by zero via a small epsilon (1e-6).
		"""
		weight_map = torch.ones_like(preds, device=preds.device, dtype=preds.dtype) 
		
		if uncertainty is not None: #and current_epoch > 5: 
			assert uncertainty.shape == boxes_mask.shape, f"Uncertainty shape mismatch: {uncertainty.shape}, preds {boxes_mask.shape}" 
			assert not uncertainty.isnan().any(), f'uncertainty is nan: {uncertainty}'
			uncertainty = uncertainty * (boxes_mask>0)  # uncertainty outside bounding box is zero
			u_in = uncertainty[boxes_mask.bool()]
			if u_in.numel() > 0:
				lo = torch.quantile(u_in, 0.05)
				hi = torch.quantile(u_in, 0.95)
				u = (uncertainty - lo) / (hi - lo + 1e-6)
				u = u.clamp(0, 1)
				u = u * (boxes_mask > 0)   # only normalize inside valid region
			else:
				u = torch.zeros_like(uncertainty)
				
			weight_map = 1-u
			
		teacher = self.__generate_teacher__(avg_preds, boxes_mask, current_epoch, weight_map)
		# -------------------------
		# Soft fallback using coverage
		# cover edge case where the average prediction mostly miss the ROI
		# -------------------------
		tau = 0.25

		# compute coverage inside box (per sample)
		coverage = (avg_preds * (boxes_mask > 0)).sum(dim=(2,3), keepdim=True) / \
				   (boxes_mask.sum(dim=(2,3), keepdim=True) + 1e-6)

		# compute miss strength
		miss_strength = ((tau - coverage) / tau).clamp(0, 1) ** 2

		# broadcast to spatial size
		miss_strength = miss_strength.expand_as(teacher)

		# soft blend teacher with box
		teacher = (1 - miss_strength) * teacher + miss_strength * boxes_mask
		
		pseudo_labels = (teacher * (boxes_mask > 0.7)).clamp(0, 1).float()  # clamp targets
				
		assert boxes_mask.shape == pseudo_labels.shape, \
			'Pseudo_labels shape does not match boxes_mask, it should be {}, but got {} instead'.format(boxes_mask.shape, pseudo_labels.shape) 
			
		return pseudo_labels, weight_map

	def __call__(self, preds, boxes_mask, current_epoch, avg_preds=None, uncertainty=None):
		"""
		Compute the total loss using a combination of user-defined loss functions,
		optionally guided by pseudo-labels and uncertainty weighting.

		Parameters
		----------
		preds : torch.Tensor
			Model logits of shape (B, C, H, W).
			These are raw (pre-sigmoid) outputs from the network.

		boxes_mask : torch.Tensor
			Binary bounding-box mask of shape (B, C, H, W), values in [0, 1].
			Represents the weak supervision region—pixels outside bounding boxes
			are treated as background and masked out.

		current_epoch : int
			Current training epoch index, used to control confidence ramp-up
			in the teacher update rule.

		avg_preds : torch.Tensor, optional
			Mean prediction map from teacher model or test-time augmentation (TTA),
			same shape as `preds` (B, C, H, W).
			Serves as a soft teacher signal for generating pseudo-labels.

		uncertainty : torch.Tensor, optional
			Per-pixel uncertainty map, same shape as `preds` (B, C, H, W).
			Used to reduce the contribution of unreliable regions during loss
			computation. Typically derived from prediction variance across
			augmentations or ensembles.

		Returns
		-------
		total_loss : torch.Tensor
			Scalar tensor — the weighted sum of all individual loss components
			defined in `self.loss_functions`.

		Notes
		-----
		- All input tensors (`preds`, `boxes_mask`, `avg_preds`, `uncertainty`)
		  must be on the same device and have identical spatial dimensions.
		- If `avg_preds` is `None`, the loss is computed directly against the
		  bounding-box mask (weak supervision phase).
		- Pseudo-labels are thresholded to binary form before loss computation.
		- `weight_map` modulates per-pixel contribution; defaults to 1 if
		  `uncertainty` is not provided.
		- The method performs NaN/Inf safety checks before and after weighting
		  to prevent numerical instability.
		"""
		boxes_mask = boxes_mask.clamp(0, 1)
	
		if avg_preds is not None:
			pseudo_labels, weight_map = self.__create_pseudo_labels__(preds, boxes_mask, current_epoch, avg_preds, uncertainty)
		else:
			pseudo_labels = boxes_mask.clamp(0, 1)
			weight_map = torch.ones(preds.shape, device=preds.device, dtype=preds.dtype)

		total_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
		pseudo_labels = pseudo_labels.to(device=preds.device, dtype=preds.dtype)

		for name, loss_fn in self.loss_functions.items():
			
			if name in ["BG", "FG"]:
				loss_value = loss_fn(preds, pseudo_labels)

			elif name == "Entropy":
				loss_value = loss_fn(preds)
				
			elif isinstance(loss_fn, nn.BCEWithLogitsLoss) and loss_fn.reduction == "none":
				loss_value = (loss_fn(preds, pseudo_labels) * weight_map).mean()			 # (B,C,H,W)
				
			elif name in ["FECE", "ECEF"]:
				loss_value = (loss_fn(preds, pseudo_labels) * weight_map).mean()
			
			else:
				loss_value = loss_fn(preds, pseudo_labels)
			'''	
			log_txt(
					f"debug/epoch_{current_epoch}_train.txt",
					{name: float(loss_value)},
					)'''

			if not torch.isfinite(loss_value).all():
				raise RuntimeError(f'Loss {name} produced NaN/Inf.')

			weight = self.loss_weights.get(name, 1.0)
			total_loss = total_loss + weight * loss_value

		return total_loss
		
class ContrastiveLoss(nn.Module):
	def __init__(
		self,
		fg_thresh=0.7,
		bg_thresh=0.3,
		var_thresh=0.2,
		min_pixels=20,
		temperature=0.1
	):
		super().__init__()

		self.fg_thresh = fg_thresh
		self.bg_thresh = bg_thresh
		self.var_thresh = var_thresh
		self.min_pixels = min_pixels
		self.temperature = temperature

	def forward(self, output, pseudo, variance, masks):
		"""
		Args:
			output:   (B*A, C, H, W) logits
			pseudo:   (B*A, C, H, W)
			variance: (B*A, C, H, W)
			masks:	(B*A, C, H, W)
		"""

		B, C, H, W = output.shape
		Preds = torch.sigmoid(output)  # (B*A,1,H,W)

		# flatten
		feat = Preds.view(B, -1)		  # (B, HW)
		pseudo = pseudo.view(B, -1)
		variance = variance.view(B, -1)
		masks = masks.view(B, -1)

		total_loss = 0.0
		valid_batches = 0

		for b in range(B):

			f = feat[b]
			p = pseudo[b]
			v = variance[b]
			m = masks[b]

			# -----------------------------
			# 2. define regions
			# -----------------------------
			fg = (p > self.fg_thresh) & (v < self.var_thresh) & (m > 0)
			bg = (p < self.bg_thresh) & (v < self.var_thresh)

			if fg.sum() < self.min_pixels or bg.sum() < self.min_pixels:
				continue

			fg_feat = f[fg]
			bg_feat = f[bg]

			# -----------------------------
			# 3. normalize (important)
			# -----------------------------
			fg_feat = F.normalize(fg_feat.unsqueeze(1), dim=0).squeeze(1)
			bg_feat = F.normalize(bg_feat.unsqueeze(1), dim=0).squeeze(1)

			# -----------------------------
			# 4. prototypes
			# -----------------------------
			fg_proto = fg_feat.mean()
			bg_proto = bg_feat.mean()

			# -----------------------------
			# 5. contrastive loss (margin form)
			# -----------------------------
			pos_fg = (fg_feat - fg_proto) ** 2
			neg_fg = (fg_feat - bg_proto) ** 2

			pos_bg = (bg_feat - bg_proto) ** 2
			neg_bg = (bg_feat - fg_proto) ** 2

			loss_fg = pos_fg.mean() - neg_fg.mean()
			loss_bg = pos_bg.mean() - neg_bg.mean()

			loss = F.relu(loss_fg + loss_bg)

			total_loss += loss
			valid_batches += 1

		if valid_batches == 0:
			return torch.zeros((), device=output.device, dtype=output.dtype)

		return total_loss / valid_batches

if __name__ == "__main__":

	import torch
	import torch.nn as nn

	torch.manual_seed(0)

	B, C, H, W = 2, 1, 128, 128

	boxes_mask = torch.zeros(B, C, H, W)
	boxes_mask[:, :, 32:96, 32:96] = 1.0

	avg_preds = torch.sigmoid(torch.randn(B, C, H, W))

	# -----------------------------
	# Define Dice
	# -----------------------------
	class DiceLoss(nn.Module):
		def forward(self, preds, targets, mask=None):
			p = torch.sigmoid(preds)

			if mask is not None:
				p = p * mask
				targets = targets * mask

			inter = (p * targets).sum(dim=(2,3))
			union = p.sum(dim=(2,3)) + targets.sum(dim=(2,3))

			dice = (2 * inter + 1) / (union + 1)
			return 1 - dice.mean()

	bce = nn.BCEWithLogitsLoss(reduction="none")
	dice = DiceLoss()

	# -----------------------------
	# Helper
	# -----------------------------
	def test_case(name, preds, pseudo):
		preds = preds.clone().detach().requires_grad_(True)

		# BCE
		bce_loss = bce(preds, pseudo)
		bce_loss = bce_loss.mean()
		bce_loss.backward(retain_graph=True)
		bce_grad = preds.grad.abs().mean().item()

		preds.grad.zero_()

		# Dice
		dice_loss = dice(preds, pseudo)
		dice_loss.backward()
		dice_grad = preds.grad.abs().mean().item()

		with torch.no_grad():
			p = torch.sigmoid(preds)
			inside = boxes_mask.bool()
			outside = ~inside

			print(f"\n=== {name} ===")
			print({
				"bce_loss": bce_loss.item(),
				"bce_grad": bce_grad,
				"dice_loss": dice_loss.item(),
				"dice_grad": dice_grad,
				"inside_mean": p[inside].mean().item(),
				"outside_mean": p[outside].mean().item(),
			})

	# -----------------------------
	# Case 1: random
	# -----------------------------
	preds = torch.randn(B, C, H, W)
	pseudo = boxes_mask.clone()

	test_case("Random", preds, pseudo)

	# -----------------------------
	# Case 2: perfect box
	# -----------------------------
	preds = torch.zeros(B, C, H, W)
	preds[boxes_mask == 1] = 2.0
	preds[boxes_mask == 0] = -2.0

	pseudo = boxes_mask.clone()

	test_case("Perfect Box", preds, pseudo)

	# -----------------------------
	# Case 3: soft pseudo
	# -----------------------------
	preds = torch.zeros(B, C, H, W)
	preds[boxes_mask == 1] = 2.0
	preds[boxes_mask == 0] = -2.0

	pseudo = avg_preds * boxes_mask

	test_case("Perfect Box + Soft Target", preds, pseudo)

	print("\nIsolation sanity completed.\n")