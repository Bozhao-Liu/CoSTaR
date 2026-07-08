import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedDiceLoss(nn.Module):
	def __init__(self, smooth=1.0):
		super().__init__()
		self.smooth = smooth
		
	def forward(self, preds, targets, mask=None):
		p = torch.sigmoid(preds)
		t = targets
		if mask is not None:
			p = p * mask
			t = t * mask
		inter = (p * t).sum((2,3))
		union = p.sum((2,3)) + t.sum((2,3))
		dice = (2*inter + self.smooth) / (union + self.smooth)

		return 1 - dice.mean()

class MaskedTverskyLoss(nn.Module):
	def __init__(self, alpha=0.3, beta=0.7, smooth=1.0): # beta>alpha ⇒ penalize FN
		super().__init__()
		self.alpha, self.beta, self.smooth = alpha, beta, smooth
		
	def forward(self, preds, targets, mask=None):
		p = torch.sigmoid(preds)
		t = targets
		if mask is None:
			mask = torch.ones_like(p)
			p = p * mask
			t = targets * mask
		TP = (p*t).sum((2,3))
		FP = (p*(1-t)).sum((2,3))
		FN = ((1-p)*t).sum((2,3))
		tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
		return 1 - tversky.mean()
		
class ContrastiveDiceLoss(nn.Module):
	def __init__(self, smooth=1.0, fg_weight=1.0, bg_weight=1.0):
		super().__init__()
		self.smooth = smooth
		self.fg_weight = fg_weight
		self.bg_weight = bg_weight

	def forward(self, preds, targets, mask=None):
		"""
		preds: logits (B,C,H,W)
		targets: (B,C,H,W) in [0,1]
		mask: optional box mask
		"""

		p = torch.sigmoid(preds)
		t = targets

		if mask is not None:
			p = p * mask
			t = t * mask

		# -----------------------------
		# Foreground Dice
		# -----------------------------
		inter_fg = (p * t).sum(dim=(2,3))
		union_fg = p.sum(dim=(2,3)) + t.sum(dim=(2,3))

		dice_fg = (2 * inter_fg + self.smooth) / (union_fg + self.smooth)

		# -----------------------------
		# Background Dice (critical)
		# -----------------------------
		p_bg = 1 - p
		t_bg = 1 - t

		inter_bg = (p_bg * t_bg).sum(dim=(2,3))
		union_bg = p_bg.sum(dim=(2,3)) + t_bg.sum(dim=(2,3))

		dice_bg = (2 * inter_bg + self.smooth) / (union_bg + self.smooth)

		# -----------------------------
		# Combined
		# -----------------------------
		#dice = self.fg_weight * dice_fg + self.bg_weight * dice_bg
		dice = (self.fg_weight * dice_fg + self.bg_weight * dice_bg) / (self.fg_weight + self.bg_weight) #normalized

		return 1 - dice.mean()
		
class DiceLoss(nn.Module):
	def __init__(self, smooth=1e-6):
		"""
		Dice Loss for segmentation tasks.

		Args:
			smooth (float): Smoothing constant to avoid division by zero.
		"""
		super(DiceLoss, self).__init__()
		self.smooth = smooth

	def forward(self, inputs, targets):
		"""
		Compute Dice loss.

		Args:
			inputs (Tensor): Predicted logits or probabilities (N, C, H, W).
			targets (Tensor): Ground truth binary mask (N, C, H, W).

		Returns:
			Tensor: Dice loss (scalar if reduction != 'none').
		"""
		# Apply sigmoid if inputs are raw logits (for binary segmentation)
		inputs = torch.sigmoid(inputs)

		# Flatten per sample
		inputs = inputs.contiguous().view(inputs.shape[0], -1)
		targets = targets.contiguous().view(targets.shape[0], -1)

		intersection = (inputs * targets).sum(dim=1)
		union = inputs.sum(dim=1) + targets.sum(dim=1)

		dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
		return 1 - dice_score.mean()


if __name__ == "__main__":
	criterion = DiceLoss()  # or "sum", "none"

	preds = torch.rand(4, 1, 128, 128) # logits
	targets = torch.randint(0, 2, (4, 1, 128, 128)).float()

	loss = criterion(preds, targets)
	print(preds, targets, loss)
	print('{}'.format((1,2,3)))

