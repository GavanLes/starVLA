import torch
import torch.nn as nn

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model


class SmolVLAFlowMatching(nn.Module):
	"""
	SmolVLA action head wrapper.

	Uses the existing layer-wise flow-matching implementation as a stable
	baseline so SmolVLA can run end-to-end in StarVLA.
	"""

	def __init__(self, config=None, **kwargs):
		super().__init__()
		self.config = config
		self.model = get_action_model(config=config)

	def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None):
		return self.model(vl_embs_list=vl_embs_list, actions=actions, state=state)

	@torch.inference_mode()
	def predict_action(self, vl_embs_list: list, state: torch.Tensor = None):
		return self.model.predict_action(vl_embs_list=vl_embs_list, state=state)
