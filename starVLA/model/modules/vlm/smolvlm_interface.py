from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image


class SmolVLMInterface(nn.Module):
	"""
	SmolVLA VLM adapter.

	This wrapper keeps a dedicated class entry for SmolVLA while reusing
	the existing VLM factory in StarVLA.
	"""

	def __init__(self, config=None, **kwargs):
		super().__init__()
		self.config = config
		framework_cfg = getattr(config, "framework", None)
		qwenvl_cfg = getattr(framework_cfg, "qwenvl", None) if framework_cfg is not None else None
		self.freeze_vision_encoder = bool(getattr(framework_cfg, "freeze_vision_encoder", True))
		self.train_expert_only = bool(getattr(framework_cfg, "train_expert_only", True))
		# SmolVLM2 default vision config is image_size=512, patch_size=16, scale_factor=4.
		# Keep image edge aligned to 64 to avoid pixel-shuffle shape mismatch.
		self.target_longest_edge = int(getattr(qwenvl_cfg, "vision_longest_edge", 512) or 512)
		self.required_alignment = int(getattr(qwenvl_cfg, "vision_alignment", 64) or 64)

		model_id = getattr(qwenvl_cfg, "base_vlm", None) if qwenvl_cfg is not None else None
		if not model_id:
			model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

		self.model = AutoModelForImageTextToText.from_pretrained(
			model_id,
			dtype=torch.bfloat16,
			_attn_implementation="flash_attention_2",
		)
		if torch.cuda.is_available():
			self.model = self.model.to("cuda")
		self.processor = AutoProcessor.from_pretrained(model_id)
		if hasattr(self.processor, "tokenizer"):
			self.processor.tokenizer.padding_side = "left"

		self._configure_lightweight_processor()
		self._freeze_vision_stack()

	def _configure_lightweight_processor(self):
		image_processor = getattr(self.processor, "image_processor", None)
		if image_processor is not None:
			if hasattr(image_processor, "do_image_splitting"):
				image_processor.do_image_splitting = False
			if hasattr(image_processor, "do_resize"):
				image_processor.do_resize = True
			if hasattr(image_processor, "size"):
				image_processor.size = {"longest_edge": self.target_longest_edge}
			if hasattr(image_processor, "max_image_size"):
				image_processor.max_image_size = {"longest_edge": self.target_longest_edge}

		video_processor = getattr(self.processor, "video_processor", None)
		if video_processor is not None:
			if hasattr(video_processor, "do_sample_frames"):
				video_processor.do_sample_frames = False
			if hasattr(video_processor, "num_frames"):
				video_processor.num_frames = 1

	def _freeze_vision_stack(self):
		if self.train_expert_only:
			for parameter in self.model.parameters():
				parameter.requires_grad = False
			self.model.eval()
			return

		if self.freeze_vision_encoder and hasattr(self.model, "vision_model"):
			for parameter in self.model.vision_model.parameters():
				parameter.requires_grad = False

		if hasattr(self.model, "multi_modal_projector"):
			for parameter in self.model.multi_modal_projector.parameters():
				parameter.requires_grad = False

	def _align_single_image(self, image_item):
		if not isinstance(image_item, Image.Image):
			return image_item

		width, height = image_item.size
		max_edge = max(width, height)

		target_edge = max(max_edge, self.target_longest_edge)
		target_edge = ((target_edge + self.required_alignment - 1) // self.required_alignment) * self.required_alignment

		if max_edge == target_edge and width % self.required_alignment == 0 and height % self.required_alignment == 0:
			return image_item

		scale = target_edge / float(max_edge)
		new_width = max(self.required_alignment, int(round(width * scale)))
		new_height = max(self.required_alignment, int(round(height * scale)))
		new_width = ((new_width + self.required_alignment - 1) // self.required_alignment) * self.required_alignment
		new_height = ((new_height + self.required_alignment - 1) // self.required_alignment) * self.required_alignment

		return image_item.resize((new_width, new_height), Image.BICUBIC)

	def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
		messages = []
		for image_items, instruction in zip(images, instructions):
			content = []
			for image_item in image_items:
				content.append({"type": "image", "image": self._align_single_image(image_item)})
			content.append({"type": "text", "text": instruction})
			messages.append([{"role": "user", "content": content}])

		inputs = self.processor.apply_chat_template(
			messages,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			processor_kwargs={"padding": True, "return_tensors": "pt"},
		)
		return inputs.to(self.model.device, dtype=torch.bfloat16)

	def build_inputs(self, images, instructions, solutions=None, **kwargs):
		return self.build_qwenvl_inputs(images=images, instructions=instructions, solutions=solutions, **kwargs)

	def embed_images(self, pixel_values):
		"""Embed images via get_image_features (handles vision_model + connector + reshaping).

		Returns a plain tensor [N, patches, hidden_dim] regardless of transformers version.
		Older versions wrap the result in BaseModelOutput and put the projected features in
		pooler_output; newer versions return a tensor directly or put it in last_hidden_state.
		"""
		output = self.model.model.get_image_features(pixel_values)
		if hasattr(output, "pooler_output") and output.pooler_output is not None:
			return output.pooler_output
		if hasattr(output, "last_hidden_state"):
			return output.last_hidden_state
		return output

	def embed_text(self, instructions):
		"""Tokenize text-only instructions (with chat template) and embed, returning (embeds, attention_mask)."""
		messages = []
		for instruction in instructions:
			messages.append([{"role": "user", "content": [{"type": "text", "text": instruction}]}])
		inputs = self.processor.apply_chat_template(
			messages,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			processor_kwargs={"padding": True, "return_tensors": "pt"},
		)
		input_ids = inputs["input_ids"].to(self.model.device)
		attention_mask = inputs["attention_mask"].to(self.model.device)
		text_embeds = self.model.model.text_model.embed_tokens(input_ids)
		return text_embeds, attention_mask

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		pixel_values=None,
		labels=None,
		image_grid_thw=None,
		inputs_embeds=None,
		past_key_values=None,
		use_cache=None,
		output_attentions=False,
		output_hidden_states=True,
		return_dict=True,
		**kwargs,
	):
		# Match current StarVLA precision behavior for multimodal backbones.
		with torch.autocast("cuda", dtype=torch.bfloat16):
			return self.model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				pixel_values=pixel_values,
				labels=labels,
				image_grid_thw=image_grid_thw,
				inputs_embeds=inputs_embeds,
				past_key_values=past_key_values,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				**kwargs,
			)
