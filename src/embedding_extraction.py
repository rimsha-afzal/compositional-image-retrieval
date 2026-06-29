"""Extract CLIP embeddings for CelebA images and attribute prompts.

The saved vectors let retrieval code reuse CLIP features without running the
model again for every experiment.
"""

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from data_loading import load_celeba_split
from embedding_io import l2_normalize, save_embeddings


CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
    "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
    "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
    "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
    "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
    "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def attribute_to_prompt(attribute: str) -> str:
    """Convert a CelebA attribute name into a CLIP text prompt."""
    return f"a photo of a person who is {attribute.replace('_', ' ').lower()}"


def load_clip(
    model_name: str,
    device: str,
) -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load the CLIP model and processor on the requested device."""
    resolved_device = torch.device(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return model, processor, resolved_device


def _as_tensor(features) -> torch.Tensor:
    """Return the tensor inside a CLIP output object, if needed."""
    return getattr(features, "pooler_output", features)


def extract_image_embeddings(
    data_root: str | Path,
    split: str,
    model_name: str,
    output_dir: str | Path,
    batch_size: int = 64,
    device: str = "cpu",
    max_images: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Encode one CelebA split as normalized CLIP image embeddings."""
    model, processor, resolved_device = load_clip(model_name, device)
    dataset = load_celeba_split(data_root=data_root, split=split, target_type="attr")
    limit = len(dataset) if max_images is None else min(max_images, len(dataset))

    embeddings = []
    ids = []
    for start in tqdm(range(0, limit, batch_size), desc=f"Encoding {split} images"):
        batch_indices = list(range(start, min(start + batch_size, limit)))
        images = [dataset[index][0].convert("RGB") for index in batch_indices]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(
            resolved_device
        )
        with torch.no_grad():
            features = _as_tensor(model.get_image_features(**inputs))
        embeddings.append(features.cpu())
        ids.extend(batch_indices)

    image_embeddings = l2_normalize(torch.cat(embeddings, dim=0))
    save_embeddings(image_embeddings, ids, Path(output_dir) / split, "image_embeddings")
    return image_embeddings, ids


def extract_text_embeddings(
    model_name: str,
    output_dir: str | Path,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Encode all CelebA attributes as normalized CLIP text embeddings."""
    model, processor, resolved_device = load_clip(model_name, device)
    prompts = [attribute_to_prompt(attribute) for attribute in CELEBA_ATTRIBUTES]

    embeddings = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding text prompts"):
        batch = prompts[start:start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(resolved_device)
        with torch.no_grad():
            features = _as_tensor(model.get_text_features(**inputs))
        embeddings.append(features.cpu())

    text_embeddings = l2_normalize(torch.cat(embeddings, dim=0))
    save_embeddings(text_embeddings, prompts, output_dir, "text_embeddings")
    return text_embeddings, prompts
