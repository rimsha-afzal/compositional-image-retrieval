"""Build prompt-ensemble CLIP text embeddings for CelebA attributes."""

from pathlib import Path

import torch
from tqdm import tqdm

from embedding_extraction import CELEBA_ATTRIBUTES, attribute_to_prompt, load_clip, _as_tensor
from embedding_io import l2_normalize, save_embeddings


POSITIVE_CARRIERS = [
    "a photo of a person {phrase}",
    "a close-up photo of a person {phrase}",
    "a headshot of a person {phrase}",
    "a portrait of someone {phrase}",
    "a cropped photo of a person {phrase}",
    "a good photo of a person {phrase}",
    "a bad photo of a person {phrase}",
    "a photo of the face of a person {phrase}",
    "a photo of a celebrity {phrase}",
    "a high quality photo of a person {phrase}",
    "a low quality photo of a person {phrase}",
    "an image of a person {phrase}",
]


ATTRIBUTE_PHRASES = {
    "5_o_Clock_Shadow": "with a five o'clock shadow",
    "Arched_Eyebrows": "with arched eyebrows",
    "Attractive": "who is attractive",
    "Bags_Under_Eyes": "with bags under their eyes",
    "Bald": "who is bald",
    "Bangs": "with bangs",
    "Big_Lips": "with big lips",
    "Big_Nose": "with a big nose",
    "Black_Hair": "with black hair",
    "Blond_Hair": "with blond hair",
    "Blurry": "who looks blurry",
    "Brown_Hair": "with brown hair",
    "Bushy_Eyebrows": "with bushy eyebrows",
    "Chubby": "who is chubby",
    "Double_Chin": "with a double chin",
    "Eyeglasses": "with eyeglasses",
    "Goatee": "with a goatee",
    "Gray_Hair": "with gray hair",
    "Heavy_Makeup": "wearing heavy makeup",
    "High_Cheekbones": "with high cheekbones",
    "Male": "who is a man",
    "Mouth_Slightly_Open": "with their mouth slightly open",
    "Mustache": "with a mustache",
    "Narrow_Eyes": "with narrow eyes",
    "No_Beard": "who is smooth-faced",
    "Oval_Face": "with an oval face",
    "Pale_Skin": "with pale skin",
    "Pointy_Nose": "with a pointy nose",
    "Receding_Hairline": "with a receding hairline",
    "Rosy_Cheeks": "with rosy cheeks",
    "Sideburns": "with sideburns",
    "Smiling": "who is smiling",
    "Straight_Hair": "with straight hair",
    "Wavy_Hair": "with wavy hair",
    "Wearing_Earrings": "wearing earrings",
    "Wearing_Hat": "wearing a hat",
    "Wearing_Lipstick": "wearing lipstick",
    "Wearing_Necklace": "wearing a necklace",
    "Wearing_Necktie": "wearing a necktie",
    "Young": "who is young",
}


def build_positive_prompts(attribute: str) -> list[str]:
    """Return the positive prompt ensemble for one CelebA attribute."""
    phrase = ATTRIBUTE_PHRASES[attribute]
    return [carrier.format(phrase=phrase) for carrier in POSITIVE_CARRIERS]


def encode_prompts(
    prompts: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode prompt strings with CLIP and L2-normalize each row."""
    model, processor, resolved_device = load_clip(model_name, device)
    embeddings = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompt ensemble"):
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
    return l2_normalize(torch.cat(embeddings, dim=0))


def compute_prompt_ensemble_embeddings(
    model_name: str,
    batch_size: int = 64,
    device: str = "cpu",
    attribute_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Average positive prompt embeddings per attribute and return baseline-compatible IDs."""
    attributes = attribute_names or CELEBA_ATTRIBUTES
    all_prompts = []
    spans = []
    for attribute in attributes:
        start = len(all_prompts)
        prompts = build_positive_prompts(attribute)
        all_prompts.extend(prompts)
        spans.append((start, len(all_prompts)))

    prompt_embeddings = encode_prompts(
        all_prompts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )

    ensemble_rows = []
    for start, end in spans:
        ensemble_rows.append(l2_normalize(prompt_embeddings[start:end].mean(dim=0)))

    ids = [attribute_to_prompt(attribute) for attribute in attributes]
    return torch.stack(ensemble_rows), ids


def build_and_save_prompt_ensemble_text_embeddings(
    output_dir: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
    device: str = "cpu",
    filename: str = "prompt_ensemble_text_embeddings",
) -> tuple[torch.Tensor, list[str]]:
    """Compute and save prompt-ensemble text embeddings."""
    embeddings, ids = compute_prompt_ensemble_embeddings(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    save_embeddings(embeddings, ids, output_dir, filename)
    return embeddings, ids
