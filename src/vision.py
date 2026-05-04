from __future__ import annotations

from pathlib import Path


def analyze_keyframes(
    keyframes: list[dict],
    model_name: str,
    device: str,
    prompt_profile: str,
) -> list[dict]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for visual analysis. Install optional vision dependencies."
        ) from exc

    use_florence = "florence" in model_name.lower()
    pipe = None
    processor = None
    model = None
    runtime_device = (
        "cpu" if device == "cpu" or not torch.cuda.is_available() else "cuda"
    )

    if use_florence:
        dtype = torch.float16 if runtime_device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(runtime_device)
    else:
        pipe_device = -1 if runtime_device == "cpu" else 0
        pipe = pipeline("image-to-text", model=model_name, device=pipe_device)

    caption_task = "<MORE_DETAILED_CAPTION>"
    ocr_task = "<OCR>"
    if prompt_profile == "meeting":
        caption_task = "<DETAILED_CAPTION>"
    elif prompt_profile == "code-demo":
        caption_task = "<MORE_DETAILED_CAPTION>"

    out: list[dict] = []
    for item in keyframes:
        frame_path = Path(item["frame_path"])
        caption = ""
        ocr_text = ""

        if use_florence:
            image = Image.open(frame_path).convert("RGB")

            cap_inputs = processor(
                text=caption_task,
                images=image,
                return_tensors="pt",
            )
            cap_inputs = {
                k: v.to(runtime_device) if hasattr(v, "to") else v
                for k, v in cap_inputs.items()
            }
            if "pixel_values" in cap_inputs:
                cap_inputs["pixel_values"] = cap_inputs["pixel_values"].to(dtype)
            cap_output = model.generate(
                input_ids=cap_inputs.get("input_ids"),
                pixel_values=cap_inputs.get("pixel_values"),
                max_new_tokens=192,
                num_beams=3,
            )
            cap_raw = processor.batch_decode(cap_output, skip_special_tokens=False)[0]
            if hasattr(processor, "post_process_generation"):
                parsed = processor.post_process_generation(
                    cap_raw,
                    task=caption_task,
                    image_size=image.size,
                )
                if isinstance(parsed, dict):
                    caption = str(parsed.get(caption_task, cap_raw)).strip()
                else:
                    caption = str(parsed).strip()
            else:
                caption = cap_raw.strip()

            ocr_inputs = processor(
                text=ocr_task,
                images=image,
                return_tensors="pt",
            )
            ocr_inputs = {
                k: v.to(runtime_device) if hasattr(v, "to") else v
                for k, v in ocr_inputs.items()
            }
            if "pixel_values" in ocr_inputs:
                ocr_inputs["pixel_values"] = ocr_inputs["pixel_values"].to(dtype)
            ocr_output = model.generate(
                input_ids=ocr_inputs.get("input_ids"),
                pixel_values=ocr_inputs.get("pixel_values"),
                max_new_tokens=192,
                num_beams=2,
            )
            ocr_raw = processor.batch_decode(ocr_output, skip_special_tokens=False)[0]
            if hasattr(processor, "post_process_generation"):
                parsed = processor.post_process_generation(
                    ocr_raw,
                    task=ocr_task,
                    image_size=image.size,
                )
                if isinstance(parsed, dict):
                    ocr_text = str(parsed.get(ocr_task, ocr_raw)).strip()
                else:
                    ocr_text = str(parsed).strip()
            else:
                ocr_text = ocr_raw.strip()
        else:
            result = pipe(str(frame_path))
            if result and isinstance(result, list):
                caption = str(result[0].get("generated_text", "")).strip()

        out.append(
            {
                "timestamp": float(item["timestamp"]),
                "frame_path": str(frame_path),
                "caption": caption,
                "ocr_text": ocr_text,
                "tags": [prompt_profile],
            }
        )
    return out
