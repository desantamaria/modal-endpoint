import io
import random
import time
from pathlib import Path

import modal

MINUTES = 60

app = modal.App("stable-diffusion")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster downloads
)

with image.imports():
    import diffusers
    import torch
    from fastapi import Response

model_id = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
model_revision_id = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"


@app.cls(
    image=image,
    gpu="A10G",
    timeout=10 * MINUTES,
)
class Inference:
    @modal.build()
    @modal.enter()
    def initialize(self):
        self.pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            model_id,
            revision=model_revision_id,
            torch_dtype=torch.bfloat16,
        )

    @modal.enter()
    def move_to_gpu(self):
        self.pipe.to("cuda")

    @modal.method()
    def run(
        self, prompt: str, batch_size: int = 4, seed: int = None
    ) -> list[bytes]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print("seeding RNG with", seed)
        torch.manual_seed(seed)
        images = self.pipe(
            prompt,
            num_images_per_prompt=batch_size,  # outputting multiple images per prompt is much cheaper than separate calls
            num_inference_steps=4,  # turbo is tuned to run in four steps
            guidance_scale=0.0,  # turbo doesn't use CFG
            max_sequence_length=512,  # T5-XXL text encoder supports longer sequences, more complex prompts
        ).images

        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="JPEG")
                image_output.append(buf.getvalue())
        torch.cuda.empty_cache()  # reduce fragmentation
        return image_output

    @modal.web_endpoint(docs=True)
    def web(self, prompt: str, seed: int = None):
        return Response(
            content=self.run.local(  # run in the same container
                prompt, batch_size=1, seed=seed
            )[0],
            media_type="image/jpeg",
        )