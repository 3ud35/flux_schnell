import torch
from diffusers import FluxPipeline
from datetime import datetime

model_id = "black-forest-labs/FLUX.1-schnell"  # you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = input("Prompt:")
if len(prompt) < 1:
    prompt = "A cat holding a sign that says hello world"

seed = 42
# seed = 0
image = pipe(
    prompt,
    # height=1000,
    # width=1000,
    # guidance_scale=3.5,
    # max_sequence_length=512,
    output_type="pil",
    num_inference_steps=4,  # use a larger number if you are using [dev]
    # num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(seed),
).images[0]


image_file_path = f"./images/{str(datetime.now())}_flux-schnell_{prompt}.png"
image.save(image_file_path)
