from PIL import Image, ImageChops

import tempfile
import json

from models.mask.SegformerB2Clothes import SegformerB2Clothes
from models.inpaint.DiffusionInpainter import DiffusionInpainter
from models.enhance.CodeFormer import CodeFormer
from utils.image_utils import compose_mask
from config import Config

import gradio as gr

SETTINGS = {
    "theme": "light",

    "magic_prompt": Config.MAGIC_PROMPT,

    "masking_model": Config.SEGFORMER_MASKER_DEFAULT_MODEL,
    "masks_all": ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
    "masks": SegformerB2Clothes.CLOTHING_LABELS,
    "mask_expansion": 0,

    "negative_prompt": "unnatural transitions, unsmooth transision, incorrect colors, inconsistent color, color missmatch",
    "resolution": 512,
    "inpainting_model": Config.DIFFUSION_INPAINTER_DEFAULT_MODEL,
    "guidance_scale": 10.,
    "strength": 0.5,
    "num_inference_steps": 75,
    "num_images": 4,
    "mask_blur": 2,
    "use_safety_checker": Config.USE_SAFETY_CHECKER,
    "load_in_4bit": Config.LOAD_IN_4BIT,

    "enhance_background": True,
    "face_upsample": True,
    "draw_box": False,
    "has_aligned": False,
    "upscale_value": 4,
    "fidelity": 0.5,

    "lora_modules": ""
}

def generate_mask(image: Image.Image, masking_model: str, masks: list[str], mask_expansion: int, state: dict, progress=gr.Progress(track_tqdm=True)):
    # Load masking model
    progress(0.1, "Loading masking model...")
    masker = SegformerB2Clothes(masking_model)

    # Generate mask
    progress(0.5, "Generating mask...")
    mask = masker(image, masks, mask_expansion)

    state["image"] = image
    state["mask"] = mask

    return {
        "background": image,
        "layers": [ image ],
        "composite": compose_mask(image, mask)
    }, state

def handle_mask_draw(mask_preview, state: dict):
    if "image" not in state.keys():
        return state

    image = state["image"]
    current_mask = state.get("mask", Image.new("L", image.size)).convert("L")
    if mask_preview is None or "layers" not in mask_preview or not mask_preview["layers"]:
        new_mask = Image.new("L", image.size)
    else:
        new_mask = mask_preview["layers"][0].convert("L")
    
    current_mask_bin = current_mask.point(lambda p: 255 if p > 0 else 0)
    new_mask_bin = new_mask.point(lambda p: 255 if p > 0 else 0)

    combined_mask = ImageChops.lighter(current_mask_bin, new_mask_bin)
    state["mask"] = combined_mask
    
    return state

def clear_mask(state: dict):
    if "image" not in state.keys():
        return gr.update(), state

    image = state["image"]
    mask = Image.new("L", image.size).convert("L")

    state["mask"] = mask

    return compose_mask(image, mask), state

def run(image: Image.Image, prompt: str, negative_prompt: str,
        inpaint_model: str,
        use_safety_checker: bool, load_in_4bit: bool,
        enhance_background: bool, face_upsample: bool, draw_box: bool, has_aligned: bool, upscale_value: int, fidelity: float,
        resolution: int, guidance_scale: float, strength: float, num_inference_steps: int, num_images_per_prompt: int, mask_blur: int,
        lora_modules: str,
        state: dict, progress=gr.Progress(track_tqdm=True)) -> list[Image.Image]:
    if image is None:
        raise FileNotFoundError("Please pass an image.")

    print(f"""
    Starting generation:
        - inpaint_model: {inpaint_model}
        - use_safety_checker: {use_safety_checker}
        - load_in_4bit: {load_in_4bit}
        - enhance_background: {enhance_background}
        - face_upsample: {face_upsample}
        - draw_box: {draw_box}
        - has_aligned: {has_aligned}
        - upscale_value: {upscale_value}
        - fidelity: {fidelity}
        - resolution: {resolution}
        - guidance_scale: {guidance_scale}
        - strength: {strength}
        - num_inference_steps: {num_inference_steps}
        - num_images_per_prompt: {num_images_per_prompt}
        - mask_blur: {mask_blur}
    """)

    # Load models
    progress(0, desc="Preparing models...")
    model = DiffusionInpainter(inpaint_model, load_in_4bit, use_safety_checker)
    enhancer = CodeFormer()

    if lora_modules:
        progress(0.1, desc="Loading LoRa modules...")
        loras: list[str] = lora_modules.replace(", ", ",").split(",")
        for lora_module in loras:
            model._load_lora_weights(lora_module)

    progress(0.2, desc="Loading mask...")
    mask = state.get("mask", None)
    if mask is None:
        raise ValueError("Mask went missing! Please press 'Recalculate mask' or redraw your own.")

    progress(0.4, desc="Starting inpainting...")

    # Generate with inpaint
    images = model(
        prompt + SETTINGS["magic_prompt"], image, mask, resolution,
        reference_image=image,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        mask_blur=mask_blur,
        negative_prompt = negative_prompt,
        num_images_per_prompt=num_images_per_prompt
    )

    # Enhance results
    progress(0.9, desc="Enhancing images...")
    images = enhancer(
        images, upscale = upscale_value,
        background_enhance = enhance_background, face_upsample = face_upsample,
        codeformer_fidelity = fidelity,
        has_aligned = has_aligned, draw_box = draw_box
    )

    # show images
    return images

def rerun_enhancer(image_paths: list[str], enhance_background: bool, face_upsample: bool, draw_box: bool, has_aligned: bool, upscale_value: int, fidelity: float, progress=gr.Progress(track_tqdm=True)):
    enhancer = CodeFormer()

    images = [ Image.open(path).convert("RGB") for path, _ in image_paths ]

    # Enhance images
    images = enhancer(
        images, upscale = upscale_value,
        background_enhance = enhance_background, face_upsample = face_upsample,
        codeformer_fidelity = fidelity,
        has_aligned = has_aligned, draw_box = draw_box
    )

    # show images
    return images

def change_magic_prompt(magic_prompt: str):
    SETTINGS["magic_prompt"] = magic_prompt

def export_config(magic_prompt: str, masking_model: str, masks: list[str], mask_expansion: int, prompt: str, negative_prompt: str,
                  resolution: int, inpaint_model: str, guidance_scale: float, strength: float,
                  num_inference_steps: int, num_images_per_prompt: int, mask_blur: int, use_safety_checker: bool, load_in_4bit: bool,
                  enhance_background: bool, face_upsample: bool, draw_box: bool, has_aligned: bool, upscale_value: int, fidelity: float, lora_modules: str):
    config_data = json.dumps({
        "magic_prompt": magic_prompt,
        "masking_model": masking_model,
        "masks": masks,
        "mask_expansion": mask_expansion,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": resolution,
        "inpaint_model": inpaint_model,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_images_per_prompt,
        "mask_blur": mask_blur,
        "use_safety_checker": use_safety_checker,
        "load_in_4bit": load_in_4bit,
        "enhance_background": enhance_background,
        "face_upsample": face_upsample,
        "draw_box": draw_box,
        "has_aligned": has_aligned,
        "upscale_value": upscale_value,
        "fidelity": fidelity,
        "lora_modules": lora_modules
    }, indent=4)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.conf') as tmpfile:
        tmpfile.write(config_data.encode('utf-8'))
        tmpfile_path = tmpfile.name
    
    return tmpfile_path

def import_config(config_path: str):
    data: dict|None = None
    with open(config_path, "r") as f:
        data = dict(json.load(f))
    
    if data is None:
        raise FileNotFoundError("Failed to load config.")

    def get(key: str):
        return data.get(key, SETTINGS.get(key, ""))
    
    return get("magic_prompt"), get("masking_model"), get("masks"), get("mask_expansion"), get("prompt"), get("negative_prompt"), \
            get("resolution"), get("inpaint_model"), get("guidance_scale"), get("strength"), get("num_inference_steps"), get("num_images_per_prompt"), \
            get("mask_blur"), get("use_safety_checker"), get("load_in_4bit"), get("enhance_background"), get("face_upsample"), get("draw_box"), \
            get("has_aligned"), get("upscale_value"), get("fidelity"), get("lora_modules")
    

css = """
* { font-family: system-ui; }
tr { background: var(--checkbox-background-color); }
tr:nth-child(even) { background: var(--checkbox-border-color); }
tr:hover { background: var(--checkbox-border-color-hover); }
footer {height: 0px; visibility: hidden}
a { color: #7d7d80 }
.svelte-1ixn6qd { height: 100%; max-height: unset; flex-grow: 1 }
.svelte-1xp0cw7 { display: flex; justify-content: center; align-items: center }
"""

js = """
function refresh() {
    const url = new URL(window.location);

    if (!url.searchParams.has("__theme")) {
        url.searchParams.set('__theme', '""" + SETTINGS["theme"] + """');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(css=css, js=js) as demo:
    state = gr.State({})

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", type="pil")
            prompt = gr.TextArea(label="Prompt")
            negative_prompt = gr.TextArea(label="Negative Prompt", value=SETTINGS["negative_prompt"])

            with gr.Row():
                cancel_btn = gr.Button("Cancel", variant="stop")
                run_btn = gr.Button("Run")

        with gr.Column():
            mask_preview = gr.ImageEditor(
                label="Mask preview (editable)",
                type="pil",
                sources=[],
                interactive=True,
                brush=gr.Brush(colors=["#447BA0A0"], color_mode="fixed"),
                eraser=False,
                layers=False,
                format="png",
                image_mode="RGBA",
                show_download_button=False
            )

            gr.Markdown("## Masking Options")
            masks = gr.CheckboxGroup(choices=list(SETTINGS["masks_all"]), label="Choose preset(s)", value=SETTINGS["masks"])
            mask_expansion = gr.Slider(0, 300, value=SETTINGS["mask_expansion"], step=1, label='Expand Mask (px)')
            masking_model = gr.Textbox(value=SETTINGS["masking_model"], label="Masking model")

            with gr.Row():
                mask_clear_button = gr.Button("Recalculate mask")
                mask_draw_own = gr.Button("Draw my own mask")
        
        with gr.Column():
            output_preview = gr.Gallery(label="Final output", interactive=False)

            gr.Markdown("## Inpaint Options")

            with gr.Row():
                use_safety_checker = gr.Checkbox(value=SETTINGS["use_safety_checker"], label="Use safety checker")
                load_in_4bit = gr.Checkbox(value=SETTINGS["load_in_4bit"], label="Load in 4bit")
                
                resolution = gr.Number(value=SETTINGS["resolution"], label="Resolution")
                inpaint_model = gr.Textbox(value=SETTINGS["inpainting_model"], label="Inpaint model (DiffusionPipeline)")

            with gr.Row():
                guidance_scale = gr.Slider(0, 15, value=SETTINGS["guidance_scale"], step=0.1, label="Guidance Scale")
                strength = gr.Slider(0, 1, value=SETTINGS["strength"], step=0.1, label="Strength")
                num_inference_steps = gr.Slider(10, 150, value=SETTINGS["num_inference_steps"], step=5, label="Amount of inference steps")
                
                with gr.Row():
                    num_images_per_prompt = gr.Number(value=SETTINGS["num_images"], label="Number of images to generate")
                    mask_blur = gr.Number(value=SETTINGS["mask_blur"], label="Mask blur value")
            
            with gr.Accordion("LoRa Configuration", open=False):
                lora_modules = gr.Textbox(placeholder="Add LoRa modules (seperated by ',')", label="LoRa Weights", value=SETTINGS["lora_modules"])

            with gr.Accordion("Enhancement options", open=False):
                with gr.Row():
                    enhance_background = gr.Checkbox(value=SETTINGS["enhance_background"], label="Enhance background")
                    face_upsample = gr.Checkbox(value=SETTINGS["face_upsample"], label="Face upsample")
                    draw_box = gr.Checkbox(value=SETTINGS["draw_box"], label="Draw box")
                    has_aligned = gr.Checkbox(value=SETTINGS["has_aligned"], label="Has aligned")
                
                upscale_value = gr.Slider(1, 4, value=SETTINGS["upscale_value"], step=1, label='Upscale value')
                fidelity = gr.Slider(0, 1, value=SETTINGS["fidelity"], step=0.01, label='Fidelity (0 for better quality, 1 for better identity)')

                rerun_enhancer_btn = gr.Button("Rerun enhancer")

    gr.HTML("<br><br><br>")

    with gr.Accordion(label="Settings"):
        gr.Markdown("### Themes:")
        # Theme switcher button
        gr.HTML(
            """
            <button id="toggle_theme_button" onclick="
            // Apparently new URL doesn't work
            let current_theme = (window.location.search.replace('?', '').split('&')
                .map(x => x.split('='))
                .find(x => x[0] == '__theme') || ['__theme', 'light'])
                [1];

            let new_theme = current_theme == 'light' ? 'dark' : 'light';

            window.location.href = `?__theme=${new_theme}`"
            class="lg secondary svelte-1ixn6qd">Toggle dark / light theme</button>
            """
        )

        gr.Markdown("### Import and export configuration:")
        with gr.Row():
            import_config_btn = gr.File(file_count="single", type="filepath", label="Import settings", file_types=[".conf"])
            export_config_btn = gr.Button("Export current configuration")

        gr.Markdown("### Prompt suffix:")
        magic_prompt = gr.TextArea(SETTINGS["magic_prompt"], label="Enter prompt suffix:")

    all_inputs = [ magic_prompt, masking_model, masks, mask_expansion, prompt, negative_prompt, resolution, inpaint_model, guidance_scale, strength, num_inference_steps, num_images_per_prompt, mask_blur, use_safety_checker, load_in_4bit, enhance_background, face_upsample, draw_box, has_aligned, upscale_value, fidelity, lora_modules ]
    export_config_btn.click(export_config, inputs=all_inputs, outputs=[import_config_btn])
    import_config_btn.upload(import_config, inputs=[import_config_btn], outputs=all_inputs)

    magic_prompt.input(change_magic_prompt, inputs=[ magic_prompt ], outputs=[])

    mask_preview.change(handle_mask_draw, inputs=[mask_preview, state], outputs=[state])
    mask_clear_button.click(generate_mask, inputs=[input_image, masking_model, masks, mask_expansion, state], outputs=[mask_preview, state])
    mask_draw_own.click(clear_mask, inputs=[state], outputs=[mask_preview, state])
    input_image.upload(generate_mask, inputs=[input_image, masking_model, masks, mask_expansion, state], outputs=[mask_preview, state])

    run_event = run_btn.click(
        fn=run,
        inputs=[ input_image, prompt, negative_prompt, inpaint_model, use_safety_checker, load_in_4bit, enhance_background, face_upsample, draw_box, has_aligned, upscale_value, fidelity, resolution, guidance_scale, strength, num_inference_steps, num_images_per_prompt, mask_blur, lora_modules, state ],
        outputs=[ output_preview ]
    )

    rerun_event = rerun_enhancer_btn.click(
        fn=rerun_enhancer,
        inputs=[ output_preview, enhance_background, face_upsample, draw_box, has_aligned, upscale_value, fidelity ],
        outputs=[ output_preview ]
    )

    cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])


    # Additional information
    gr.HTML("""
        <hr><br>
        <h1 style="text-align: center">About this tool</h1>
        <p style="text-align: center; font-size: 1.1rem; max-width: 120ch; margin: auto; font-weight: 100; font: apple-system">This application is designed for <b>selective image inpainting and enhancement</b>. It allows you to modify specific parts of an image, such as clothing, background, or facial features, while preserving natural textures, proportions, and lighting.</p>
        <br>
        """)
    
    gr.HTML("""
        <h1 style="text-align: center">Usage</h1>
        <p style="font-size: 1.1rem; max-width: 120ch; width: max-content; margin: auto; font-weight: 100; font: apple-system">
            1. Upload an image you want to edit.<br>
            2. Use the <b>Masking Options</b> to select which parts of the image to modify.<br>
            3. Enter your <b>Prompt</b> describing the changes you want, and optionally a <b>Negative Prompt</b> to avoid unwanted artifacts.<br>
            4. Adjust <b>Inpaint Options</b> such as resolution, guidance scale, number of inference steps, and mask blur for more control.<br>
            5. Optionally, expand <b>Enhancement Options</b> to improve background, upscale faces, or enhance image quality.<br>
            6. Click <b>Run</b> to generate your edited images.<br>
            7. You can preview, adjust the mask, and re-run as needed to refine results.<br><br>
        </p>
    """)


    with gr.Accordion("Parameter overview", open=False):
        gr.Markdown("""
        | Setting | Description | Default value |
        |---------|-------------|---------------|
        | | <center>**Masking Options**</center>
        Preset Masks | Select parts of the image to automatically mask (e.g., Hair, Face, Upper-clothes). Multiple selections allowed. | <center>{masks}</center>
        Mask Expansion | Expand the mask by a specified number of pixels to cover more area around the selected region. | <center>{mask_expansion}px</center>
        Masking Model | The model used to generate masks. Default is a clothing segmentation model but can be replaced with another compatible model. | <center>{masking_model}</center>
        Draw My Own Mask | Manually paint the area you want to inpaint or modify. | <center>--</center>
        Recalculate Mask | Automatically regenerate the mask based on your selections and settings. | <center>--</center>
        | | <center>**Inpainting Options**</center>
        Prompt | Describe what you want the inpainted area to look like (e.g., "red shirt, realistic lighting"). | <center>--</center>
        Negative Prompt | Specify what you do **NOT** want in the output (e.g., "blurry, unnatural colors"). | <center>{negative_prompt}</center>
        Resolution | Output image resolution. Higher resolution = more detail but slower generation. | <center>{resolution}</center>
        Inpaint Model | The diffusion model used for inpainting. | <center>--</center>
        Guidance Scale | Controls how strongly the model follows your prompt. Higher = more faithful to prompt, lower = more creative. | <center>{guidance_scale}</center>
        Strength | How much the masked region is changed. 0 = no change, 1 = full modification. | <center>{strength}</center>
        Number of Inference Steps | More steps = higher quality, slower generation. | <center>{num_inference_steps}</center>
        Number of Images per Prompt | How many variations the model will generate for each prompt. | <center>{num_images}</center>
        Mask Blur | Softens mask edges for smoother transitions between edited and original areas. | <center>{mask_blur}</center>
        Use Safety Checker | Filters NSFW content. | <center>{use_safety_checker}</center>
        Load in 4-bit | Reduces GPU memory usage at a minor cost to model precision. | <center>{load_in_4bit}</center>
        LoRa Weights | Add Low rank adaptation modules to the base inpaint model. (Paths can be local paths or huggingface repos) | <center>{lora_modules}</center>
        | | <center>**Enhancement Options**</center> 
        Enhance Background | Improves quality and sharpness of non-masked areas. | <center>{enhance_background}</center>
        Face Upsample | Enhances facial detail and clarity. | <center>{face_upsample}</center>
        Draw Box | Adds a bounding box around faces or objects, useful for debugging or inspection. | <center>{draw_box}</center>
        Has Aligned | Check if the input image is already aligned for face/pose enhancement. | <center>{has_aligned}</center>
        Upscale Value | Multiplier for final image resolution to increase quality. | <center>{upscale_value}</center>
        Fidelity | Balances identity preservation vs. image quality in face enhancements (0 = more quality, 1 = more identity fidelity). | <center>{fidelity}</center>
        """.format(**SETTINGS))

    gr.HTML("""                       
    <hr>    
    <center><p style="font-size: 20px; display: flex; gap: 20px; justify-content: center">
            <span>A project by <a style="margin-left: 4px" href="https://c4vxl.de/">c4vxl</a></span>
            <span style="opacity: 0.5">|</span>
            <span><a href="https://github.com/c4vxl/Inpainter/">View on GitHub</a></span>
    </p></center>
    """)

demo.launch()