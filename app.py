import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image

print("Modeller yükleniyor...")
model_id = "runwayml/stable-diffusion-v1-5"

txt2img_pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)
txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(txt2img_pipe.scheduler.config)

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)
img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(img2img_pipe.scheduler.config)

if torch.cuda.is_available():
    txt2img_pipe = txt2img_pipe.to("cuda")
    txt2img_pipe.enable_attention_slicing()
    img2img_pipe = img2img_pipe.to("cuda")
    img2img_pipe.enable_attention_slicing()
    print("GPU kullanılıyor ✅")
else:
    print("CPU kullanılıyor")

def generate_image(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(int(seed))
    image = txt2img_pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        num_inference_steps=int(num_steps), guidance_scale=guidance_scale,
        width=int(width), height=int(height), generator=generator
    ).images[0]
    return image

def transform_image(init_image, prompt, negative_prompt, num_steps, guidance_scale, strength, seed):
    if init_image is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(int(seed))
    init_image = init_image.resize((512, 512))
    image = img2img_pipe(
        prompt=prompt, negative_prompt=negative_prompt, image=init_image,
        num_inference_steps=int(num_steps), guidance_scale=guidance_scale,
        strength=strength, generator=generator
    ).images[0]
    return image

with gr.Blocks(title="🎨 Stable Diffusion", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Stable Diffusion Görüntü Üretici
    **Model:** Stable Diffusion v1.5 | **Geliştirici:** Mehmet Tuğrul Kaya
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 Text-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(label="✍️ Prompt", lines=3)
                    t2i_negative = gr.Textbox(label="🚫 Negative Prompt", lines=2, value="blurry, low quality")
                    with gr.Accordion("⚙️ Ayarlar", open=False):
                        t2i_steps = gr.Slider(10, 50, 25, step=1, label="Steps")
                        t2i_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
                        with gr.Row():
                            t2i_width = gr.Slider(256, 768, 512, step=64, label="Width")
                            t2i_height = gr.Slider(256, 768, 512, step=64, label="Height")
                        t2i_seed = gr.Number(label="Seed", value=42, precision=0)
                    t2i_btn = gr.Button("🎨 Üret", variant="primary")
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Görüntü", type="pil")
            t2i_btn.click(generate_image, [t2i_prompt, t2i_negative, t2i_steps, t2i_guidance, t2i_width, t2i_height, t2i_seed], t2i_output)
        
        with gr.Tab("🖼️ Image-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(label="📤 Görüntü Yükle", type="pil")
                    i2i_prompt = gr.Textbox(label="✍️ Prompt", lines=3)
                    i2i_negative = gr.Textbox(label="🚫 Negative", lines=2, value="blurry")
                    with gr.Accordion("⚙️ Ayarlar", open=False):
                        i2i_steps = gr.Slider(10, 50, 30, step=1, label="Steps")
                        i2i_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
                        i2i_strength = gr.Slider(0, 1, 0.75, step=0.05, label="Strength")
                        i2i_seed = gr.Number(label="Seed", value=42, precision=0)
                    i2i_btn = gr.Button("🔄 Dönüştür", variant="primary")
                with gr.Column(scale=1):
                    i2i_output = gr.Image(label="Dönüştürülmüş", type="pil")
            i2i_btn.click(transform_image, [i2i_input, i2i_prompt, i2i_negative, i2i_steps, i2i_guidance, i2i_strength, i2i_seed], i2i_output)

if __name__ == "__main__":
    demo.launch()
