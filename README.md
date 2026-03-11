# Stable Diffusion Text2Image & Image2Image

[![Hugging Face](https://img.shields.io/badge/🤗-Spaces-blue)](https://huggingface.co/spaces/tugrulkaya/stable-diffusion-text2img)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

AI destekli görüntü üretme ve dönüştürme uygulaması.

## Özellikler

- **Text-to-Image**: Metin açıklamalarından görüntü üret
- **Image-to-Image**: Mevcut görüntüleri dönüştür
- **Özelleştirilebilir**: Steps, guidance, boyut kontrolü
- **Türkçe UI**: Kullanıcı dostu arayüz

## Kurulum
```bash
pip install -r requirements.txt
python app.py
```

## Demo

👉 [Hugging Face Spaces](https://huggingface.co/spaces/tugrulkaya/stable-diffusion-text2img)

## Kullanım

### Text-to-Image
```python
prompt = "A beautiful sunset, 8k, detailed"
# Görüntü üretilir
```

### Image-to-Image
```python
prompt = "watercolor painting style"
strength = 0.75
# Görüntü dönüştürülür
```

## ️ Teknoloji

- Stable Diffusion v1.5
- Hugging Face Diffusers
- Gradio
- PyTorch

## ‍ Geliştirici

**Mehmet Tuğrul Kaya**
- GitHub: [@mtkaya](https://github.com/mtkaya)
- Hugging Face: [@tugrulkaya](https://huggingface.co/tugrulkaya)

## Lisans

Apache 2.0
