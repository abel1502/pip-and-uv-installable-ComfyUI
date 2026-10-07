# Ideogram 4 model card

Vendored snapshot from: https://huggingface.co/Comfy-Org/Ideogram-4

Fetched: 2026-10-02

---

---
license: other
license_name: ideogram-non-commercial-model-agreement
license_link: https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md
tags:
- comfyui
- diffusion-single-file
base_model:
- ideogram-ai/ideogram-4-fp8
---

# Ideogram 4

Repackaged model files for ComfyUI.

Original model repository: https://huggingface.co/ideogram-ai/ideogram-4-fp8

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── ideogram4_fp8_scaled.safetensors
│   │   ├── ideogram4_int8_convrot.safetensors
│   │   ├── ideogram4_nvfp4_mixed.safetensors
│   │   ├── ideogram4_unconditional_fp8_scaled.safetensors
│   │   ├── ideogram4_unconditional_int8_convrot.safetensors
│   │   └── ideogram4_unconditional_nvfp4_mixed.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_8b_fp8_scaled.safetensors
│   │   └── qwen3vl_8b_nvfp4.safetensors
│   ├── 📂 vae/
│   │   └── flux2-vae.safetensors
```

## Workflows

<table>
<thead>
<tr><th align="center" valign="middle" style="text-align:center;vertical-align:middle">Workflow</th><th colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle">Thumb</th></tr>
</thead>
<tbody>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ideogram4_t2i_int8.json">Ideogram v4 Int8: Text to Image</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ideogram4_t2i_int8.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_ideogram4_t2i_int8-1.webp" width="200" height="200" alt="Ideogram v4 Int8: Text to Image"></a></td></tr>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ideogram4_t2i.json">Ideogram v4: Text to Image</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ideogram4_t2i.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_ideogram4_t2i-1.webp" width="200" height="200" alt="Ideogram v4: Text to Image"></a></td></tr>
</tbody>
</table>
