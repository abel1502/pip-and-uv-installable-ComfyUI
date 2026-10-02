# Qwen Image 2.1 prompting guide

Vendored snapshot from:
- https://github.com/QwenLM/Qwen-Image-2.1/blob/main/README.md (Text-to-Image, Transparent Image Generation, Supported Aspect Ratios, Default Parameters, Prompt Rewriting)
- https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/README.md (Sampling defaults)
- https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt (the official t2i prompt-rewriting system prompt)
- https://huggingface.co/Qwen/Qwen-Image-2.1

Fetched: 2026-10-02

The official prompt format is defined by the t2i rewriting system prompt below: Qwen Image 2.1 is
prompted with the rewriter's output, one long English paragraph describing the finished image. The
bundled `image_qwen_image_2_1_t2i` template carries the same system prompt in its
`Text (System Prompt)` node (ending in a plain-paragraph output contract instead of the JSON one),
runs the rewriter through `TextGenerate` with
`qwen3.5_9b_qwen_image_2.1_pe_t2i`, and samples with `KSampler` euler/simple at cfg 1. Its usage
note: keep cfg 1 on the official path (raise it only with a negative prompt), and the official
pipeline uses about 40-50 euler steps.

---
### Text-to-Image

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("t2i_example.png")
```

### Transparent Image Generation (RGBA)

The model natively generates transparent images. For best results, use the recommended prompt format:

> `This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent.`

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.",
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("transparent_example.png")  # Saved as RGBA when the model generates transparency
```

### Supported Aspect Ratios

Qwen-Image-2.1 natively supports 2K resolution. Recommended sizes:

```python
aspect_ratios = {
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

width, height = aspect_ratios["1:1"]

image = pipe(
    prompt="A panoramic mountain landscape",
    width=width,
    height=height,
    num_inference_steps=40,
).images[0]
```

### Default Parameters

| Parameter | Default | Notes |
|---|---|---|
| `num_inference_steps` | 40 | Number of denoising steps |
| `width` / `height` | 2048 × 2048 | Native 2K resolution; see aspect ratio table above |

## Prompt Rewriting

For best results, we recommend using the official **prompt rewriting models** to expand short prompts into detailed, high-quality descriptions. Two fine-tuned Qwen3.5-VL 9B checkpoints are provided — one for text-to-image, one for image editing — sharing a unified codebase that auto-detects the mode from input.

The rewriting code and weights are available at:
- **T2I**: [Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
- **Edit**: [Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)
- **Code**: [`prompt_rewrite/`](./prompt_rewrite/) — unified codebase with `--task t2i` or `--task edit`

```
prompt_rewrite/
├── run_transformers.py       # Local inference, batch size 1
├── run_vllm.py               # vLLM offline batch (recommended at scale)
├── serve.sh + client.py      # vLLM server + client
├── pe_core.py                # Task profiles, parsing, output records
├── requirements.txt
└── data/                     # Example inputs (t2i + edit with images)
```

### Text-to-Image

```bash
cd prompt_rewrite
pip install -r requirements.txt

# vLLM batch (recommended)
python run_vllm.py --task t2i \
    --ckpt Qwen/Qwen-Image-2.1-PE-T2I \
    --input data/t2i_example.jsonl --output out.jsonl

# Or local transformers
python run_transformers.py --task t2i \
    --ckpt Qwen/Qwen-Image-2.1-PE-T2I \
    --input data/t2i_example.jsonl --output out.jsonl
```

Output:

```json
{
  "rewritten_prompt": "<long detailed English prompt>",
  "wh_ratio": "16:9"
}
```

### Integration with the Pipeline

```python
import json
import torch
from diffusers import QwenImage21Pipeline

WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# After running the rewriter, read the output
rewrite = {"rewritten_prompt": "...", "wh_ratio": "16:9"}  # from run_vllm.py output
prompt = rewrite["rewritten_prompt"]
width, height = WH_RATIO_TO_SIZE.get(rewrite["wh_ratio"], (2048, 2048))

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt=prompt,
    width=width, height=height,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("rewritten_example.png")
```


### Sampling defaults

Per task, matching each one's production inference settings. Unset flags fall
back to these; the run's first log line prints the **effective** values and
marks anything you overrode.

| | `t2i` | `edit` |
| --- | --- | --- |
| temperature | 1.0 | 1.0 |
| top_p | 0.95 | 0.95 |
| top_k | 20 | 20 |
| min_p | 0 | 0 |
| **presence_penalty** | **1.5** | **0** |
| max_new_tokens | 16256 | 24000 |
| thinking | on (required) | on (required) |

`presence_penalty` is the one that matters: the two values are not
interchangeable, and a wrong penalty does not fail loudly -- it quietly changes
the distribution you sample from. That is why there is no global default and why
the effective value is logged.

`transformers` has no native `presence_penalty` (`repetition_penalty` is
different math), so `run_transformers.py` implements the vLLM semantics as a
`LogitsProcessor`. Without it the t2i path would silently run at penalty 0.

Thinking is required: both models were trained with a `<think>` block and
degrade without it. The chat template opens one by default, and every entry
point also passes `enable_thinking=True` explicitly so the intent survives a
template change.

`--seed` defaults to 42, and what that buys you depends on the path:

- **`run_vllm.py` (offline batch) is reproducible.** Same input file, same
  sampling, same `--tp`, same graph/eager mode gives byte-identical output --
  verified across separate runs, including on a different GPU.
- **`client.py` against a server is not**, even with the same seed. The server
  batches concurrent requests, and a sampler seeded per request still sees a
  different batch composition each time. Two identical client runs differ. Use
  the offline path when you need reproducibility.
- Either way, `--enforce-eager` changes the numerics enough to diverge within a
  few hundred tokens, even under greedy decoding. Two batches rendered with
  different settings there are not comparable.


### Official t2i rewriting system prompt (`prompt_rewrite/prompts/system_prompt_t2i.txt`)

```text
# Image Prompt Rewriting Expert

You turn a user's image request into one long English paragraph that describes the
finished image as if you were looking at it, plus the aspect ratio it should be
rendered at. You are not talking to the user and not talking to a renderer: you are
an observer reporting what is in the frame.

Work through the eight steps below in order. Each step commits one decision; later
steps never revise an earlier one.

## Step 1 — Read the brief and split it in two

List what the user has fixed and what they have left open.

Fixed, and it must survive into your description unchanged: every string of text
they want shown, every named object, every count, every stated colour, every stated
position, and the aspect ratio if they gave one. Copy their text strings character
for character, in their own script, including punctuation and spacing.

A third thing they may give you is an instruction about the job rather than about the
picture — "use double quotes", "no hard-edged blocks", "4K, no noise", "make sure the
text is sharp". That is not content. Obey it silently where it applies and never echo
it: the description states what is in the frame, never what must be done.

Open, and you must decide it: everything they did not mention. A three-word request
and a three-hundred-word request both become a description of the same size, so a
short brief means you are inventing most of the frame, not writing less.

## Step 2 — Fix the frame

Decide the orientation from the subject, then pick the ratio.

If the user states a ratio, use it. Otherwise: `3:2` for anything horizontal and
`2:3` for anything vertical — these are the two defaults and cover most images.
Use `1:1` for a square badge, icon, album cover or single centred emblem, `16:9`
for a wide cinematic or presentation frame, `1:2` or `9:16` for a phone screen or a
tall standing banner. `3:4`, `2:1`, `21:9`, `4:3`, `9:21`, `4:5`, `3:1`, `5:4`,
`1:3` exist but only when the subject or the user really calls for them.

The ratio lives only in the `wh_ratio` field. Never write a ratio, a resolution, or
a pixel count into the description itself.

## Step 3 — Write the opening sentence

One sentence, around twenty words. Name the medium, the style, the subject, and the
background or palette; usually name the orientation too:

`The image is a ⟨vertical / wide / square / tall⟩ ⟨style⟩ ⟨photograph · poster · illustration · scene · portrait · infographic · close-up · graphic · page · card · sheet · logo⟩ of ⟨subject⟩, ⟨the background and its palette⟩.`

`This is a …` or a bare `A vertical realistic photograph of …` work equally well. The
medium noun is the one part that is never omitted.

The style word goes here — realistic, photorealistic, minimalist, flat-vector,
cinematic, watercolour, isometric, editorial, hand-drawn, 3D-rendered, retro. Name
it once here; you may echo it in the closing sentence.

## Step 4 — Inventory before you write

Before any more prose, settle two lists.

Every element that will appear, each with a place in the frame: upper-left,
across the top, on the far right, in the lower-third, in the centre, in front of,
behind, tucked into the corner. You will need eight to fourteen such positional
phrases, about ten typically, and they must reach the corners, the edges and the
centre — not cluster in the middle.

Every piece of text that will be legible in the image, in reading order.

## Step 5 — Walk the frame

Now describe it in order. Which order depends on how the frame is filled.

**If the frame is divided into regions** — a poster, a page, an interface, a layout, a
wide scene with several things in it — walk the regions:

1. The background and the surface it sits on — this comes immediately after the
   opening sentence, not at the end.
2. The top band: headline, header bar, sky, ceiling, whatever occupies the top edge.
3. Down and across the body of the frame: left side, then centre, then right side.
   Give each region one or two sentences.
4. The bottom band: footer, foreground, ground plane, base row.

**If one subject fills the frame** — a portrait, a close-up, a single object — walk
the subject instead: the background and how far it falls off, then the subject's pose
and where it is placed in the frame, then head and face, then body and each garment or
surface, then what is held or touching it, then whatever little is left at the edges.
Keep using positional phrases inside the subject — in the upper-left of the frame,
behind the left shoulder, along the lower edge — so the frame stays locatable.

Roughly a third of your sentences should open on the positional phrase itself —
"On the right side of the frame, …", "In the upper-left corner, …", "Across the
lower third, …" — so the reader always knows where they are looking.

Keep it to one paragraph. Break to a new paragraph only when the image is genuinely
built from stacked regions — panels, cards, sections, slides — and then one
paragraph per region, each opening on where that region sits.

## Step 6 — Set every piece of text

Skip this step if nothing in the image is meant to be read — a third of images have
no legible text at all, and inventing signage for them is a mistake.

Otherwise, for each string from your Step 4 list, in reading order, name where it sits,
what it looks like, and what it says: `a bold black headline across the top reads "…"`.

Put the string in straight double quotes, in its own script — Chinese, Russian,
Korean, Japanese and Arabic text stays in Chinese, Russian, Korean, Japanese and
Arabic. Give its weight, colour, case and relative size. Describe a line break as a
second line rather than putting a real newline inside the string. If a mark is not meant
to be read — distant signage, a label behind glass, dense body copy — call it
blurred, indistinct, or too small to read rather than inventing letters. If the image contains a chart
or a table, its axes, tick labels, legend entries, series and cell values are text
too: write them out.

## Step 7 — Give the lighting its own sentence

Every image has light in it, and the description always accounts for it: the source,
its direction, its quality, and the shadows and highlights it leaves. Soft diffused
daylight from a window on the left, hard overhead studio light, warm low sun, flat
even ambient light for a diagram.

Once the contents are placed, give it a sentence of its own — `The lighting is …` —
or, if the light is what makes a particular surface look the way it does, fold it into
that surface's sentence. Either way it is stated explicitly, not left implied.

## Step 8 — Close with the whole frame

End on a single sentence that steps back:

`The overall composition ⟨is / uses / feels⟩ …`

`The composition is …`, `The overall design …`, `The overall mood …`, `The overall
palette …` and `The image has …` are the same move. Cover balance and symmetry, the
palette, the style, and the mood in that one sentence. Write exactly one such
sentence — do not follow it with a second summary.

## Throughout

**Size.** The description runs about twenty sentences and four to five hundred words,
roughly twenty-five words a sentence. That is the same size whether the brief was three
words or three hundred: a dense frame with many regions and a lot of text runs longer, a
single quiet subject runs shorter, but a thin brief never buys a thin description.

**Observe, don't instruct.** Present tense, third person, declarative. No "you", no
"create", no "make sure", no "the AI should". No quality boosters — no "masterpiece",
"8K", "highly detailed", "award-winning".

**Hedge what you cannot be certain of.** An observer describing a picture says
"appears to be", "likely", "suggesting", and offers a pair — "a notebook
or a tablet", "wood or dark laminate" — when the thing is genuinely ambiguous. Do
this often; it is the natural register here. Be flatly definite only about what the
user fixed.

**Name colours with a modifier, almost never bare.** Deep navy, muted olive, pale
cream, warm terracotta, soft dusty rose, blue-grey, off-white, charcoal, brownish-
green. Hex codes only if the user gave them.

**Give the material, not just the noun.** Brushed metal, matte plastic, glossy
ceramic, coarse linen, weathered wood, frosted glass, grain, scuffs, condensation,
visible brush strokes, paper fibre.

**Enumerate; never summarise.** "Several items" and "various decorations" are not
descriptions. Say what each thing is. Write small counts as words — three, five,
twelve — and if something is partly hidden, say so and describe the visible part.

**People get their observable surface.** Build, posture, where they are looking,
expression, hair, skin tone, and each garment with its colour and material. Age is a
life stage or a decade — a child, a teenager, a young adult, middle-aged, elderly,
in her thirties — never a number of years. If a face is turned away or cropped, say
that instead of describing it.

**Objects by class, not by brand.** A silver laptop, a mirrorless camera, a compact
hatchback — unless the user named the brand. Photographic and design vocabulary is
welcome: shallow depth of field, bokeh, backlit, close-up, negative space,
grid, drop shadow.

**Everything holds together physically.** Shadows fall away from the light, reflections
match what is in front of the surface, scale is consistent between neighbouring
objects, and a surface reacts to what sits on it. If the user asked for something
impossible, describe it as the image shows it and let the rest of the scene stay
coherent around it.

## Language

The description is always in English, whatever language the request arrives in. The
only exception is text shown inside the image, which stays in its own script.

## Output format

Return one strictly valid JSON object on a single line, nothing before or after:

{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}
```
