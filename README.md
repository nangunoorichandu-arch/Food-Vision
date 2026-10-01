# Food Vision - Pizza, Steak, or Sushi?

A Gradio web app that classifies a food photo as **pizza**, **steak**, or **sushi** using a fine-tuned EfficientNet-B2 model. Deployable to [Render](https://render.com) out of the box.

## How it works

- `model.py` builds an EfficientNet-B2 with a 3-class classifier head (torchvision's ImageNet backbone, no pretrained download needed).
- `app.py` loads the trained weights from `effnetb2.pth`, wraps the model in a Gradio `Interface`, and serves predictions with their probabilities and inference time.
- `examples/` holds sample images (pizza, steak, sushi) shown in the UI for quick testing.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

The app starts on `http://localhost:7860` (or the port set in the `PORT` environment variable).

## Deploying to Render

This repo includes a `render.yaml` blueprint — connect the repo in the Render dashboard and it will:

1. Install dependencies with `pip install -r requirements.txt`
2. Start the app with `python app.py`

Render injects the `PORT` environment variable automatically; `app.py` binds to it.

## Project structure

```
app.py             Gradio app entry point
model.py           EfficientNet-B2 model builder
effnetb2.pth        Trained model weights
examples/           Sample images for the UI
requirements.txt    Python dependencies
render.yaml         Render deployment config
```
