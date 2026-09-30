import os
from timeit import default_timer as timer

import gradio as gr
import torch

from model import create_effnetb2_model

torch.set_num_threads(1)  # free tier has a fraction of a CPU
class_names = ["pizza", "steak", "sushi"]

model, transforms = create_effnetb2_model(num_classes=len(class_names))
model.load_state_dict(torch.load("effnetb2.pth", map_location="cpu"))
model.eval()


def predict(img):
    start = timer()
    x = transforms(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return pred, round(timer() - start, 4)


example_list = [[os.path.join("examples", f)] for f in sorted(os.listdir("examples"))]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediction"),
        gr.Number(label="Prediction Time (s)"),
    ],
    examples=example_list,
    title="Pizza, Steak, or Sushi?",
    description="Upload a photo and the model will classify it as pizza, steak, or sushi.",
)

# Render provides the port in the PORT variable
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
