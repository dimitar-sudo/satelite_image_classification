"""Interactive web application for satellite terrain classification.

Upload a 64x64 RGB satellite tile and the app displays the image, the predicted
land-use class with its confidence, and a table of probabilities for all ten
classes. Built with Shiny for Python.

Run from the repository root::

    shiny run --reload app/app.py
"""
import os
import sys

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

from shiny.express import input, render, ui
from shiny import reactive

# Make the project's ``src`` package importable when launched from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import CLASS_NAMES, MEAN, STD, IMAGE_SIZE, NUM_CLASSES, WEIGHTS_PATH
from src.model import SatelliteCNN

# --------------------------------------------------------------------------- #
# Model loading (once at startup)
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

model = SatelliteCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()


def predict(image_path: str) -> pd.DataFrame:
    """Run the model on an image file and return a class/probability table,
    sorted by descending probability."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
    df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
    return df.sort_values("Probability", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
ui.page_opts(title="Satellite Terrain Classifier", fillable=True)

with ui.sidebar(title="Upload image"):
    ui.input_file("image", "Select satellite image",
                  accept=[".jpg", ".jpeg", ".png"], multiple=False)
    ui.input_action_button("predict_btn", "Predict class", class_="btn-primary")


@reactive.calc
def uploaded_path():
    """Path of the currently uploaded image, or None."""
    file_info = input.image()
    if not file_info:
        return None
    return file_info[0]["datapath"]


@reactive.calc
@reactive.event(input.predict_btn)
def prediction():
    """Probability DataFrame, recomputed when the Predict button is pressed."""
    path = uploaded_path()
    if path is None:
        return None
    return predict(path)


with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("Image")

        @render.image
        def show_image():
            path = uploaded_path()
            if path is None:
                return None
            return {"src": path, "width": "300px", "height": "300px"}

    with ui.card():
        ui.card_header("Class Probabilities")

        @render.table
        def prob_table():
            df = prediction()
            if df is None:
                return None
            out = df.copy()
            out["Probability"] = out["Probability"].map(lambda p: f"{p:.4f}")
            return out

with ui.card():
    ui.card_header("Prediction")

    @render.ui
    def prediction_text():
        df = prediction()
        if df is None:
            return ui.p("Upload an image and click 'Predict class'.")
        top = df.iloc[0]
        return ui.h4(
            f"Predicted Class: {top['Class']} "
            f"(Confidence: {top['Probability'] * 100:.2f}%)"
        )
