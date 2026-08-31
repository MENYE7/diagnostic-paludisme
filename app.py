import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("palu_best.keras")

def predict(img):
    if img is None:
        return "⚠️ Veuillez charger une image"
    img_resized = img.resize((128, 128))
    arr = np.array(img_resized).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    score = float(model.predict(arr, verbose=0)[0][0])
    if score > 0.5:
        return f"🔴 CELLULE PARASITÉE\nConfiance : {score*100:.1f}%\n\n➡️ Faire confirmer par goutte épaisse ou TDR."
    else:
        return f"🟢 CELLULE SAINE\nConfiance : {(1-score)*100:.1f}%\n\n➡️ Résultat négatif. Consulter un professionnel si doute."

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Image de cellule sanguine"),
    outputs=gr.Textbox(label="Résultat", lines=4),
    title="🩺 Dépistage du Paludisme par IA",
    description="Chargez une image de cellule sanguine.\n⚕️ Outil expérimental.",
    theme=gr.themes.Soft(),
    flagging_mode="never"
)

app.launch(server_name="0.0.0.0", server_port=7860)
