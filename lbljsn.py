import json
import pickle
import os

MODEL_PATH = "./saved_model_xlmroberta"

# Загружаем сохранённые энкодеры
with open(f"{MODEL_PATH}/artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)


routing_mlb = artifacts["routing_encoder"]
single_encoders = artifacts["single_encoders"]


labels = {
    "multi": {
        "Марштизация IT поддержка": routing_mlb.classes_.tolist()
    },
    "single": {
        col: encoder.classes_.tolist()
        for col, encoder in single_encoders.items()
    }
}


with open(f"{MODEL_PATH}/labels.json", "w", encoding="utf-8") as f:
    json.dump(
        labels,
        f,
        ensure_ascii=False,
        indent=2
    )

print("labels.json создан")