import os
import pickle
import torch
import torch.nn as nn

from Utils.config import Config
from Utils.model import XLMRobertaMultiTask


config = Config("config.yaml")

MODEL_NAME = config.model["name"]
MODEL_PATH = config.model["path"]

ONNX_PATH = os.path.join(
    MODEL_PATH,
    "model.onnx"
)

MAX_LENGTH = config.training["max_length"]

DEVICE = torch.device("cpu")
OPSET = 18


with open(
    os.path.join(MODEL_PATH, "artifacts.pkl"),
    "rb"
) as f:
    artifacts = pickle.load(f)

routing_encoder = artifacts["routing_encoder"]
single_encoders = artifacts["single_encoders"]


task_dims = {}

routing_column = config.tasks["routing"]["column"]

task_dims[routing_column] = len(
    routing_encoder.classes_
)

for column, encoder in single_encoders.items():

    task_dims[column] = len(
        encoder.classes_
    )

model = XLMRobertaMultiTask(
    model_name=MODEL_NAME,
    task_dims=task_dims,
    tasks=config.tasks,
    smooth=config.training["smooth"],
    pos_weights=None
)

state_dict = torch.load(
    os.path.join(MODEL_PATH, "weight.bin"),
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()
model.to(DEVICE)

# Обёртка задаёт формат входов и выходов, который будет использоваться при ONNX-экспорте
class ONNXWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        # Сохраняет исходную модель внутри обёртки
        self.model = model

    # Метод forward определяет вычисления,которые будут записаны в ONNX-граф
    def forward(
        self,
        input_ids,
        attention_mask
    ):
        return self.model.predict_forward(
            input_ids,
            attention_mask
        )

# Создаём обёртку и переводит её в режим оценки
wrapper = ONNXWrapper(model)
wrapper.eval()

# Создаём тестовые входные тензоры для трассировки модели во время экспорта. Размер батча — 1, длина последовательности — MAX_LENGTH.

dummy_input_ids = torch.ones(
    (1, MAX_LENGTH),
    dtype=torch.long
)

dummy_attention_mask = torch.ones(
    (1, MAX_LENGTH),
    dtype=torch.long
)


# Формирует имена выходов ONNX
output_names = []

for task_name, task in config.tasks.items():

    output_names.append(
        task["column"] + "_logits"
    )

# Экспорт в ONNX
torch.onnx.export(
    wrapper,
    (
        dummy_input_ids,
        dummy_attention_mask
    ),
    ONNX_PATH,
    input_names=[
        "input_ids",
        "attention_mask"
    ],
    output_names=output_names,
    dynamic_axes={
        "input_ids": {
            0: "batch_size",
            1: "sequence_length"
        },
        "attention_mask": {
            0: "batch_size",
            1: "sequence_length"
        },
        **{
            name: {
                0: "batch_size"
            }
            for name in output_names
        }
    },
    opset_version=OPSET,
    do_constant_folding=False
)

print("ONNX экспорт завершён")
print("Файл:", ONNX_PATH)
print()

print("Входы:")
print(" - input_ids")
print(" - attention_mask")

print()

print("Выходы:")
for name in output_names:
    print(" -", name)
