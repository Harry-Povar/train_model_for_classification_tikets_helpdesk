import torch
import pickle

from transformers import AutoTokenizer

from Utils.config import Config
from Utils.model import XLMRobertaMultiTask


config = Config("config.yaml")

TASKS = config.tasks

MODEL_NAME = config.model["name"]
MODEL_PATH = config.model["path"]

MAX_LENGTH = config.training["max_length"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


with open(
        f"{MODEL_PATH}/artifacts.pkl",
        "rb"
) as f:
    artifacts = pickle.load(f)

# Кодировщик многометочных классов маршрутизации
routing_encoder = artifacts["routing_encoder"]

# Словарь кодировщиков single-label задач:
single_encoders = artifacts["single_encoders"]

task_dims = {}

routing_col = TASKS["routing"]["column"]



task_dims[routing_col] = len(
    routing_encoder.classes_
)


for col, encoder in single_encoders.items():
    task_dims[col] = len(
        encoder.classes_
    )

# Создаёт модель той же архитектуры, которая использовалась при обучении
model = XLMRobertaMultiTask(
    model_name=MODEL_NAME,
    task_dims=task_dims,
    tasks=TASKS,
    smooth=config.training["smooth"],
    pos_weights=None
)

# Загружает сохранённые параметры модели
state_dict = torch.load(
    f"{MODEL_PATH}/weight.bin",
    map_location=DEVICE
)

# Применяет загруженные параметры к созданной модели
model.load_state_dict(state_dict)

model.to(DEVICE)

# Режим предсказания
model.eval()


# Загружает токенизатор из папки сохранённой модели
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


def predict(text):

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


    input_ids = encoded["input_ids"].to(DEVICE)

    attention_mask = encoded["attention_mask"].to(DEVICE)


    with torch.no_grad():
        # Запускает модель только по входным токенам
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Здесь будут собраны результаты по всем задачам
    result = {}


    # Получает logits маршрутизации
    routing_logits = outputs["routing_logits"]

    routing_probs = torch.sigmoid(routing_logits)[0]

    # Названия классов идут в том же порядке, что и выходы маршрутизационной головы
    routing_classes = routing_encoder.classes_

    routing_result = []

    for idx, prob in enumerate(routing_probs):

        if prob >= 0.5:
            routing_result.append(
                routing_classes[idx]
            )
    result[routing_col] = routing_result

    #  Получает logits всех задач из результата модели
    logits_dict = outputs["logits"]

    # Обрабатывает каждую задачу, кроме уже разобранной маршрутизации
    for col, logits in logits_dict.items():

        if col == routing_col:
            continue
        probabilities = torch.softmax(
            logits,
            dim=1
        )
        class_id = torch.argmax(
            probabilities,
            dim=1
        ).item()
        encoder = single_encoders[col]
        value = encoder.inverse_transform([class_id])[0]
        result[col] = value
    return result
# Тест
# Этот блок выполняется только при прямом запуске predict.py
# При импорте функции predict из другого файла этот цикл не запускается

if __name__ == "__main__":
    print("Для выхода напишите: exit\n")

    while True:

        ticket = input("Текст заявки: ")

        if ticket.lower() == "exit":
            break

        if not ticket.strip():
            print("Заявка не может быть пустой\n")
            continue

        prediction = predict(ticket)

        print("\nРезультат классификации:")

        for key, value in prediction.items():
            print(
                f"{key}: {value}"
            )

        print("\n" + "=" * 50 + "\n")