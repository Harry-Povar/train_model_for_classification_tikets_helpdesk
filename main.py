import os
import pickle
import pandas as pd
import numpy as np
import torch

from Utils.model import XLMRobertaMultiTask
from Utils.config import Config
from Utils.dataset import HelpdeskDataset
from Utils.trainer import RoutingTrainer

from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



# Загрузка конфигурации из файла config.yaml
config = Config("config.yaml")

# Столбцы, формирующие текст заявки
TEXT_COLS = config.dataset["text_columns"]

# Словарь задач классификации
TASKS = config.tasks

MODEL_NAME = config.model["name"]
MODEL_PATH = config.model["path"]

# Параметры обучения
SMOOTH = config.training["smooth"]
EPOCHS = config.training["epochs"]
LR = config.training["learning_rate"]
BATCH_TRAIN = config.training["batch_train"]
BATCH_EVAL = config.training["batch_eval"]
MAX_LENGTH = config.training["max_length"]
MAX_ROWS = config.dataset["max_rows"]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Читаем таблицу с заявками
df = pd.read_csv(
    config.dataset["csv_path"],
    sep=config.dataset["separator"]
)
if MAX_ROWS:
    df = df.sample(n=min(MAX_ROWS, len(df)), random_state=42)

# Объединяем содержимое ячеек, формирующих текст заявки, в одну общую ячейку, для всех тикетов
df["input_text"] = (
    df[TEXT_COLS].fillna("").astype(str).agg(". ".join, axis=1)
)

# Кодирование меток маршрутизации
# Маршрутизация — многометочная задача: заявка может относиться сразу к нескольким отделам.
# В исходной ячейке ожидается строка с названиями отделов, разделёнными запятыми, например: "Отдел А, Отдел Б"
routing_values = df[TASKS["routing"]["column"]].fillna("").astype(str).apply(
    lambda x: [s.strip() for s in x.split(",") if s.strip()]
)

# Собираем и сортируем все уникальные классы маршрутизации.
routing_classes = sorted({cls for v in routing_values for cls in v})
routing_mlb = MultiLabelBinarizer(classes=routing_classes)
y_routing = routing_mlb.fit_transform(routing_values)





# Вывод о частоте отделов
print(
    df[TASKS["routing"]["column"]]
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)

# Считаем количество положительных примеров по каждому классу
routing_counts = y_routing.sum(axis=0)
total = len(y_routing)

# Вычисление весов для каждого класса
pos_weights = np.sqrt((total - routing_counts) / (routing_counts + 1e-4))
pos_weights = torch.tensor(pos_weights, dtype=torch.float).to(DEVICE)

# Кодирование остальных задач с одиночными метками
y_single = {}
single_encoders = {}

for task in TASKS.values():
    if task["type"] == "single":
        col = task["column"]
        # если значение пропущено, инициализируем его как "неизвестно"
        values = df[col].fillna("неизвестно").astype(str).str.strip()
        le = LabelEncoder()
        y_single[col] = le.fit_transform(values)
        single_encoders[col] = le

# Вычисление метрик
# RoutingTrainer передаёт сюда logits и правильные метки маршрутизации.
# sigmoid преобразует logits в вероятности для каждого отдела.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"routing_macro_f1": f1}

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Считаем длину каждого текста после токенизации, если токенов больше чем MAX_LENGTH, текст усекается
# Необходимо для группировки текстов с похожей длинной
df["token_len"] = [
    len(tokenizer.encode(text, truncation=True, max_length=MAX_LENGTH))
    for text in df["input_text"]
]

# Собираем заявки в один батч и приводим их к длине самого длинного примера
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt"
)

# HelpdeskDataset возвращает текст в виде input_ids и attention_mask, а также метки задач
dataset = HelpdeskDataset(
    texts=df["input_text"].tolist(),
    labels={
        TASKS["routing"]["column"]: y_routing,
        **y_single
    },
    tokenizer=tokenizer,
    lengths=df["token_len"].tolist(),
    max_length=MAX_LENGTH,
    routing_column=TASKS["routing"]["column"]
)

# Разделение датасета на обучающую и валидационную выборки
idx = np.arange(len(dataset))
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds = torch.utils.data.Subset(dataset, val_idx)

# Рассчет размеров выходов модели для каждой головы(задачи)
# Для маршрутизации это количество отделов, для остальных задач - количество классов в конкретной задаче
task_dims = {}
task_dims[TASKS["routing"]["column"]] = y_routing.shape[1]

for col, enc in single_encoders.items():
    task_dims[col] = len(enc.classes_)


# Создание модели
# XLMRobertaMultiTask загружает базовый XLM-RoBERTa, создаёт общий энкодер и отдельную классификационную голову для каждой задачи.
model = XLMRobertaMultiTask(
    model_name=MODEL_NAME,
    task_dims=task_dims,
    tasks=TASKS,
    smooth=SMOOTH,
    pos_weights=pos_weights
)


trainable = 0
total = 0

for name, param in model.named_parameters():
    total += param.numel()
    if param.requires_grad:
        trainable += param.numel()

print(f"Всего параметров: {total:,}")
print(f"Обучаемых параметров: {trainable:,}")
print(f"Процент обучения: {trainable / total * 100:.2f}%")


#for name, param in model.named_parameters():
#    if not param.requires_grad:
#        print(name)

# Настройки обучения
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    lr_scheduler_type="linear",
    weight_decay=0.01,
    length_column_name="length",
    group_by_length=True,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="routing_macro_f1",
    greater_is_better=True,
    save_total_limit=1,
    remove_unused_columns=False
)

trainer = RoutingTrainer(
    routing_column=TASKS["routing"]["column"],
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

# Запускаем обучение
trainer.train()

# Сохраняем веса
os.makedirs(MODEL_PATH, exist_ok=True)
torch.save(model.state_dict(), f"{MODEL_PATH}/weight.bin")

# Сохраняем токенизатор
tokenizer.save_pretrained(MODEL_PATH)

# Сохраняем информацию, нужную для интерпретации выходов модели:
artifacts = {
    "routing_encoder": routing_mlb,
    "single_encoders": single_encoders,
    "text_columns": TEXT_COLS
}

with open(f"{MODEL_PATH}/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Обучение завершено")
