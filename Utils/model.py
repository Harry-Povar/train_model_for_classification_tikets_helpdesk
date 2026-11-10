import torch
import torch.nn as nn

from transformers import AutoModel


class XLMRobertaMultiTask(nn.Module):
    # Создаёт энкодер и классификационные головы
    def __init__(
            self,
            model_name,
            task_dims,
            tasks,
            smooth,
            pos_weights
    ):
        # Инициализирует базовый класс PyTorch-модели
        super().__init__()

        self.tasks = tasks
        # Сила сглаживания меток маршрутизации
        self.smooth = smooth
        # Веса положительных примеров для классов маршрутизации
        self.pos_weights = pos_weights

        self.encoder = AutoModel.from_pretrained(model_name)

        #Заморозка эмбеддинга
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Размер вектора, который энкодер формирует для каждого токена
        # Этот размер используется как вход для классификационных голов
        hidden_size = self.encoder.config.hidden_size

        #Отключение части значений при обучении
        self.dropout = nn.Dropout(0.4)

        self.heads = nn.ModuleDict()

        # Создаёт отдельную линейную голову для каждой задачи
        for task_name, task in tasks.items():

            self.heads[task["column"]] = nn.Linear(
                hidden_size,
                task_dims[task["column"]]
            )

    # Передаёт входные данные через модель и вычисляет предсказания
    def forward(
            self,
            input_ids,
            attention_mask,
            **labels
    ):

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        last_hidden = out.last_hidden_state


        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float()
        )


        pooled = (
            last_hidden * mask
        ).sum(1) / mask.sum(1)


        pooled = self.dropout(pooled)


        losses = []

        logits_dict = {}


        # Главная задача

        routing_col = self.tasks["routing"]["column"]
        routing_logits = self.heads[routing_col](pooled)
        logits_dict[routing_col] = routing_logits


        routing_labels = labels.get(routing_col)


        if routing_labels is not None:

            routing_labels = (
                routing_labels * (1 - self.smooth)
                + self.smooth * 0.5
            )


            routing_loss = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weights
            )(
                routing_logits,
                routing_labels
            )


            losses.append(
                routing_loss *
                self.tasks["routing"]["weight"]
            )


        # Второстепенные задачи
        for task_name, task in self.tasks.items():

            if task_name == "routing":
                continue


            col = task["column"]


            logits = self.heads[col](pooled)


            logits_dict[col] = logits


            y = labels.get(col)


            if y is None:
                continue


            loss = nn.CrossEntropyLoss()(
                logits,
                y
            )


            losses.append(
                loss * task["weight"]
            )


        total_loss = (
            torch.stack(losses).sum()
            if losses
            else None
        )


        return {
            "loss": total_loss,
            "routing_logits": routing_logits,
            "logits": logits_dict
        }

    # Проход модели для получения предсказаний без меток и расчёта loss. Используется при экспорте модели в ONNX
    def predict_forward(
            self,
            input_ids,
            attention_mask
    ):

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden = out.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(
            last_hidden.size()
        ).float()

        pooled = (
                         last_hidden * mask
                 ).sum(1) / mask.sum(1)

        pooled = self.dropout(pooled)

        outputs = []

        for name, head in self.heads.items():
            outputs.append(
                head(pooled)
            )

        return tuple(outputs)
