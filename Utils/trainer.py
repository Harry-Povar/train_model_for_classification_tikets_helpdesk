import torch

from transformers import Trainer

# Кастомный Trainer
class RoutingTrainer(Trainer):
    # Принимает название колонки с метками маршрутизации, а остальные аргументы передаёт базовому Trainer
    def __init__(
        self,
        routing_column,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.routing_column = routing_column

    # Выполняет один шаг получения результатов на этапе оценки. Этот метод вызывается Trainer при проверке модели на validation-наборе.
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):

        with torch.no_grad():

            outputs = model(**inputs)

        # Получает общую потерю модели и отсоединяет её от графа вычислений
        loss = outputs["loss"].detach()

        # Получает logits только для задачи маршрутизации, они будут использоваться для вычисления routing_macro_f1.
        logits = outputs["routing_logits"].detach()

        # Получает правильные метки маршрутизации из входного батча
        labels = inputs[self.routing_column].detach()

        return (
            loss,
            logits,
            labels
        )