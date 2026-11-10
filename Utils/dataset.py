import torch
from torch.utils.data import Dataset

class HelpdeskDataset(Dataset):
    def __init__(
            self,
            texts,
            labels,
            tokenizer,
            lengths,
            max_length,
            routing_column
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.lengths = lengths
        self.max_length = max_length
        self.routing_column = routing_column


    def __len__(self):
        return len(self.texts)

    #Возвращает один пример по его индексу
    def __getitem__(self, idx):
        #Токенизирует текст заявки, если он длинее max_length, то он усекается
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length
        )


        item = {
            "input_ids": torch.tensor(
                enc["input_ids"],
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                enc["attention_mask"],
                dtype=torch.long
            ),

            "length": self.lengths[idx],

            "idx": idx,
            #Метка маршрутизации преобразуется в тензор типа float
            self.routing_column:
                torch.tensor(
                    self.labels[self.routing_column][idx],
                    dtype=torch.float
                )
        }

        # Добавляет в пример метки остальных задач
        for col, values in self.labels.items():

            if col == self.routing_column:
                continue

            item[col] = torch.tensor(
                values[idx],
                dtype=torch.long
            )

        return item