import yaml


class Config:

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)


    @property
    def dataset(self):
        return self.data["dataset"]


    @property
    def tasks(self):
        return self.data["tasks"]


    @property
    def model(self):
        return self.data["model"]


    @property
    def training(self):
        return self.data["training"]