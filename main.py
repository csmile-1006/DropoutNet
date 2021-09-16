import fire
from src.model.trainer import Trainer


class Main(object):
    def __init__(self, conf_fname="./src/conf/config.json"):
        self.tr = Trainer(conf_fname)

    def run(self):
        self.tr.load()
        self.tr.train()


if __name__ == "__main__":
    fire.Fire(Main)
