import os


class Settings:
    MODEL_NAME: str = "Bert-cased"
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", "weights/model_epoch1.pth"
    )  # if env is not set, it defaults to the best weights
    NUM_LABELS: int = 15
    MAX_LENGTH: int = 100
    TRUNCATION: bool = True
    PADDING: bool = True
    IDX2CLASS: list = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "Z",
    ]
    THRESHOLD: float = 0.5


settings = Settings()
