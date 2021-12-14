from typing import List
from Image import Image

class Client:
    id_client: str
    models: List[int]
    images: List[Image]

    def __init__(self, id_client: str, models: List[int], images: List[Image]) -> None:
        self.id_client = id_client
        self.models = models
        self.images = images
