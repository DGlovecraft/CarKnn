from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "data" / "car_evaluation.csv"

COLUMN_NAMES = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class",
]

RESULT_INFO = {
    "unacc": {
        "title": "UNACCEPTABLE",
        "detail": "ไม่เหมาะสม / ไม่เป็นที่ยอมรับ",
    },
    "acc": {
        "title": "ACCEPTABLE",
        "detail": "อยู่ในระดับที่ยอมรับได้",
    },
    "good": {
        "title": "GOOD",
        "detail": "อยู่ในระดับดี",
    },
    "vgood": {
        "title": "VERY GOOD",
        "detail": "อยู่ในระดับดีมาก",
    },
}


class CarKnnModel:
    def __init__(self):
        self.df = self._load_dataset()
        self.X = self.df.drop("class", axis=1)
        self.y = self.df["class"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y,
        )

        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.X_train_encoded = self.encoder.fit_transform(self.X_train)
        self.X_test_encoded = self.encoder.transform(self.X_test)

        self.models = {}
        self.accuracies = {}

        for k in [1, 3, 5]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.X_train_encoded, self.y_train)

            predictions = model.predict(self.X_test_encoded)
            accuracy = accuracy_score(self.y_test, predictions)

            self.models[k] = model
            self.accuracies[k] = accuracy

    def _load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(
            DATA_PATH,
            header=None,
            names=COLUMN_NAMES,
        )

    def predict(self, selected_k: int, new_car: pd.DataFrame):
        model = self.models[selected_k]
        new_car_encoded = self.encoder.transform(new_car)

        prediction = model.predict(new_car_encoded)[0]
        distances, indices = model.kneighbors(new_car_encoded)

        vote_count = {}

        for number, index in enumerate(indices[0], start=1):
            neighbor_data = self.X_train.iloc[index]
            neighbor_class = self.y_train.iloc[index]
            distance = distances[0][number - 1]

            vote_count[neighbor_class] = vote_count.get(neighbor_class, 0) + 1

        vote_text = "Voting: "
        votes = []

        for car_class in ["unacc", "acc", "good", "vgood"]:
            count = vote_count.get(car_class, 0)
            if count > 0:
                votes.append(f"{car_class.upper()} = {count}")

        vote_text += "   |   ".join(votes) if votes else "No votes"

        return prediction, vote_text, distance

    @property
    def dataset_summary(self):
        return {
            "total": len(self.df),
            "training": len(self.X_train),
            "testing": len(self.X_test),
        }
