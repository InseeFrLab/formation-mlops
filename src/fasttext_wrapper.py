"""
FastText wrapper for MLflow.
"""
from typing import Tuple, Dict
import fasttext
import mlflow
import pandas as pd

from preprocessor import Preprocessor
from constants import TEXT_FEATURE


class FastTextWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to wrap and use FastText models with MLflow.
    """

    def __init__(self):
        """
        Construct a FastTextWrapper object.
        """
        self.model = None
        self.preprocessor = Preprocessor()

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the FastText model and its configuration file from an MLflow model
        artifact. This method is called when loading an MLflow model with
        pyfunc.load_model(), as soon as the PythonModel is constructed.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context where
                the model artifact is stored. It should contain the following
                artifacts:
                    - "model_path": path to the FastText model file.
                    - "config_path": path to the configuration file.
        """

        self.model = fasttext.load_model(context.artifacts["model_path"])

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: Dict
    ) -> Tuple:
        """
        Predicts the k most likely codes to a query.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model
                context.
            model_input (dict): A dictionary containing the input data for the
                model. It should have the following keys:
                - 'query': A dictionary containing the query features.
                - 'k': An integer representing the number of predicted codes to
                return.

        Returns:
            A tuple containing the k most likely codes to the query.
        """
        df = self.preprocessor.clean_text(
            pd.DataFrame(model_input["query"],
                         columns=[TEXT_FEATURE]),
            text_feature=TEXT_FEATURE
        )

        texts = df.apply(self._format_item, axis=1).to_list()

        return self.model.predict(texts, k=model_input["k"])

    def _format_item(self, row: pd.Series) -> str:
        """
        Formats a row of data into a string.

        Args:
            row (pandas.Series): A pandas series containing the row data.

        Returns:
            A formatted item string.
        """
        formatted_item = row[TEXT_FEATURE]
        return formatted_item
