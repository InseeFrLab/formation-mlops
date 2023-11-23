"""
FastText wrapper for MLflow.
"""
from typing import Tuple, Optional, Dict, Any, List
import fasttext
import mlflow
import pandas as pd

from preprocessor import Preprocessor
from constants import TEXT_FEATURE, LABEL_PREFIX


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
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: List,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple:
        """
        Predicts the most likely codes for a list of texts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model
                context.
            model_input (List): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.

        Returns:
            A tuple containing the k most likely codes to the query.
        """
        df = self.preprocessor.clean_text(
            pd.DataFrame(model_input, columns=[TEXT_FEATURE]),
            text_feature=TEXT_FEATURE,
        )

        texts = df.apply(self._format_item, axis=1).to_list()

        predictions = self.model.predict(
            texts,
            **params
        )

        predictions_formatted = {
            i: {
                rank_pred
                + 1: {
                    "nace": predictions[0][i][rank_pred].replace(LABEL_PREFIX, ""),
                    "probability": float(predictions[1][i][rank_pred]),
                }
                for rank_pred in range(params["k"])
            }
            for i in range(len(predictions[0]))
        }

        return predictions_formatted

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
