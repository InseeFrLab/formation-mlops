"""
Utils.
"""
from typing import Dict, Tuple, Union
import mlflow
import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import JSONResponse


def get_model(
    model_name: str, model_version: str
) -> mlflow.pyfunc.PyFuncModel:
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.
    Returns:
        model (mlflow.pyfunc.PyFuncModel): The loaded machine learning model.
    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error


def preprocess_query(
    description: str,
    nb_echoes_max: int = 5,
) -> Dict:
    """
    This function preprocesses the input query parameters for making
    predictions using the fetched machine learning model.

    Args:
        description (str): The activity description to be used for prediction.
        nb_echoes_max (int, optional): The maximum number of echoed
            predictions. Default is 5.
    Returns:
        query (Dict): The preprocessed query in the required format for
        making predictions.
    """
    query = {
        "query": {"TEXT_FEATURE": [description]},
        "k": nb_echoes_max,
    }
    return query


def process_response(
    predictions: Tuple,
    prediction_index: int,
    nb_echoes_max: int
) -> Union[Dict, JSONResponse]:
    """
    Process model predictions and generates response.

    Args:
        predictions (Tuple): The model predictions as a tuple of two numpy
        arrays.
        prediction_index (int): Index of prediction.
        nb_echoes_max (int): The maximum number of echo predictions.
        prob_min (float): The minimum probability threshold for predictions.
    Returns:
        response (Dict or JSONResponse): The processed response as a
        dictionary or a JSONResponse object containing the
        predicted results.
    """
    
    output_dict = {
        rank_pred
        + 1: {
            "code": predictions[0][prediction_index][rank_pred].replace(
                "__label__", ""
            ),
            "proba": float(predictions[1][prediction_index][rank_pred]),
        }
        for rank_pred in range(nb_echoes_max)
    }

    response = output_dict | {
        "confidence": output_dict[1]["probabilite"]
        - float(predictions[1][prediction_index][1])
    }
    return response

