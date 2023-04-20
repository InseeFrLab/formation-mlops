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
            f"Failed to fetch model {model_name} version "
            f"{model_version}: {str(error)}"
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


def preprocess_batch(query: Dict, nb_echoes_max: int) -> Dict:
    """
    Preprocesses a batch of data in a dictionary format for prediction.

    Args:
        query (dict): A dictionary containing the batch of data.
        nb_echoes_max (int): The maximum number of echoes allowed.
    Returns:
        Dict: A dictionary containing the preprocessed data ready for further
        processing.
    Raises:
        HTTPException: If the 'text_description' field is missing for any
            liasses in the batch, a HTTPException is raised with a 400
            status code and a detailed error message.
    """
    df = pd.DataFrame(query)
    df = df.apply(lambda x: x.str.strip())
    df = df.replace(["null", "", "NA", "NAN", "nan", "None"], np.nan)

    if df["text_description"].isna().any():
        matches = df.index[df["text_description"].isna()].to_list()
        raise HTTPException(
            status_code=400,
            detail=(
                "The descriptions is missing for some entries."
                f"See line(s): {*matches,}"
            ),
        )

    df.rename(
        columns={"text_description": "TEXT_FEATURE"},
        inplace=True,
    )

    query = {
        "query": df.to_dict("list"),
        "k": nb_echoes_max,
    }
    return query


def process_response(
    predictions: Tuple,
    prediction_index: int,
    nb_echoes_max: int,
    prob_min: float,
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
    k = nb_echoes_max
    if predictions[1][prediction_index][-1] < prob_min:
        k = np.min(
            [
                np.argmax(
                    np.logical_not(predictions[1][prediction_index] > prob_min)
                ),
                nb_echoes_max,
            ]
        )

    output_dict = {
        rank_pred
        + 1: {
            "code": predictions[0][prediction_index][rank_pred].replace(
                "__label__", ""
            ),
            "probabilite": float(predictions[1][prediction_index][rank_pred]),
        }
        for rank_pred in range(k)
    }

    try:
        response = output_dict | {
            "IC": output_dict[1]["probabilite"]
            - float(predictions[1][prediction_index][1])
        }
        return response
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "message": "The minimal probability requested is higher "
                "than the highest prediction probability of the model."
            },
        )
