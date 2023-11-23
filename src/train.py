"""
Main script.
"""
import s3fs
import sys
import fasttext
import mlflow
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
from constants import TEXT_FEATURE, Y, DATA_PATH, LABEL_PREFIX
from utils import write_training_data
from fasttext_wrapper import FastTextWrapper


def load_data():
    """
    Load data for training and test.
    """
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
        anon=True
    )
    df = pq.ParquetDataset(DATA_PATH, filesystem=fs).read_pandas().to_pandas()
    return df.sample(frac=0.1)


def train(
    remote_server_uri,
    experiment_name,
    run_name,
    dim,
    lr,
    epoch,
    wordNgrams,
    minn,
    maxn,
    minCount,
    bucket,
    thread,
):
    """
    Train a FastText model.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Get preprocessor and data
        preprocessor = Preprocessor()
        df = load_data()

        # Preprocess data to train and test a FastText model
        df = preprocessor.clean_text(df, TEXT_FEATURE)

        X_train, X_test, y_train, y_test = train_test_split(
            df[TEXT_FEATURE],
            df[Y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        # Train the model and log to MLflow tracking server
        params = {
            "dim": dim,
            "lr": lr,
            "epoch": epoch,
            "wordNgrams": wordNgrams,
            "minn": minn,
            "maxn": maxn,
            "minCount": minCount,
            "bucket": bucket,
            "thread": thread,
            "loss": "ova",
            "label_prefix": LABEL_PREFIX,
        }

        # Write training data in a .txt file (fasttext-specific)
        training_data_path = write_training_data(df_train, params)

        # Train the fasttext model
        model = fasttext.train_supervised(
            training_data_path,
            **params,
            verbose=2
        )

        # Save model for logging
        model_path = f"models/{run_name}.bin"
        model.save_model(model_path)

        artifacts = {
            "model_path": model_path,
            "train_data": training_data_path,
        }

        inference_params = {
            "k": 1,
        }
        # Infer the signature including parameters
        signature = mlflow.models.infer_signature(
            params=inference_params,
        )

        mlflow.pyfunc.log_model(
            artifact_path=run_name,
            python_model=FastTextWrapper(),
            code_path=[
                "src/fasttext_wrapper.py",
                "src/preprocessor.py",
                "src/constants.py",
            ],
            artifacts=artifacts,
            signature=signature
        )

        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Evaluation
        test_texts = []
        for item in df_test.iterrows():
            formatted_item = item[1][TEXT_FEATURE]
            test_texts.append(formatted_item)

        predictions = model.predict(test_texts, k=1)
        predictions = [x[0].replace(LABEL_PREFIX, "") for x in predictions[0]]

        booleans = [
            prediction == label
            for prediction, label in zip(predictions, df_test[Y])
        ]
        accuracy = sum(booleans) / len(booleans)

        # Log accuracy
        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    dim = int(sys.argv[4])
    lr = float(sys.argv[5])
    epoch = int(sys.argv[6])
    wordNgrams = int(sys.argv[7])
    minn = int(sys.argv[8])
    maxn = int(sys.argv[9])
    minCount = int(sys.argv[10])
    bucket = int(sys.argv[11])
    thread = int(sys.argv[12])

    train(
        remote_server_uri,
        experiment_name,
        run_name,
        dim,
        lr,
        epoch,
        wordNgrams,
        minn,
        maxn,
        minCount,
        bucket,
        thread,
    )
