import os
from typing import Dict, Tuple

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from knockknock import telegram_sender
from py4j.protocol import Py4JJavaError
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from src.data import change_column_type, make_dummies, map_column_values


def transform_dataset(dataset: DataFrame) -> DataFrame:
    """
    Performs dataset transform to be able to work with this dataset
    later, making dummy variables from categorical ones and cast to
    correct types. Also removes `customerID` column, because
    it is not used for prediction
    """
    categorical_columns = [
        "PhoneService",
        "StreamingTV",
        "gender",
        "MultipleLines",
        "SeniorCitizen",
        "Contract",
        "Partner",
        "DeviceProtection",
        "OnlineSecurity",
        "StreamingMovies",
        "PaperlessBilling",
        "Dependents",
        "PaymentMethod",
        "OnlineBackup",
        "TechSupport",
        "InternetService",
    ]

    transformed_dataset = change_column_type(dataset, "tenure", "int")
    transformed_dataset = change_column_type(
        transformed_dataset, "MonthlyCharges", "double"
    )
    transformed_dataset = change_column_type(
        transformed_dataset, "TotalCharges", "double"
    )
    transformed_dataset = map_column_values(
        transformed_dataset, "Churn", {"Yes": 1, "No": 0}
    )
    transformed_dataset = change_column_type(transformed_dataset, "Churn", "int")

    transformed_dataset = transformed_dataset.replace("?", None).dropna(how="any")

    transformed_dataset = transformed_dataset.drop("customerID")

    transformed_dataset = make_dummies(transformed_dataset, categorical_columns)

    return transformed_dataset


def get_classifier(classifier_name: str):

    if classifier_name == "gbdt":
        classifier = GBTClassifier
    elif classifier_name == "svm":
        classifier = LinearSVC
    elif classifier_name == "rf":
        classifier = RandomForestClassifier
    elif classifier_name == "log_reg":
        classifier = LogisticRegression
    else:
        raise ValueError(
            "Unknown algorithm. Options:\n"
            + "'gbdt' for GBTCClassifier\n"
            + "'svm' for LinearSVC\n"
            + "'rf' for RandomForrestClassifier\n"
            + "'log_reg' for LogisticRegression"
        )
    return classifier


class Trainer:
    """
    Class to run best model selection on raw dataset.
    """

    def __init__(
        self,
        spark: SparkSession,
        path_to_dataset: str,
        search_space,
        save_model_path: str,
        split_seed: int = 17,
        train_val_test_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        metric_name: str = "f1",
        greater_is_better: bool = True,
        max_evals: int = 100,
    ) -> None:
        dataset = spark.read.csv(path_to_dataset, header=True)
        dataset = transform_dataset(dataset)
        self.train, self.val, self.test = dataset.randomSplit(
            list(train_val_test_ratio), seed=split_seed
        )
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        if max_evals <= 0:
            raise ValueError("max_evals must be positive")
        self.max_evals = max_evals
        self.search_space = search_space
        self.save_model_path = save_model_path

    def _run_hyperparam_search(self, search_space):
        trials = Trials()
        fmin(
            self._objective,
            search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
        )

        return trials.results[np.argmin([r["loss"] for r in trials.results])]["params"]

    def _build_pipeline(self, params: Dict) -> Pipeline:
        classifier_name = params["type"]

        classifier = get_classifier(classifier_name)(
            **{k: params[k] for k in params.keys() if k != "type"}
        )
        classifier.setFeaturesCol("features")
        classifier.setLabelCol("Churn")

        vector_assembler = VectorAssembler(
            inputCols=list(set(self.train.columns) - {"Churn"}),
            outputCol="features",
        )

        pipeline_stages = [vector_assembler, classifier]

        if classifier_name in ["svm", "log_reg"]:
            scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
            classifier.setFeaturesCol("features_scaled")
            pipeline_stages.insert(1, scaler)

        return Pipeline(stages=pipeline_stages)

    def _objective(self, params: Dict) -> Dict:
        # function to minimize with hyperopt

        pipeline = self._build_pipeline(params)

        predictions = pipeline.fit(self.train).transform(self.val)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="Churn",
            predictionCol="prediction",
            metricName=self.metric_name,
        )

        score = evaluator.evaluate(predictions)

        result = {
            "loss": -score if self.greater_is_better else score,
            "params": params,
            "status": STATUS_OK,
        }

        return result

    @telegram_sender(os.environ['BOT_TOKEN'], int(os.environ['TG_CHAT_ID']))
    def run(
        self,
    ) -> str:
        best = self._run_hyperparam_search(self.search_space)
        print(best)
        pipeline = self._build_pipeline(best)
        df = self.train.unionByName(self.val)
        model = pipeline.fit(df)
        predictions = model.transform(self.test)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="Churn",
            predictionCol="prediction",
            metricName=self.metric_name,
        )

        score = evaluator.evaluate(predictions)

        error_msg = ''
        try:
            model.save(self.save_model_path)
        except Py4JJavaError:
            error_msg = "Did not manage to save model"

        return (
            "Model selection ended succesfully."
            + f"Best params: {best}\n"
            + f"{self.metric_name} on test part of dataset: {abs(score)}\n"
            + error_msg
        )
