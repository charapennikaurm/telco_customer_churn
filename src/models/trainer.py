import os
from typing import Dict

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

from src.data import transform_dataset


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
    Args:
        spark (pyspark.sql.SparkSession):
            SparkSession, used to load datasets.
        path_to_train(str):
            Path to csv with training data.
        path_to_test(str):
            Path to csv with test data.
        train_val_ratio(float): Defaults to float.
            Ratio of training data, that will be used for training models during
            hyperparameters search. Must be in (0.0, 1.0)
        metric_name(str): Defaults to 'f1'
            Metric that will be used to compare models with different hyperparameters.
        greater_is_better(bool): Defaults to True.
            If True, then greater metric value is better.
    """

    def __init__(
        self,
        spark: SparkSession,
        path_to_train: str,
        path_to_test: str,
        train_val_ratio: float = 0.8,
        metric_name: str = "f1",
        greater_is_better: bool = True,
    ) -> None:
        if train_val_ratio >= 1 or train_val_ratio <= 0:
            raise ValueError(
                f'train_val_ratio must be in (0, 1), but got {train_val_ratio}'
            )
        train = transform_dataset(spark.read.csv(path_to_train, header=True))
        self.train, self.val = train.randomSplit([train_val_ratio, 1 - train_val_ratio])
        self.test = transform_dataset(spark.read.csv(path_to_test, header=True))
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better

    def _run_hyperparam_search(self, search_space, max_evals):
        trials = Trials()
        fmin(
            self._objective,
            search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
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
        save_model_path: str,
        search_space,
        max_evals: int = 100,
    ) -> str:
        """
        Run search of the best hyperparameters, train model on whole train dataset and
        save best option.
        Args:
            save_model_path(str):
                Path where to save model

            search_space:
                search space for hyperparameter optimization

            max_evals(int): Defaults to 100.
                Maximum number of tries to find the best hyperparameters
        Returns:
            str: info about training results
        """
        if max_evals <= 0:
            raise ValueError(f"max_evals must be positive, but got {max_evals}")

        best = self._run_hyperparam_search(search_space, max_evals)
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
            model.save(save_model_path)
        except Py4JJavaError:
            error_msg = "Did not manage to save model"

        return (
            "Model selection ended successfully."
            + f"Best params: {best}\n"
            + f"{self.metric_name} on test part of dataset: {abs(score)}\n"
            + error_msg
        )
