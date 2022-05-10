import os
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import boto3
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from knockknock import telegram_sender
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


def upload_spark_model_to_s3(model, bucket_name, save_model_path):
    s3_folder_name = save_model_path
    if s3_folder_name[-1] != os.sep:
        s3_folder_name = s3_folder_name + os.sep

    with TemporaryDirectory() as tmpdir:
        model.write().overwrite().save(tmpdir)
        for subdir, _, files in os.walk(tmpdir):
            for file in files:
                full_path = os.path.join(subdir, file)
                bucket = boto3.resource('s3').Bucket(bucket_name)
                s3_path = s3_folder_name + full_path[len(tmpdir) + 1 :]
                bucket.upload_file(full_path, s3_path)


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
        s3_bucket_name: Optional[str] = None,
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

            s3_bucket_name(Optional[str]): Defaults to None.
                Name of s3 bucket to save model to. If None model will be saved locally.
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

        if s3_bucket_name is None:
            model.write().overwrite().save(save_model_path)
        else:
            upload_spark_model_to_s3(model, s3_bucket_name, save_model_path)

        return (
            "Model selection ended successfully."
            + f"Best params: {best}\n"
            + f"{self.metric_name} on test part of dataset: {abs(score)}\n"
        )
