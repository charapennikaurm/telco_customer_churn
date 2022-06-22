import json
import os
import time

import boto3
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StringType, StructField, StructType

from src.data import transform_dataset


class InferenceRunner:
    COLUMNS = [
        'customerID',
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'tenure',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod',
        'MonthlyCharges',
        'TotalCharges',
    ]

    def __init__(
        self,
        spark: SparkSession,
        path_to_model: str,
        kinesis_stream_name: str,
        s3_bucket_name: str,
    ):
        self.kinesis = (
            spark.readStream.format('kinesis')
            .option('streamName', kinesis_stream_name)
            .option('region', os.environ['AWS_REGION'])
            .option(
                'endpointUrl',
                f'https://kinesis.{os.environ["AWS_REGION"]}.amazonaws.com/',
            )
            .option('startingPosition', 'LATEST')
            .option('awsAccessKeyId', os.environ['AWS_ACCESS_KEY_ID'])
            .option('awsSecretKey', os.environ['AWS_SECRET_ACCESS_KEY'])
            .load()
        )

        self.schema = StructType(
            [StructField(column, StringType()) for column in InferenceRunner.COLUMNS]
        )

        self.model = PipelineModel.load(path_to_model)
        self.s3_bucket_name = s3_bucket_name

    def run(self, trigger_once: bool = False):
        def process_batch(df, batch_id):
            if df.count() == 0:
                return
            ids = df.select('customerID').rdd.flatMap(lambda x: x).collect()
            data = transform_dataset(df)
            pred = self.model.transform(data)
            pred = pred.select('prediction').rdd.flatMap(lambda x: x).collect()
            res = {id_: churn for id_, churn in zip(ids, pred)}
            s3 = boto3.resource('s3')
            s3.Object(self.s3_bucket_name, f'prediction_{time.time()}.json').put(
                Body=(bytes(json.dumps(res).encode('UTF-8')))
            )

        self.kinesis.select(col('data').cast(StringType())).select(
            from_json('data', self.schema).alias('data')
        ).select(
            [f"data.{column}" for column in InferenceRunner.COLUMNS]
        ).writeStream.foreachBatch(
            process_batch
        ).trigger(
            once=trigger_once
        ).start().awaitTermination()
