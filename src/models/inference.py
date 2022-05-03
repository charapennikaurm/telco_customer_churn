import json
import os
import time

import boto3
import findspark
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StringType, StructField, StructType

from src.data import transform_dataset

PATH_TO_MODEL = "../../models/test_model"
KINESIS_STREAM = "telco-inference"
S3_BUCKET = "test-iot-s3-bucket"


def process_batch(df, batch_id):
    if df.count() == 0:
        return
    ids = df.select('customerID').rdd.flatMap(lambda x: x).collect()
    data = transform_dataset(df)
    model = PipelineModel.load(PATH_TO_MODEL)
    pred = model.transform(data)
    pred = pred.select('prediction').rdd.flatMap(lambda x: x).collect()
    res = {id_: churn for id_, churn in zip(ids, pred)}
    s3 = boto3.resource('s3')
    s3.Object(S3_BUCKET, f'prediction_{time.time()}.json').put(
        Body=(bytes(json.dumps(res).encode('UTF-8')))
    )


if __name__ == "__main__":
    findspark.init()

    spark = (
        SparkSession.builder.appName('telco-inference')
        .config(
            "spark.jars",
            f"{os.environ['SPARK_HOME']}/jars/spark-sql-kinesis_2.12-1.2.0_spark-3.0.jar",
        )
        .config(
            "spark.jars",
            f"{os.environ['SPARK_HOME']}/jars/spark-streaming-kinesis-asl_2.13-3.2.1.jar",
        )
        .enableHiveSupport()
        .getOrCreate()
    )

    kinesis = (
        spark.readStream.format('kinesis')
        .option('streamName', KINESIS_STREAM)
        .option('region', 'us-west-2')
        .option('endpointUrl', 'https://kinesis.us-west-2.amazonaws.com/')
        .option('startingPosition', 'LATEST')
        .option('awsAccessKeyId', os.environ['AWS_ACCESS_KEY'])
        .option('awsSecretKey', os.environ['AWS_SECRET_KEY'])
        .load()
    )

    columns = [
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
    schema = StructType([StructField(column, StringType()) for column in columns])

    kinesis.selectExpr('CAST(data AS STRING)').select(
        from_json('data', schema).alias('data')
    ).select('data.*').writeStream.foreachBatch(
        process_batch
    ).start().awaitTermination()
