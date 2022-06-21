import base64
import json
import os
import time

import boto3
from data_processing import transform_dataset
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

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

PATH_TO_MODEL = "s3a://test-iot-s3-bucket/test_model"
S3_BUCKET = "test-iot-s3-bucket"


def decode_message(base64_message: str):
    base64_bytes = base64_message.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)
    message = message_bytes.decode('ascii')
    return message


def lambda_handler(event, context):
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
        .config("spark.hadoop.fs.s3a.path.style.access", True)
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY_ID'])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_ACCESS_KEY'])
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            f"s3-{os.environ['AWS_REGION']}.amazonaws.com",
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("com.amazonaws.services.s3.enableV4", True)
        .config(
            "spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        )
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.shuffle.s3.enabled", True)
        .config("spark.lambda.concurrent.requests.max", 50)
        .config("spark.dynamicAllocation.enabled", True)
        .getOrCreate()
    )

    records = event['Records']
    data = [decode_message(json.dumps(record['kinesis']['data'])) for record in records]
    rdd = spark.sparkContext.parallelize(data)

    schema = StructType([StructField(column, StringType()) for column in COLUMNS])

    model = PipelineModel.load(PATH_TO_MODEL)
    s3_bucket_name = S3_BUCKET

    df = spark.read.json(rdd, schema=schema)
    if df.count() == 0:
        return {'statusCode': 200, 'body': 'OK'}
    ids = df.select('customerID').rdd.flatMap(lambda x: x).collect()
    data = transform_dataset(df)
    pred = model.transform(data)
    pred = pred.select('prediction').rdd.flatMap(lambda x: x).collect()
    res = {id_: churn for id_, churn in zip(ids, pred)}
    s3 = boto3.resource('s3')
    s3.Object(s3_bucket_name, f'prediction_{time.time()}.json').put(
        Body=(bytes(json.dumps(res).encode('UTF-8')))
    )
    return {'statusCode': 200, 'body': 'OK'}
