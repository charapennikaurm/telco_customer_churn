import os

from inference_runner import InferenceRunner
from pyspark.sql import SparkSession

PATH_TO_MODEL = "s3a://test-iot-s3-bucket/test_model"
KINESIS_STREAM = "telco-inference"
S3_BUCKET = "test-iot-s3-bucket"


if __name__ == "__main__":
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
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY'])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_KEY'])
        .config("spark.hadoop.fs.s3a.endpoint", "s3-us-west-2.amazonaws.com")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("com.amazonaws.services.s3.enableV4", True)
        .config(
            "spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        )
        .getOrCreate()
    )

    runner = InferenceRunner(
        spark=spark,
        path_to_model=PATH_TO_MODEL,
        kinesis_stream_name=KINESIS_STREAM,
        s3_bucket_name=S3_BUCKET,
    )

    print('START')
    runner.run()
