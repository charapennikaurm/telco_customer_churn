# TELCO CUSTOMER CHURN

## About
Solution for customer churn prediction.
You can find used dataset [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Built with
- Python 3.9
- Spark 3.2.1

## Getting started
To get a local copy of the repository please run the following commands on your terminal:
```
$ cd <folder>
$ git clone https://github.com/charapennikaurm/telco_customer_churn
```
#### Install python dependencies
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
#### Install required Spark jars
```
$ wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.12.211/aws-java-sdk-1.12.211.jar -P $SPARK_HOME/jars/
$ wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-s3/1.12.211/aws-java-sdk-s3-1.12.211.jar -P $SPARK_HOME/jars/
$ wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-core/1.12.211/aws-java-sdk-core-1.12.211.jar -P $SPARK_HOME/jars/
$ wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-dynamodb/1.12.211/aws-java-sdk-dynamodb-1.12.211.jar -P $SPARK_HOME/jars/
$ wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.1/hadoop-aws-3.3.1.jar -P $SPARK_HOME/jars/
$ wget https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kinesis-asl_2.13/3.2.1/spark-streaming-kinesis-asl_2.13-3.2.1.jar -P $SPARK_HOME/jars/
$ wget https://repo1.maven.org/maven2/com/qubole/spark/spark-sql-kinesis_2.12/1.2.0_spark-3.0/spark-sql-kinesis_2.12-1.2.0_spark-3.0.jar -P $SPARK_HOME/jars/
```

## Model training
To run training, put your own env variables in `src/models/example.env` and then run following commands:
```
$ source src/models/example.env
$ python src/models/run.py
```
#### Notes
+ [Hyperopt](https://github.com/hyperopt/hyperopt) is used for hyperparameter optimization
+ [Knock Knock](https://github.com/huggingface/knockknock) is used for telegram notifications about training
+ Best trained model can be saved locally as well as to Amazon S3


## Inference
### Local inference
This options starts a process that connects to AWS Kinesis using SparkStreaming.
The code is located in `src/inference`
To run it set env variables in file `src/inference/example.env` and then run following commands:
```
$ source src/inference/example.env
$ python src/inference/inference.py
```

### AWS Lambda Inference
The code presented in `src/lambda_inference` is used for runnig inference using AWS Lambda. Lambda function is triggered Kinesis Stream when records arrives.
Docker container is used because PySpark is too big to upload code to AWS Lambda as zip-package.
