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
