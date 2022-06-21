FROM public.ecr.aws/lambda/python:3.9

RUN yum -y install java-1.8.0-openjdk wget curl

RUN pip install --upgrade pip
COPY requirements_lambda.txt .
RUN pip install -r requirements_lambda.txt


ENV PYSPARK_SUBMIT_ARGS="--master local[*] pyspark-shell"
ENV JAVA_HOME="/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.312.b07-1.amzn2.0.2.x86_64/jre"
ENV PATH=${PATH}:${JAVA_HOME}/bin
ENV SPARK_HOME="/var/lang/lib/python3.9/site-packages/pyspark"
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
ENV PATH=$SPARK_HOME/python:$PATH


#requited spark dependencies
RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-s3/1.12.211/aws-java-sdk-s3-1.12.211.jar -P $SPARK_HOME/jars/
RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.12.211/aws-java-sdk-1.12.211.jar -P $SPARK_HOME/jars/
RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-core/1.12.211/aws-java-sdk-core-1.12.211.jar -P $SPARK_HOME/jars/
RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-dynamodb/1.12.211/aws-java-sdk-dynamodb-1.12.211.jar -P $SPARK_HOME/jars/
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.1/hadoop-aws-3.3.1.jar -P $SPARK_HOME/jars/
RUN wget https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kinesis-asl_2.13/3.2.1/spark-streaming-kinesis-asl_2.13-3.2.1.jar -P $SPARK_HOME/jars/
RUN wget https://repo1.maven.org/maven2/com/qubole/spark/spark-sql-kinesis_2.12/1.2.0_spark-3.0/spark-sql-kinesis_2.12-1.2.0_spark-3.0.jar -P $SPARK_HOME/jars/

RUN mkdir $SPARK_HOME/conf
RUN echo "SPARK_LOCAL_IP=127.0.0.1" > $SPARK_HOME/conf/spark-env.sh
RUN chmod 777 $SPARK_HOME/conf/spark-env.sh

COPY spark-class $SPARK_HOME/bin/spark-class
RUN chmod 777 $SPARK_HOME/bin/spark-class

COPY data_processing.py ${LAMBDA_TASK_ROOT}
COPY lambda_inference.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_inference.lambda_handler"]
