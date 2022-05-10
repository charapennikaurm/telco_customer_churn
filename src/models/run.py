import os

from hyperopt import hp
from pyspark.sql import SparkSession
from trainer import Trainer

if __name__ == '__main__':
    spark = (
        SparkSession.builder.appName("telco_models")
        .config("spark.hadoop.fs.s3a.path.style.access", True)
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY'])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_KEY'])
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            f"s3-{os.environ['AWS_REGION']}.amazonaws.com",
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("com.amazonaws.services.s3.enableV4", True)
        .config(
            "spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        )
        .getOrCreate()
    )

    search_space_rf = hp.choice(
        "classifier_type",
        [
            {
                "type": "rf",
                "maxDepth": hp.choice("maxDepth", range(1, 20)),
                "numTrees": hp.choice("numTrees", range(10, 100)),
            },
        ],
    )

    search_space_lr = hp.choice(
        "classifier_type",
        [
            {
                "type": "log_reg",
                "regParam": hp.uniform("regParam", 0.0, 0.3),
                "elasticNetParam": hp.uniform("elasticNetParam", 0.0, 0.3),
            },
        ],
    )

    search_space_gbdt = hp.choice(
        "classifier_type",
        [
            {
                "type": "gbdt",
                "maxDepth": hp.choice("maxDepth_gbdt", range(1, 20)),
                "maxIter": hp.choice("maxIter", range(20, 30)),
                "stepSize": hp.uniform("stepSize", 0.0, 0.1),
            },
        ],
    )

    search_space_svm = hp.choice(
        "classifier_type",
        [
            {"type": "svm", "regParam": hp.uniform("regParam_svm", 0.0, 0.3)},
        ],
    )

    trainer = Trainer(
        spark=spark,
        path_to_train="../../data/processed/train.csv",
        path_to_test="../../data/processed/test.csv",
    )

    # find best params for random forrest
    trainer.run(
        search_space=search_space_rf,
        save_model_path="/models/rf",
        max_evals=40,
        s3_bucket_name='test-iot-s3-bucket',
    )

    # find best params for logistic regression
    trainer.run(
        search_space=search_space_lr,
        save_model_path="/models/lr",
        max_evals=40,
        s3_bucket_name='test-iot-s3-bucket',
    )

    # find best params for gbdt
    trainer.run(
        search_space=search_space_gbdt,
        save_model_path="/models/gbdt",
        max_evals=40,
        s3_bucket_name='test-iot-s3-bucket',
    )

    # find best params for svm
    trainer.run(
        search_space=search_space_svm,
        save_model_path="/models/svm",
        max_evals=40,
        s3_bucket_name='test-iot-s3-bucket',
    )
