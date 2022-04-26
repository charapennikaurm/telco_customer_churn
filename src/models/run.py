from hyperopt import hp
from pyspark.sql import SparkSession
from trainer import Trainer

spark = SparkSession.builder.appName("telco_models").getOrCreate()

search_space = hp.choice(
    "classifier_type",
    [
        {
            "type": "rf",
            "maxDepth": hp.choice("maxDepth", range(1, 20)),
            "numTrees": hp.choice("numTrees", range(10, 100)),
        },
        {
            "type": "log_reg",
            "regParam": hp.uniform("regParam", 0.0, 0.3),
            "elasticNetParam": hp.uniform("elasticNetParam", 0.0, 0.3),
        },
        {
            "type": "gbdt",
            "maxDepth": hp.choice("maxDepth_gbdt", range(1, 20)),
            "maxIter": hp.choice("maxIter", range(20, 30)),
            "stepSize": hp.uniform("stepSize", 0.0, 0.1),
        },
        {
            "type": "svm",
        },
    ],
)

trainer = Trainer(
    spark=spark,
    path_to_dataset="../../data/raw/dataset.csv",
    search_space=search_space,
    save_model_path="../../models/test_model",
    max_evals=250,
)

trainer.run()
