from typing import Dict, List

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import UserDefinedFunction, col, when


def make_dummy(df: DataFrame, column: str) -> DataFrame:
    data = df
    options = data.select(column).distinct().rdd.flatMap(lambda x: x).collect()
    options_expr = [
        when(col(column) == opt, 1).otherwise(0).alias(f"{column}_" + opt)
        for opt in options
    ]
    data = data.select(*data.columns, *options_expr)
    return data.drop(column)


def make_dummies(df: DataFrame, columns: List[str]) -> DataFrame:
    data = df
    for column in columns:
        data = make_dummy(data, column)
    return data


def change_column_type(df: DataFrame, column: str, new_type: str) -> DataFrame:
    return df.withColumn(column, df[column].cast(new_type))


def map_column_values(df: DataFrame, column: str, mapping: Dict) -> DataFrame:
    map_fn = UserDefinedFunction(lambda x: mapping[x])
    return df.withColumn(column, map_fn(column))
