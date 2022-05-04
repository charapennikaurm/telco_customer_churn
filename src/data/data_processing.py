from itertools import chain
from typing import Any, Dict, List, Optional

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, create_map, lit, when

CATEGORICAL_VARIABLES = {
    "PhoneService": ['No', 'Yes'],
    "StreamingTV": ['No', 'Yes', 'No internet service'],
    "gender": ['Female', 'Male'],
    "MultipleLines": ['No phone service', 'No', 'Yes'],
    "SeniorCitizen": ['0', '1'],
    "Contract": ['Month-to-month', 'One year', 'Two year'],
    "Partner": ['No', 'Yes'],
    "DeviceProtection": ['No', 'Yes', 'No internet service'],
    "OnlineSecurity": ['No', 'Yes', 'No internet service'],
    "StreamingMovies": ['No', 'Yes', 'No internet service'],
    "PaperlessBilling": ['No', 'Yes'],
    "Dependents": ['No', 'Yes'],
    "PaymentMethod": [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)',
    ],
    "OnlineBackup": ['Yes', 'No', 'No internet service'],
    "TechSupport": ['No', 'Yes', 'No internet service'],
    "InternetService": ['DSL', 'Fiber optic', 'No'],
}


def make_dummy(
    df: DataFrame, column: str, options: Optional[List[Any]] = None
) -> DataFrame:
    data = df
    if options is None:
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


def make_dummies_with_options(
    df: DataFrame, columns: Dict[str, List[Any]]
) -> DataFrame:
    data = df
    for column, options in columns.items():
        data = make_dummy(data, column, options)
    return data


def change_column_type(df: DataFrame, column: str, new_type: str) -> DataFrame:
    return df.withColumn(column, df[column].cast(new_type))


def map_column_values(df: DataFrame, column: str, mapping: Dict) -> DataFrame:
    map_fn = create_map([lit(x) for x in chain(*mapping.items())])
    return df.withColumn(column, map_fn[col(column)])


def transform_dataset(dataset: DataFrame) -> DataFrame:
    """
    Performs dataset transform to be able to work with this dataset
    later, making dummy variables from categorical ones and cast to
    correct types. Also removes `customerID` column, because
    it is not used for prediction
    """
    transformed_dataset = change_column_type(dataset, "tenure", "int")
    transformed_dataset = change_column_type(
        transformed_dataset, "MonthlyCharges", "double"
    )
    transformed_dataset = change_column_type(
        transformed_dataset, "TotalCharges", "double"
    )
    if 'Churn' in transformed_dataset.columns:
        transformed_dataset = map_column_values(
            transformed_dataset, "Churn", {"Yes": 1, "No": 0}
        )

        transformed_dataset = change_column_type(transformed_dataset, "Churn", "int")

    transformed_dataset = transformed_dataset.replace("?", None).dropna(how="any")

    transformed_dataset = transformed_dataset.drop("customerID")

    transformed_dataset = make_dummies_with_options(
        transformed_dataset, CATEGORICAL_VARIABLES
    )

    transformed_dataset = transformed_dataset.select(
        sorted(transformed_dataset.columns)
    )

    return transformed_dataset
