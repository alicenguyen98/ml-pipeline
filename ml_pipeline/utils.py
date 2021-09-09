import pandas as pd

def optimize_numeric_dtype(df: pd.DataFrame, from_dtype: str, to_dtype: str):
	cols = df.select_dtypes(include=[from_dtype]).columns.tolist()
	df[cols] = df[cols].apply(pd.to_numeric, downcast=to_dtype)
	return df

def optimize_category_dtype(df: pd.DataFrame, columns: list):
	for col in columns:
		df[col] = df[col].astype('category')
	return df