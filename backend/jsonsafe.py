"""
NaN-safe conversion of a DataFrame to a list of record dicts.

Plain `df.to_dict("records")` leaves pandas NaN as float('nan'), which
FastAPI's JSONResponse rejects (`allow_nan=False`) with "Out of range float
values are not JSON compliant: nan". Routing through DataFrame.to_json()
converts NaN to JSON null first, matching how Streamlit's st.dataframe
displayed missing values (blank) without changing any underlying values.
"""
import json


def records(df):
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))
