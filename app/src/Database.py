import pandas as pd
from sqlalchemy import create_engine


class Database:

    def __init__(
        self, user: str, password: str, host: str, port: int, database: str
    ) -> None:
        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )

    def fetch_data(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.engine)
