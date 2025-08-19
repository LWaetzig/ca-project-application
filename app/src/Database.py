import pandas as pd
from sqlalchemy import create_engine


class Database:

    def __init__(
        self, user: str, password: str, host: str, port: int, database: str
    ) -> None:
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

        # init SQLAlchemy engine
        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    def fetch_data(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.engine)
