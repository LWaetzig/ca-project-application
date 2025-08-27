import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


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
        
        # Health check - verify database is accessible
        self._health_check()

    def _health_check(self) -> None:
        """Run a health check to verify the database is running and accessible.

        Raises:
            ConnectionError: If the database is not reachable.
            ConnectionError: If the database is not accessible.
        """
        try:
            df = pd.read_sql_query("SELECT * FROM open_discourse.speeches LIMIT 1;", self.engine)
            assert not df.empty, "No data found in speeches table."
            print("INFO: Database is accessible")
        except SQLAlchemyError as e:
            raise ConnectionError(
                f"Database health check failed. Unable to connect to "
                f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Database health check failed due to unexpected error: {str(e)}"
            ) from e

    def fetch_data(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.engine)
