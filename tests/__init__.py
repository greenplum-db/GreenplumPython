from os import environ

import pytest

import greenplumpython as gp


@pytest.fixture()
def db():
    # for the connection both work for GitHub Actions and concourse
    db = gp.database(
        params={
            "host": "localhost",
            "dbname": environ.get("TESTDB", "gpadmin"),
            "user": environ.get("PGUSER"),
            "password": environ.get("PGPASSWORD"),
        }
    )
    yield db
    db.close()


@pytest.fixture()
def con():
    host = "localhost"
    dbname = environ.get("TESTDB", "gpadmin")
    con = f"postgresql://{host}/{dbname}"
    yield con


gp.config.print_sql = True
