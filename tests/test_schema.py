from os import environ

import pytest

import greenplumpython as gp
from greenplumpython.builtins.functions import generate_series
from tests import db


@pytest.fixture
def t(db: gp.Database):
    db._execute("DROP TABLE IF EXISTS test.test_t", has_results=False)
    db._execute("DROP TABLE IF EXISTS test.test_t2", has_results=False)
    db.assign(id=lambda: generate_series(0, 9)).save_as(
        table_name="test_t", column_names=["id"], schema="test"
    )
    t = db.create_dataframe(table_name="test_t", schema="test")
    return t


def test_schema_get_slice(db: gp.Database, t: gp.DataFrame):
    assert len(list(t[2:5])) == 3


def test_schema_get_columns(db: gp.Database, t: gp.DataFrame):
    assert len(list(t[["id"]])) == 10


def test_schema_get_cond(db: gp.Database, t: gp.DataFrame):
    assert len(list(t[lambda t: t["id"] < 5])) == 5


def test_schema_join(db: gp.Database, t: gp.DataFrame):
    db.assign(id=lambda: generate_series(0, 9)).save_as(
        table_name="test_t2", column_names=["id"], schema="test"
    )
    t2 = db.create_dataframe(table_name="test_t2", schema="test")
    ret: gp.DataFrame = t.join(
        t2,
        cond=lambda s, o: s["id"] == o["id"],
        other_columns={"id": "id_2"},
    )
    assert len(list(ret)) == 10


def test_schema_self_join_on(db: gp.Database, t: gp.DataFrame):
    ret: gp.DataFrame = t.join(
        t,
        on=["id"],
        self_columns={"id"},
        other_columns={"id"},
    )
    assert len(list(ret)) == 10


def test_schema_self_join_cond(db: gp.Database, t: gp.DataFrame):
    ret: gp.DataFrame = t.join(
        t,
        cond=lambda s, o: s["id"] == o["id"],
        other_columns={"id": "id_1"},
    )
    assert len(list(ret)) == 10


def test_schema_join_same_name(db: gp.Database, t: gp.DataFrame):
    # This testcase tests scenario when joining two tables
    # with same table name but in two different schema.
    # One doesn't specify schema name (when in default schema or temporary table)
    db.assign(id=lambda: generate_series(-9, 0)).save_as(
        table_name="test_t", column_names=["id"], temp=True
    )
    t2 = db.create_dataframe(table_name="test_t")
    ret: gp.DataFrame = t.join(
        t2,
        cond=lambda s, o: s["id"] == o["id"],
        other_columns={"id": "id_1"},
    )
    assert len(list(ret)) == 1


def test_schema_join_same_name_with_schemas(db: gp.Database, t: gp.DataFrame):
    # This testcase tests scenario when joining two tables
    # with same table name but in two different schema.
    # Both have specifed schema name
    db._execute("DROP TABLE IF EXISTS public.test_t", has_results=False)
    db.assign(id=lambda: generate_series(-9, 0)).save_as(
        table_name="test_t", column_names=["id"], schema="public"
    )
    t2 = db.create_dataframe(table_name="test_t", schema="public")
    ret: gp.DataFrame = t.join(
        t2,
        cond=lambda s, o: s["id"] == o["id"],
        other_columns={"id": "id_1"},
    )
    assert len(list(ret)) == 1


def test_schema_distinct(db: gp.Database, t: gp.DataFrame):
    result = list(t.distinct_on("id"))
    assert len(result) == 10


def test_schema_in(db: gp.Database, t: gp.DataFrame):
    assert len(list(t[lambda t: t["id"].in_([1, 2, 3])])) == 3


def test_schema_group_assign(db: gp.Database, t: gp.DataFrame):
    ret = t.assign(is_even=lambda t: t["id"] % 2 == 0)
    count = gp.aggregate_function("count")
    results = ret.group_by("is_even").assign(count=lambda row: count(row["*"]))
    assert len(list(results)) == 2


def test_schema_order(db: gp.Database, t: gp.DataFrame):
    assert len(list(t.order_by("id")[:])) == 10


def test_type_schema(db: gp.Database):
    db._execute(
        f"""
        DROP TYPE IF EXISTS test.complex_number CASCADE;
        CREATE TYPE test.complex_number AS (r float8, i float8);
        """,
        has_results=False,
    )
    complex_type = gp.type_("complex_number", schema="test")
    result = db.assign(complex=lambda: complex_type("(1, 2)"))
    for row in result:
        assert row["complex"] == {"i": 2, "r": 1}
