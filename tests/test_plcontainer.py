from dataclasses import dataclass

import pytest

import greenplumpython as gp
from tests import db


@dataclass
class Int:
    i: int


@dataclass
class Pair:
    i: int
    j: int


@pytest.fixture
def t(db: gp.Database):
    rows = [(i, i) for i in range(10)]
    return db.create_dataframe(rows=rows, column_names=["a", "b"])


@gp.create_function(language_handler="plcontainer", runtime="plc_python_example")
def add_one(x: list[Int]) -> list[Int]:
    return [{"i": arg["i"] + 1} for arg in x]


def test_simple_func(db: gp.Database):
    assert (
        len(
            list(
                db.create_dataframe(columns={"i": range(10)}).apply(
                    lambda t: add_one(t), expand=True
                )
            )
        )
        == 10
    )


def test_func_no_input(db: gp.Database):

    with pytest.raises(Exception) as exc_info:  # no input data for func raises Exception
        db.create_dataframe(columns={"i": range(10)}).apply(lambda _: add_one(), expand=True)
    assert "No input data specified, please specify a DataFrame or Columns" in str(exc_info.value)


def test_func_column(db: gp.Database, t: gp.DataFrame):
    @gp.create_function(language_handler="plcontainer", runtime="plc_python_example")
    def add(x: list[Pair]) -> list[Int]:
        return [{"i": arg["i"] + arg["j"]} for arg in x]

    assert len(list(t.apply(lambda t: add(t["a"], t["b"]), expand=True))) == 10
