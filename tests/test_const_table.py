from os import environ

import pytest

import greenplumpython as gp
from tests import db


@pytest.fixture
def t(db: gp.Database):
    generate_series = gp.function("generate_series")
    t = (
        generate_series(0, 9, as_name="id", db=db)
        .to_table()
        .save_as("temp_table", temp=True, column_names=["id"])
    )
    return t


def test_const_table(db: gp.Database):
    rows = [(1,), (2,), (3,)]
    t = gp.values(rows, db=db)
    t = t.save_as("const_table", column_names=["id"], temp=True)
    assert sorted([tuple(row.values()) for row in t.fetch()]) == sorted(rows)

    t_cols = t.column_names().fetch()
    assert len(list(t_cols)) == 1
    for row in t_cols:
        assert row["column_name"] == "id"


def test_table_getitem_str(db: gp.Database):
    rows = [(1,), (2,), (3,)]
    t = gp.values(rows, db=db)
    t = t.save_as("const_table", temp=True, column_names=["id"])
    c = t["id"]
    assert str(c) == (t.name + ".id")


def test_table_getitem_sub_columns(db: gp.Database):
    # fmt: off
    rows = [(1, 2,), (1, 3,), (2, 2,), (3, 1,), (3, 4,)]
    # fmt: on
    t = gp.values(rows, db=db)
    t = t.save_as("const_table", temp=True, column_names=["id", "num"])
    t_sub = t[["id", "num"]]
    assert t_sub.columns == ["id", "num"]


def test_table_getitem_slice_limit(db: gp.Database, t: gp.Table):
    ret = list(t[:2].fetch())
    assert len(ret) == 2


def test_table_getitem_slice_offset(db: gp.Database, t: gp.Table):
    ret = list(t[7:].fetch())
    assert len(ret) == 3


def test_table_getitem_slice_off_limit(db: gp.Database, t: gp.Table):
    ret = list(t[2:5].fetch())
    assert len(ret) == 3


def test_table_display_repr(db: gp.Database):
    # fmt: off
    rows = [(1, "Lion",), (2, "Tiger",), (3, "Wolf",), (4, "Fox")]
    # fmt: on
    t = gp.values(rows, db=db).save_as("zoo1", column_names=["id", "animal"])
    expected = (
        "| id         || animal     |\n"
        "============================\n"
        "|          1 || Lion       |\n"
        "|          2 || Tiger      |\n"
        "|          3 || Wolf       |\n"
        "|          4 || Fox        |\n"
    )
    assert str(t.order_by(t["id"]).head(4)) == expected


def test_table_display_repr_long_content(db: gp.Database):
    # fmt: off
    rows = [(1, "Lion",), (2, "Tigerrrrrrrrrrrr",), (3, "Wolf",), (4, "Fox")]
    # fmt: on
    t = gp.values(rows, db=db).save_as("zoo1", column_names=["iddddddddddddddddddd", "animal"])
    expected = (
        "| iddddddddddddddddddd || animal     |\n"
        "============================\n"
        "|          1 || Lion       |\n"
        "|          2 || Tigerrrrrrrrrrrr |\n"
        "|          3 || Wolf       |\n"
        "|          4 || Fox        |\n"
    )
    assert str(t.order_by(t["iddddddddddddddddddd"]).head(4)) == expected


def test_table_display_repr_html(db: gp.Database):
    # fmt: off
    rows = [(1, "Lion",), (2, "Tiger",), (3, "Wolf",), (4, "Fox")]
    # fmt: on
    t = gp.values(rows, db=db).save_as("zoo1", column_names=["id", "animal"])
    expected = (
        "<table>\n"
        "\t<tr>\n"
        "\t\t<th>id</th>\n"
        "\t\t<th>animal</th>\n"
        "\t</tr>\n"
        "\t<tr>\n"
        "\t\t<td>1</td>\n"
        "\t\t<td>Lion</td>\n"
        "\t</tr>\n"
        "\t<tr>\n"
        "\t\t<td>2</td>\n"
        "\t\t<td>Tiger</td>\n"
        "\t</tr>\n"
        "\t<tr>\n"
        "\t\t<td>3</td>\n"
        "\t\t<td>Wolf</td>\n"
        "\t</tr>\n"
        "\t<tr>\n"
        "\t\t<td>4</td>\n"
        "\t\t<td>Fox</td>\n"
        "\t</tr>\n"
        "</table>"
    )
    assert (t.order_by(t["id"]).head(4)._repr_html_()) == expected