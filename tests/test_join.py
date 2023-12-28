from os import environ

import pytest

import greenplumpython as gp
from tests import db


@pytest.fixture
def t1(db: gp.Database):
    # fmt: off
    rows1 = [(1, 0, "a1",), (2, 0, "a2",), (3, 0, "a3",)]
    # fmt: on
    return db.create_dataframe(rows=rows1, column_names=["id1", "idd1", "n1"])


@pytest.fixture
def t2(db: gp.Database):
    # fmt: off
    rows2 = [(1, 0, "b1",), (2, 0, "b2",), (3, 0, "b3",)]
    # fmt: on
    return db.create_dataframe(rows=rows2, column_names=["id2", "idd2", "n2"])


@pytest.fixture
def zoo_1(db: gp.Database):
    # fmt: off
    rows = [(1, "Lion",), (2, "Tiger",), (3, "Wolf",), (4, "Fox")]
    # fmt: on
    return db.create_dataframe(rows=rows, column_names=["id", "animal"])


@pytest.fixture
def zoo_2(db: gp.Database):
    # fmt: off
    rows = [(1, "Tiger",), (2, "Lion",), (3, "Rhino",), (4, "Panther")]
    # fmt: on
    return db.create_dataframe(rows=rows, column_names=["id", "animal"])


def test_join_all_and_all_columns(db: gp.Database, t1: gp.DataFrame, t2: gp.DataFrame):
    row = next(
        iter(
            t1.join(
                t2,
                cond=lambda t1, t2: t1["id1"] == t2["id2"],
                self_columns={"*"},
                other_columns={"*"},
            )
        )
    )
    assert sorted(row.keys()) == sorted(["id1", "idd1", "n1", "id2", "idd2", "n2"])


def test_join_no_and_no_columns(db: gp.Database, t1: gp.DataFrame, t2: gp.DataFrame):
    ret = next(iter(t1.join(t2, cond=lambda t1, t2: t1["id1"] == t2["id2"])))
    assert sorted(ret.keys()) == sorted(["id1", "idd1", "n1", "id2", "idd2", "n2"])


def test_join_all_and_no_columns(db: gp.Database, t1: gp.DataFrame, t2: gp.DataFrame):
    ret = next(
        iter(
            t1.join(
                t2, cond=lambda t1, t2: t1["id1"] == t2["id2"], self_columns={}, other_columns={"*"}
            )
        )
    )
    assert sorted(ret.keys()) == sorted(["id2", "idd2", "n2"])


def test_join_all_and_one_columns(db: gp.Database, t1: gp.DataFrame, t2: gp.DataFrame):
    ret = next(
        iter(
            t1.join(
                t2,
                cond=lambda t1, t2: t1["id1"] == t2["id2"],
                self_columns={"id1"},
                other_columns={"*"},
            )
        )
    )
    assert sorted(ret.keys()) == sorted(["id1", "id2", "idd2", "n2"])


def test_join_columns_from_list(db: gp.Database, t1: gp.DataFrame, t2: gp.DataFrame):
    ret = next(
        iter(
            t1.join(
                t2,
                cond=lambda t1, t2: t1["id1"] == t2["id2"],
                self_columns=dict.fromkeys(["id1", "n1"]),
                other_columns={"idd2"},
            )
        )
    )
    assert sorted(ret.keys()) == sorted(["id1", "n1", "idd2"])


# FIXME : Test for no exist target column


def test_join_same_column_using(db: gp.Database):
    rows = [(1,), (2,), (3,)]
    t1 = db.create_dataframe(rows=rows, column_names=["id"])
    t2 = db.create_dataframe(rows=rows, column_names=["id"])
    ret = t1.join(t2, on=["id"], self_columns={"id"}, other_columns={"id"})
    assert list(next(iter(ret)).keys()) == ["id"]


def test_join_same_column_names(db: gp.Database):
    rows = [(1, 1), (2, 1), (3, 1)]
    t1 = db.create_dataframe(rows=rows, column_names=["id", "n1"])
    t2 = db.create_dataframe(rows=rows, column_names=["id", "n2"])
    ret = t1.cross_join(t2)
    with pytest.raises(Exception) as e:
        print(ret)
    assert "Duplicate column name(s) found" in str(e.value)


def test_join_on_multi_columns(db: gp.Database):
    rows = [(1, 1), (2, 1), (3, 1)]
    t1 = db.create_dataframe(rows=rows, column_names=["id", "n"])
    t2 = db.create_dataframe(rows=rows, column_names=["id", "n"])
    ret = t1.join(t2, on=["id", "n"], self_columns={"id", "n"}, other_columns={"id", "n"})
    assert sorted(next(iter(ret)).keys()) == sorted(["id", "n"])


def test_dataframe_inner_join(db: gp.Database, zoo_1: gp.DataFrame, zoo_2: gp.DataFrame):
    ret: gp.DataFrame = zoo_1.join(
        zoo_2,
        on=["animal"],
        self_columns={"animal": "animal_l", "id": "id_zoo1"},
        other_columns={"animal": "animal_r", "id": "id_zoo2"},
    )
    assert len(list(ret)) == 2
    assert sorted(next(iter(ret)).keys()) == sorted(["animal", "id_zoo1", "id_zoo2"])
    for row in ret:
        assert row["animal"] == "Lion" or row["animal"] == "Tiger"


def test_dataframe_left_join(db: gp.Database, zoo_1: gp.DataFrame, zoo_2: gp.DataFrame):
    ret = zoo_1.left_join(
        zoo_2,
        on=["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    assert len(list(ret)) == 4
    for row in ret:
        if row["animal"] != "Lion" and row["animal"] != "Tiger":
            assert row["zoo2_id"] is None


def test_dataframe_right_join(db: gp.Database, zoo_1: gp.DataFrame, zoo_2: gp.DataFrame):
    ret = zoo_1.right_join(
        zoo_2,
        on=["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    assert len(list(ret)) == 4
    for row in ret:
        if row["animal"] != "Lion" and row["animal"] != "Tiger":
            assert row["zoo1_id"] is None


def test_dataframe_full_join(db: gp.Database, zoo_1: gp.DataFrame, zoo_2: gp.DataFrame):
    ret = zoo_1.full_join(
        zoo_2,
        on=["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    assert len(list(ret)) == 6
    for row in ret:
        if row["animal"] != "Lion" and row["animal"] != "Tiger":
            assert (row["zoo1_id"] is None and row["zoo2_id"] is not None) or (
                row["zoo1_id"] is not None and row["zoo2_id"] is None
            )


def test_dataframe_full_join_with_empty(db: gp.Database):
    # fmt: off
    rows1 = [(1, 100,), (2, 200,), (3, 300,), (4, 400,)]
    rows2 = [(3, 300, 3000,), (4, 400, 4000,), (5, 500, 5000,), (6, 600, 6000)]
    # fmt: on
    l_df = db.create_dataframe(rows=rows1, column_names=["a", "b"])
    r_df = db.create_dataframe(rows=rows2, column_names=["a", "b", "c"])
    ret = l_df.full_join(
        r_df,
        self_columns={"a", "b"},
        other_columns={"a", "b", "c"},
        on=["a", "b"],
    ).order_by("a")[:]
    assert len(list(ret)) == 6
    expected = (
        "----------------\n"
        " a | b   | c    \n"
        "---+-----+------\n"
        " 1 | 100 |      \n"
        " 2 | 200 |      \n"
        " 3 | 300 | 3000 \n"
        " 4 | 400 | 4000 \n"
        " 5 | 500 | 5000 \n"
        " 6 | 600 | 6000 \n"
        "----------------\n"
        "(6 rows)\n"
    )
    assert str(ret) == expected


def test_join_natural(db: gp.Database):
    # fmt: off
    rows1 = [("Smart Phone", 1,), ("Laptop", 2,), ("DataFramet", 3,)]
    rows2 = [("iPhone", 1,), ("Samsung Galaxy", 1,), ("HP Elite", 2,),
             ("Lenovo Thinkpad", 2,), ("iPad", 3,), ("Kindle Fire", 3)]
    # fmt: on
    categories = db.create_dataframe(rows=rows1, column_names=["category_name", "category_id"])
    products = db.create_dataframe(rows=rows2, column_names=["product_name", "category_id"])

    ret = categories.join(
        products,
        on=["category_id"],
        self_columns={"category_id", "category_name"},
        other_columns={"category_id", "product_name"},
    )
    assert len(list(ret)) == 6
    assert sorted(next(iter(ret)).keys()) == sorted(
        ["category_id", "category_name", "product_name"]
    )
    for row in ret:
        if row["category_name"] == "Smart Phone":
            assert row["category_id"] == 1
        elif row["category_name"] == "Laptop":
            assert row["category_id"] == 2
        elif row["category_name"] == "DataFramet":
            assert row["category_id"] == 3


def test_dataframe_cross_join(db: gp.Database, zoo_1: gp.DataFrame, zoo_2: gp.DataFrame):
    ret = zoo_1.cross_join(
        zoo_2,
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    assert len(list(ret)) == 16
    ini1_dict = {"Tiger": 0, "Lion": 0, "Wolf": 0, "Fox": 0}
    ini2_dict = {"Tiger": 0, "Lion": 0, "Rhino": 0, "Panther": 0}
    zoo2_cpt = {"Tiger": ini2_dict, "Lion": ini2_dict, "Wolf": ini2_dict, "Fox": ini2_dict}
    zoo1_cpt = {"Tiger": ini1_dict, "Lion": ini1_dict, "Rhino": ini1_dict, "Panther": ini1_dict}
    for row in ret:
        zoo2_cpt[row["zoo1_animal"]][row["zoo2_animal"]] += 1
        zoo1_cpt[row["zoo2_animal"]][row["zoo1_animal"]] += 1
    for _, values in zoo1_cpt.items():
        assert values == {"Tiger": 4, "Lion": 4, "Wolf": 4, "Fox": 4}
    for _, values in zoo2_cpt.items():
        assert values == {"Tiger": 4, "Lion": 4, "Rhino": 4, "Panther": 4}


def test_dataframe_self_join(db: gp.Database, zoo_1: gp.DataFrame):
    ret: gp.DataFrame = zoo_1.join(
        zoo_1,
        on=["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    assert len(list(ret)) == 4


def test_dataframe_self_join_cond(db: gp.Database, zoo_1: gp.DataFrame):
    ret: gp.DataFrame = zoo_1.join(
        zoo_1,
        cond=lambda s, o: s["animal"] == o["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    assert len(list(ret)) == 4
    for row in ret:
        assert row["zoo1_animal"] == row["zoo2_animal"]


def test_dataframe_join_save(db: gp.Database, zoo_1: gp.DataFrame):
    t_join: gp.DataFrame = zoo_1.join(
        zoo_1,
        on=["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    t_join.save_as(
        "dataframe_join",
        column_names=["animal", "zoo1_id", "zoo2_id"],
        temp=True,
    )
    t_join_reload = gp.DataFrame.from_table("dataframe_join", db=db)
    assert sorted(next(iter(t_join_reload)).keys()) == sorted(
        [
            "animal",
            "zoo1_id",
            "zoo2_id",
        ]
    )


def test_dataframe_join_ine(db: gp.Database):
    rows1 = [(1,), (2,), (3,)]
    rows2 = [(2,), (3,), (4,)]
    t1 = db.create_dataframe(rows=rows1, column_names=["a"])
    t2 = db.create_dataframe(rows=rows2, column_names=["b"])
    ret = t1.join(t2, cond=lambda t1, t2: t1["a"] < t2["b"])
    assert len(list(ret)) == 6
    for row in ret:
        assert row["a"] < row["b"]


def test_dataframe_multiple_self_join(db: gp.Database, zoo_1: gp.DataFrame):
    t_join = zoo_1.join(
        zoo_1,
        on=["animal"],
        self_columns={"animal": "zoo1_animal", "id": "zoo1_id"},
        other_columns={"animal": "zoo2_animal", "id": "zoo2_id"},
    )
    ret = t_join.join(
        zoo_1,
        on=["animal"],
        self_columns={"animal", "zoo1_id", "zoo2_id"},
        other_columns={"animal", "id"},
    )
    assert len(list(ret)) == 4
    assert sorted(next(iter(ret)).keys()) == sorted(
        [
            "animal",
            "id",
            "zoo1_id",
            "zoo2_id",
        ]
    )


# This test case is to guarantee that the CTEs are generated in the reversed
# topological order (i.e. the DFS order) in the lineage graph.
#
# For that, we must ensure there are at least 4 nodes in the graph and the 2
# direct parents of the final node must not be adjacent.
def test_lineage_dfs_order(db: gp.Database):
    rows = [(i,) for i in range(10)]
    numbers = db.create_dataframe(rows=rows, column_names=["val"])
    mod = numbers.assign(mod=lambda t: t["val"] % 2)
    mod3 = mod.assign(mod3=lambda t: t["val"] % 3)
    results: gp.DataFrame = mod3.join(numbers, on=["val"], other_columns={})
    assert len(list(results)) == 10
