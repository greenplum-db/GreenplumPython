"""
`DataFrame` is the core data structure in GreenplumPython. Conceptually, a `DataFrame`
is a two-dimensional unordered structure containing data. This aligns with `the
definition of "DataFrame" on Wikipedia
<https://en.wikipedia.org/wiki/DataFrame_(information)>`_.

In the data science world, a `DataFrame` is similar to a `DataFrame` in `pandas
<https://pandas.pydata.org/>`_, except that

- | Data in a `DataFrame` is lazily evaluated rather than eagerly. That is, they are computed only when
  | they are observed. This can improve efficiency in many cases.
- | Data in a `DataFrame` is located and manipulated on a remote database system rather than locally. As
  | a consequence,

    - | Retrieving them from the database system can be expensive. Therefore, once the data 
      | of a :class:`DataFrame` is fetched from the database system, it will be cached locally for later use.
    - | They might be modified concurrently by other users of the database system. You might 
      | need to use :meth:`~dataframe.DataFrame.refresh()` to sync the updates if the data becomes stale.

In the database world, a `DataFrame` is similar to a **materialized view** in a
database system in that

- They both result from a possibly complex query.
- They both hold data, as oppose to views.
- | The data can become stale due to concurrent modification. And the :meth:`~dataframe.DataFrame.refresh()` method
  | is similar to the :code:`REFRESH MATERIALIZED VIEW` `command in PostgreSQL <https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html>`_ for syncing updates.
"""
import collections
import json
from collections import abc
from functools import partialmethod, singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

if TYPE_CHECKING:
    from greenplumpython.func import FunctionExpr

from uuid import uuid4

from psycopg2.extras import RealDictRow

from greenplumpython.col import Column, Expr
from greenplumpython.db import Database
from greenplumpython.expr import serialize
from greenplumpython.group import DataFrameGroupingSets
from greenplumpython.order import DataFrameOrdering
from greenplumpython.row import Row


class DataFrame:
    """
    Representation of DataFrame object.
    """

    def __init__(
        self,
        query: str,
        parents: List["DataFrame"] = [],
        name: Optional[str] = None,
        db: Optional[Database] = None,
        columns: Optional[Iterable[Column]] = None,
    ) -> None:
        self._query = query
        self._parents = parents
        self._name = "cte_" + uuid4().hex if name is None else name
        self._columns = columns
        self._contents: Optional[Iterable[RealDictRow]] = None
        if any(parents):
            self._db = next(iter(parents))._db
        else:
            self._db = db

    @singledispatchmethod
    def _getitem(self, _) -> "DataFrame":
        raise NotImplementedError()

    @_getitem.register(abc.Callable)  # type: ignore reportMissingTypeArgument
    def _(self, predicate: Callable[["DataFrame"], Expr]):
        return self.where(predicate)

    @_getitem.register(list)
    def _(self, column_names: List[str]) -> "DataFrame":
        targets_str = [serialize(self[col]) for col in column_names]
        return DataFrame(
            f"""
                SELECT {','.join(targets_str)} 
                FROM {self._name}
            """,
            parents=[self],
        )

    @_getitem.register(str)
    def _(self, column_name: str) -> "DataFrame":
        return Column(column_name, self)

    @_getitem.register(slice)
    def _(self, rows: slice) -> "DataFrame":
        if rows.step is not None:
            raise NotImplementedError()
        offset_clause = "" if rows.start is None else f"OFFSET {rows.start}"
        limit_clause = (
            ""
            if rows.stop is None
            else f"LIMIT {rows.stop if rows.start is None else rows.stop - rows.start}"
        )
        return DataFrame(
            f"SELECT * FROM {self.name} {limit_clause} {offset_clause}",
            parents=[self],
        )

    @overload
    def __getitem__(self, _) -> "DataFrame":
        ...

    @overload
    def __getitem__(self, column_names: List[str]) -> "DataFrame":
        ...

    @overload
    def __getitem__(self, predicate: Callable[["DataFrame"], Expr]) -> "DataFrame":
        ...

    @overload
    def __getitem__(self, column_name: str) -> Expr:
        ...

    @overload
    def __getitem__(self, rows: slice) -> "DataFrame":
        ...

    def __getitem__(self, _):
        """
        Returns
            - a :class:`~expr.Column` of the current DataFrame if key is string

            .. code-block::  python

               id_col = tab["id"]

            - a new :class:`DataFrame` from the current DataFrame per the type of key:

                - if key is a list, then SELECT a subset of columns, a.k.a. targets;

                .. code-block::  python

                   id_dataframe = tab[["id"]]

                - if key is an :class:`~expr.Expr`, then SELECT a subset of rows per the value of the Expr;

                .. code-block::  python

                   id_cond_dataframe = tab[lambda t: t["id"] == 0]

                - if key is a slice, then SELECT a portion of consecutive rows

                .. code-block::  python

                   slice_dataframe = tab[2:5]

        """
        return self._getitem(_)

    def __repr__(self):
        """
        :meta private:

        Return a string representation for a dataframe
        """
        repr_string: str = ""
        if len(list(self)) != 0:
            # Iterate over the given dataframe to calculate the column width for its ASCII representation.
            width = [0] * len(next(iter(self)).column_names())
            for row in self:
                for col_idx, col in enumerate(row):
                    width[col_idx] = max(width[col_idx], len(col), len(str(row[col])))

            # DataFrame header.
            repr_string += (
                "".join(
                    [
                        "| {:{}} |".format(col, width[idx])
                        for idx, col in enumerate(next(iter(self)))
                    ]
                )
                + "\n"
            )
            # Dividing line below dataframe header.
            repr_string += ("=" * (sum(width) + 4 * len(width))) + "\n"
            # DataFrame contents.
            for row in self:
                content = [row[c] for c in row]
                for idx, c in enumerate(content):
                    if isinstance(c, list):
                        repr_string += ("| {:{}} |").format("{}".format(c), width[idx])  # type: ignore
                    else:
                        repr_string += ("| {:{}} |").format(c if c is not None else "", width[idx])
                repr_string += "\n"
        return repr_string

    def _repr_html_(self):
        """:meta private:"""
        repr_html_str = ""
        if len(list(self)) != 0:
            repr_html_str = "<table>\n"
            repr_html_str += "\t<tr>\n"
            repr_html_str += ("\t\t<th>{:}</th>\n" * len(list(next(iter(self))))).format(
                *((next(iter(self))))
            )
            repr_html_str += "\t</tr>\n"
            for row in self:
                content = [row[c] for c in row]
                repr_html_str += "\t<tr>\n"
                for c in content:
                    if isinstance(c, list):
                        repr_html_str += ("\t\t<td>{:}</td>\n").format("{}".format(c))  # type: ignore
                    else:
                        repr_html_str += ("\t\t<td>{:}</td>\n").format(c if c is not None else "")
                repr_html_str += "\t</tr>\n"
            repr_html_str += "</table>"
        return repr_html_str

    # FIXME: Add test
    def where(self, predicate: Callable[["DataFrame"], "Expr"]) -> "DataFrame":
        """
        Returns the :class:`DataFrame` filtered by Expression.

        Args:
            predicate: :class:`~expr.Expr` : where condition statement

        Returns:
            DataFrame : DataFrame filtered according **expr** passed in argument
        """
        v = predicate(self)
        assert v.dataframe == self, "Predicate must based on current dataframe"
        parents = [self]
        if v.other_dataframe is not None and self.name != v.other_dataframe.name:
            parents.append(v.other_dataframe)
        return DataFrame(f"SELECT * FROM {self._name} WHERE {v.serialize()}", parents=parents)

    def apply(
        self,
        func: Callable[["DataFrame"], "FunctionExpr"],
        expand: bool = False,
        as_name: Optional[str] = None,
    ) -> "DataFrame":
        """
        Apply a function to the :class:`DataFrame`
        Args:
            func: Callable[[:class:`DataFrame`], :class:`~func.FunctionExpr`]: a lambda function of a FunctionExpr
            expand: bool: expand field of composite returning type
            as_name: str: rename returning column
        Returns:
            DataFrame: resulted DataFrame
        Example:
            .. code-block::  python

                rows = [(i,) for i in range(-10, 0)]
                series = gp.values(rows, db=db, column_names=["id"])
                abs = gp.function("abs", db=db)
                result = series.apply(lambda t: abs(t["id"]))

            If we want to give constant as attribute, it is also easy to use. Suppose *label* function takes a str and a int:

            .. code-block::  python

                result = series.apply(lambda t: label("label", t["id"]))
        """
        # We need to support calling functions with constant args or even no
        # arg. For example: SELECT count(*) FROM t; In that case, the
        # arguments do not contain information on any dataframe or any database.
        # As a result, the generated SQL cannot be executed.
        #
        # To fix this, we need to pass the dataframe to the resulting FunctionExpr
        # explicitly.
        return func(self).bind(dataframe=self).apply(expand=expand, as_name=as_name)

    def assign(self, **new_columns: Callable[["DataFrame"], Any]) -> "DataFrame":
        """
        Assigns new columns to the current :class:`DataFrame`. Existing columns
        cannot be reassigned.

        Args:
            new_columns: a :class:`dict` whose keys are column names and values
                are :class:`Callable`s returning column data when applied to the
                current :class:`DataFrame`.

        Returns:
            DataFrame: New dataframe including the new assigned columns

        Example:
            .. code-block::  python

                rows = [(i,) for i in range(-10, 0)]
                series = gp.to_dataframe(rows, db=db, column_names=["id"])
                abs = gp.function("abs")
                results = series.assign(abs=lambda nums: abs(nums["id"]))

        """

        if len(new_columns) == 0:
            return self
        targets: List[str] = []
        other_parents: Dict[str, DataFrame] = {}
        if len(new_columns):
            for k, f in new_columns.items():
                v: Any = f(self)
                if isinstance(v, Expr):
                    assert (
                        v.dataframe is None or v.dataframe == self
                    ), "Newly included columns must be based on the current dataframe"
                    if v.other_dataframe is not None and v.other_dataframe.name != self.name:
                        if v.other_dataframe.name not in other_parents:
                            other_parents[v.other_dataframe.name] = v.other_dataframe
                targets.append(f"{serialize(v)} AS {k}")
            return DataFrame(
                f"SELECT *, {','.join(targets)} FROM {self.name}",
                parents=[self] + list(other_parents.values()),
            )

    def order_by(
        self,
        column_name: str,
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
        operator: Optional[str] = None,
    ) -> DataFrameOrdering:
        """
        Returns :class:`DataFrame` order by the given arguments.

        Args:
            column_name: name of column to order the dataframe by
            ascending: Optional[Bool]: Define ascending of order, True = ASC / False = DESC
            nulls_first: Optional[bool]: Define if nulls will be ordered first or last, True = First / False = Last
            operator: Optional[str]: Define order by using operator. **Can't combine with ascending.**

        Returns:
            DataFrameOrdering : DataFrame ordered by the given arguments

        Example:
            .. code-block::  Python

                t.order_by("id")
        """
        # State transition diagram:
        # DataFrame --order_by()-> DataFrameOrdering --head()-> DataFrame
        if ascending is not None and operator is not None:
            raise Exception(
                "Could not use 'ascending' and 'operator' together to order by one column"
            )
        return DataFrameOrdering(
            self,
            [column_name],
            [ascending],
            [nulls_first],
            [operator],
        )

    def join(
        self,
        other: "DataFrame",
        how: str = "",
        cond: Optional[Callable[["DataFrame", "DataFrame"], Expr]] = None,
        using: Optional[Iterable[str]] = None,
        self_columns: Union[Dict[str, Optional[str]], Set[str]] = {"*"},
        other_columns: Union[Dict[str, Optional[str]], Set[str]] = {"*"},
    ) -> "DataFrame":
        """
        Joins the current :class:`DataFrame` with another :class:`DataFrame`.

        Args:
            other: :class:`DataFrame` to join with
            how: How the two dataframes are joined. The value can be one of
                - `"INNER"`: inner join,
                - `"LEFT"`: left outer join,
                - `"LEFT"`: right outer join,
                - `"FULL"`: full outer join, or
                - `"CROSS"`: cross join, i.e. the Cartesian product
                The default value `""` is equivalent to "INNER".

            cond: :class:`Callable` lambda function as the join condition
            using: a list of column names that exist in both dataframes to join on.
                `cond` and `using` cannot be used together.
            self_columns: A :class:`dict` whose keys are the column names of
                the current dataframe to be included in the resulting
                dataframe. The value, if not `None`, is used for renaming
                the corresponding key to avoid name conflicts. Asterisk `"*"`
                can be used as a key to indicate all columns.
            other_columns: Same as `self_columns`, but for the `other`
                dataframe.

        Note:
            When using `"*"` as key in `self_columns` or `other_columns`,
            please ensure that there will not be more than one column with the
            same name by applying proper renaming. Otherwise there will be an
            error.
        """
        # FIXME : Raise Error if target columns don't exist
        assert how.upper() in [
            "",
            "INNER",
            "LEFT",
            "RIGHT",
            "FULL",
            "CROSS",
        ], "Unsupported join type"
        assert cond is None or using is None, 'Cannot specify "cond" and "using" together'

        def bind(t: DataFrame, columns: Union[Dict[str, Optional[str]], Set[str]]) -> List[str]:
            target_list: List[str] = []
            for k in columns:
                col: Column = t[k]
                v = columns[k] if isinstance(columns, dict) else None
                target_list.append(col.serialize() + (f" AS {v}" if v is not None else ""))
            return target_list

        other_name = other.name if self.name != other.name else "cte_" + uuid4().hex
        other_clause = other.name if self.name != other.name else self.name + " AS " + other_name
        target_list = bind(self, self_columns) + bind(
            DataFrame(query="", name=other_name), other_columns
        )
        on_clause = f"ON {cond(self, other).serialize()}" if cond is not None else ""
        using_clause = f"USING ({','.join(using)})" if using is not None else ""
        return DataFrame(
            f"""
                SELECT {",".join(target_list)}
                FROM {self.name} {how} JOIN {other_clause} {on_clause} {using_clause}
            """,
            parents=[self, other],
        )

    inner_join = partialmethod(join, how="INNER")
    """
    Inner joins the current :class:`DataFrame` with another :class:`DataFrame`.

    Equivalent to calling :meth:`DataFrame.join` with `how="INNER"`.
    """

    left_join = partialmethod(join, how="LEFT")
    """
    Left-outer joins the current :class:`DataFrame` with another :class:`DataFrame`.

    Equivalent to calling :meth:`DataFrame.join` with `how="LEFT"`.
    """

    right_join = partialmethod(join, how="RIGHT")
    """
    Right-outer joins the current :class:`DataFrame` with another :class:`DataFrame`.

    Equivalent to calling :meth:`DataFrame.join` with `how="RIGHT"`.
    """

    full_join = partialmethod(join, how="FULL")
    """
    Full-outer joins the current :class:`DataFrame` with another :class:`DataFrame`.

    Equivalent to calling :meth:`DataFrame.join` with argutment `how="FULL"`.
    """

    cross_join = partialmethod(join, how="CROSS", cond=None, using=None)
    """
    Cross joins the current :class:`DataFrame` with another :class:`DataFrame`,
    i.e. the Cartesian product.

    Equivalent to calling :meth:`DataFrame.join` with `how="CROSS"`.
    """

    @property
    def name(self) -> str:
        """
        Returns name of :class:`DataFrame`

        Returns:
            str: DataFrame name
        """
        return self._name

    @property
    def db(self) -> Optional[Database]:
        """
        Returns :class:`~Database` associated with :class:`DataFrame`

        Returns:
            Optional[Database]: database associated with dataframe
        """
        return self._db

    # @property
    # def columns(self) -> Optional[Iterable[Column]]:
    #     """
    #     Returns its :class:`~expr.Column` name of :class:`DataFrame`, has
    #     results only for selected dataframe and joined dataframe with targets.

    #     Returns:
    #         Optional[Iterable[str]]: None or List of its columns names of dataframe
    #     """
    #     return self._columns

    # This is used to filter out dataframes that are derived from other dataframes.
    #
    # Actually we cannot determine if a dataframe is recorded in the system catalogs
    # without querying the db.
    def _in_catalog(self) -> bool:
        """:meta private:"""
        return self._query.startswith("TABLE")

    def _list_lineage(self) -> List["DataFrame"]:
        """:meta private:"""
        lineage: List["DataFrame"] = [self]
        dataframes_visited: Set[str] = set()
        current = 0
        while current < len(lineage):
            if (
                lineage[current].name not in dataframes_visited
                and not lineage[current]._in_catalog()
            ):
                self._depth_first_search(lineage[current], dataframes_visited, lineage)
            current += 1
        return lineage

    def _depth_first_search(self, t: "DataFrame", visited: Set[str], lineage: List["DataFrame"]):
        """:meta private:"""
        visited.add(t.name)
        for i in t._parents:
            if i.name not in visited and not i._in_catalog():
                self._depth_first_search(i, visited, lineage)
        lineage.append(t)

    def _build_full_query(self) -> str:
        """:meta private:"""
        lineage = self._list_lineage()
        cte_list: List[str] = []
        for dataframe in lineage:
            if dataframe._name != self._name:
                cte_list.append(f"{dataframe._name} AS ({dataframe._query})")
        if len(cte_list) == 0:
            return self._query
        return "WITH " + ",".join(cte_list) + self._query

    def __iter__(self) -> "DataFrame.DictIterator":
        """:meta private:"""
        if self._contents is not None:
            return DataFrame.DictIterator(self._contents)
        assert self._db is not None
        self._contents = self._fetch()
        assert self._contents is not None
        return DataFrame.DictIterator(self._contents)

    class DictIterator:
        """:meta private:"""

        def __init__(self, contents: Iterable[RealDictRow]) -> None:
            """:meta private:"""
            self._proxy_iter: Iterator[RealDictRow] = iter(contents)

        def __iter__(self):
            return self

        def __next__(self):
            """:meta private:"""

            def tuple_to_dict(json_pairs: List[tuple[str, Any]]):
                json_dict = dict(json_pairs)
                if len(json_dict) != len(json_pairs):
                    raise Exception("Duplicate column name(s) found: {}".format(json_dict.keys()))
                return json_dict

            current_row = next(self._proxy_iter)
            for name in current_row.keys():
                # According our current _fetch(), name == "to_json" will be always True
                json_dict: Dict[str, Union[Any, List[Any]]] = json.loads(
                    current_row[name], object_pairs_hook=tuple_to_dict
                )
                assert isinstance(json_dict, dict), "Failed to fetch the entire row of dataframe."
                return Row(json_dict)

    def refresh(self) -> "DataFrame":
        """
        Refresh self._contents

        Returns:
            self
        """

        assert self._db is not None
        self._contents = self._fetch()
        assert self._contents is not None
        return self

    def _fetch(self, is_all: bool = True) -> Iterable[Tuple[Any]]:
        """
        Fetch rows of this :class:`DataFrame`.
        - if is_all is True, fetch all rows at once
        - otherwise, open a CURSOR and FETCH one row at a time

        Args:
            is_all: bool: Define if fetch all rows at once

        Returns:
            Iterable[Tuple[Any]]: results of query received from database
        """
        if not is_all:
            raise NotImplementedError()
        assert self._db is not None
        output_name = "cte_" + uuid4().hex
        json_df = DataFrame(
            f"SELECT to_json({output_name})::TEXT FROM {self.name} AS {output_name}",
            parents=[self],
        )
        result = self._db.execute(json_df._build_full_query())
        return result if result is not None else []

    def save_as(
        self, dataframe_name: str, column_names: List[str] = [], temp: bool = False
    ) -> "DataFrame":
        """
        Save the dataframe to database as a real Greenplum DataFrame

        Args:
            dataframe_name : str
            temp : bool : if dataframe is temporary
            column_names : List : list of column names

        Returns:
            DataFrame : dataframe saved in database
        """
        assert self._db is not None
        # TODO: Remove assertion below after implementing schema inference.
        assert len(column_names) > 0, "Column names of new dataframe are unknown."
        self._db.execute(
            f"""
            CREATE {'TEMP' if temp else ''} TABLE {dataframe_name} ({','.join(column_names)}) 
            AS {self._build_full_query()}
            """,
            has_results=False,
        )
        return DataFrame.from_table(dataframe_name, self._db)

    # TODO: Uncomment or remove this.
    #
    # def create_index(
    #     self,
    #     columns: Iterable[Union["Column", str]],
    #     method: str = "btree",
    #     name: Optional[str] = None,
    # ) -> None:
    #     if not self._in_catalog():
    #         raise Exception("Cannot create index on dataframes not in the system catalog.")
    #     index_name: str = name if name is not None else "idx_" + uuid4().hex
    #     indexed_cols = ",".join([str(col) for col in columns])
    #     assert self._db is not None
    #     self._db.execute(
    #         f"CREATE INDEX {index_name} ON {self.name} USING {method} ({indexed_cols})",
    #         has_results=False,
    #     )

    # FIXME: Should we choose JSON as the default format?
    def explain(self, format: str = "TEXT") -> Iterable[Tuple[str]]:
        """
        Explained the dataframe's query

        Args:
            format: str: format of explain

        Returns:
            Iterable[Tuple[str]]: EXPLAIN query answer
        """
        assert self._db is not None
        results = self._db.execute(f"EXPLAIN (FORMAT {format}) {self._build_full_query()}")
        assert results is not None
        return results

    def group_by(self, *column_names: str) -> DataFrameGroupingSets:
        """
        Group the current :class:`~dataframe.DataFrame` by columns specified by
        `column_names`.

        Args:
            column_names: one or more column names of the dataframe

        Returns:
            DataFrameGroupingSets: a list of grouping sets. Each group is identified
            by a different set of values of the columns in the arguments.
        """
        #  State transition diagram:
        #  DataFrame --group_by()-> DataFrameRowGroup --aggregate()-> FunctionExpr
        #    ^                                                    |
        #    |------------------------- to_dataframe() ---------------|
        return DataFrameGroupingSets(self, [column_names])

    def distinct_on(self, *column_names: str) -> "DataFrame":
        """
        Deduplicate the current :class:`DataFrame` with respect to the given columns.

        Args:
            column_names: name of column of the current :class:`DataFrame`.

        Returns:
            :class:`DataFrame`: DataFrame containing only the distinct values of the
                            given columns.
        """
        cols = [Column(name, self).serialize() for name in column_names]
        return DataFrame(
            f"SELECT DISTINCT ON ({','.join(cols)}) * FROM {self.name}", parents=[self]
        )

    # dataframe_name can be table/view name
    @classmethod
    def from_table(cls, table_name: str, db: Database) -> "DataFrame":
        """
        Returns a :class:`DataFrame` using dataframe name and associated :class:`~Database`

        Args:
            name: str: DataFrame name
            db: :class:`~Database`: database which contains the dataframe
        """
        return DataFrame(f'TABLE "{table_name}"', name=table_name, db=db)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Union[Tuple[Any], Dict[str, Any]]],
        db: Database,
        column_names: Optional[List[str]] = None,
    ) -> "DataFrame":
        """
        Returns a :class:`DataFrame` using list of values given

        Args:
            rows: Iterable[Tuple[Any]]: List of values
            db: :class:`~Database`: database which will be associated with dataframe
            column_names: Iterable[str]: List of given column names

        Returns:
            DataFrame: dataframe generated with given values

        .. code-block::  python

           rows = [(1,), (2,), (3,)]
            t = gp.create_dataframe(rows, db=db)

        """
        row_tuples = [row.values() if isinstance(row, dict) else row for row in rows]
        if column_names is None:
            first_row = next(iter(rows))
            if isinstance(first_row, dict):
                column_names = first_row.keys()
        assert column_names is not None, "Column names of the DataFrame is unknown."
        rows_string = ",".join(
            [f"({','.join(serialize(datum) for datum in row)})" for row in row_tuples]
        )
        column_names = [f'"{name}"' for name in column_names]
        columns_string = f"({','.join(column_names)})"
        table_name = "cte_" + uuid4().hex
        return DataFrame(
            f"SELECT * FROM (VALUES {rows_string}) AS {table_name} {columns_string}", db=db
        )

    @classmethod
    def from_columns(cls, columns: Dict[str, List[Any]], db: Database) -> "DataFrame":
        """
        Returns a :class:`DataFrame` using list of columns values given

        Args:
            columns: Dict[str, List[Any]]: List of column values
            db: :class:`~Database`: database which will be associated with dataframe

        Returns:
            DataFrame: dataframe generated with given values

        .. code-block::  python

           columns = {"a": [1, 2, 3], "b": [1, 2, 3]}
            t = gp.create_dataframe(columns, db=db)

        """
        columns_string = ",".join([f'unnest({serialize(v)}) AS "{k}"' for k, v in columns.items()])
        return DataFrame(f"SELECT {columns_string}", db=db)