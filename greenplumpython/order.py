"""
This module creates a Python object OrderedTable for order by table.
"""
import sys
from typing import TYPE_CHECKING, List, Optional

from greenplumpython.col import Column

if TYPE_CHECKING:
    from greenplumpython.table import Table


class OrderedTable:
    """
    Represents an ordered :class:`~table.Table` generated by
    :func:`Table.order_by <table.Table.order_by>` or :func:`OrderedTable.order_by`.
    """

    def __init__(
        self,
        table: "Table",
        column_name_list: List[str],
        ascending_list: List[Optional[bool]],
        nulls_first_list: List[Optional[bool]],
        operator_list: List[Optional[str]],
    ) -> None:
        self._table = table
        self._column_name_list = column_name_list
        self._ascending_list = ascending_list
        self._nulls_first_list = nulls_first_list
        self._operator_list = operator_list

    def order_by(
        self,
        column_name: str,
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
        operator: Optional[str] = None,
    ) -> "OrderedTable":
        """
        Refine the order by adding another ordering definition to break the tie.

        Args:
            column_name: name of column to order the table by
            ascending: Optional[Bool]: Define ascending of order, True = ASC / False = DESC
            nulls_first: Optional[bool]: Define if nulls will be ordered first or last, True = First / False = Last
            operator: Optional[str]: Define order by using operator. **Can't combine with ascending.**

        Returns:
            OrderedTable : Table ordered by the given arguments

        Example:
            .. code-block::  Python

                t.order_by("id").order_by("num", ascending=False)
        """
        if ascending is not None and operator is not None:
            raise Exception(
                "Could not use 'ascending' and 'operator' at the same time to order by one column"
            )
        return OrderedTable(
            self._table,
            self._column_name_list + [column_name],
            self._ascending_list + [ascending],
            self._nulls_first_list + [nulls_first],
            self._operator_list + [operator],
        )

    # FIXME : Not sure about return type
    def __getitem__(self, rows: slice) -> "Table":
        """
        Returns a :class:`~table.Table` that contains the slice of table in order.

        Args:
            rows: slice: number of first rows

        Returns:
            Table: Ordered by table

        Example:
            .. code-block::  Python

                t.order_by("id").head(5)
        """
        from greenplumpython.table import Table

        if rows.step is not None:
            raise NotImplementedError()
        offset_clause = "" if rows.start is None else f"OFFSET {rows.start}"
        limit = (
            sys.maxsize
            if rows.stop is None
            else rows.stop
            if rows.start is None
            else rows.stop - rows.start
        )
        return Table(
            f"SELECT * FROM {self._table.name} {self._clause()} LIMIT {limit} {offset_clause}",
            parents=[self._table],
        )

    @property
    def table(self) -> "Table":
        """
        Returns :class:`~table.Table` associated for ORDER BY

        Returns:
            Table
        """
        return self._table

    def _clause(self) -> str:
        """:meta private:"""
        # FIXME : If user define ascending and operator, will get syntax error
        order_by_str = ",".join(
            [
                " ".join(
                    [
                        Column(self._column_name_list[i], self.table).serialize(),
                        ""
                        if self._ascending_list[i] is None
                        else "ASC"
                        if self._ascending_list[i]
                        else "DESC",
                        ""
                        if self._operator_list[i] is None
                        else ("USING " + self._operator_list[i]),
                        ""
                        if self._nulls_first_list[i] is None
                        else "NULLS FIRST"
                        if self._nulls_first_list[i]
                        else "NULLS LAST",
                    ]
                )
                for i in range(len(self._column_name_list))
            ]
        )
        return "ORDER BY " + order_by_str
