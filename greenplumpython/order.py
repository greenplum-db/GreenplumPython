"""
This module creates a Python object OrderedTable for order by table.
"""
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from greenplumpython.expr import Expr
    from greenplumpython.table import Table


class OrderedTable:
    """
    Represents an ordered :Table generated by order_by().
    """

    def __init__(
        self,
        table: "Table",
        ordering_list: List["Expr"],
        ascending_list: List[Optional[bool]],
        nulls_first_list: List[Optional[bool]],
        operator_list: List[Optional[str]],
    ) -> None:
        self._table = table
        self._ordering_list = ordering_list
        self._ascending_list = ascending_list
        self._nulls_first_list = nulls_first_list
        self._operator_list = operator_list

    def order_by(
        self,
        sort_expr: "Expr",
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
        operator: Optional[str] = None,
    ) -> "OrderedTable":
        """
        Refine the order by adding another ordering definition to break the tie.

        Args:
            order_col: Expr : Column which used to order by the table
            ascending: Optional[Bool]: Define ascending of order, True = ASC / False = DESC
            nulls_first: Optional[bool]: Define if nulls will be ordered first or last, True = First / False = Last
            operator: Optional[str]: Define order by using operator. **Can't combine with ascending.**

        Returns:
            OrderedTable : Table ordered by the given arguments

        Example:
            .. code-block::  Python

                t.order_by(t["id"]).order_by(t["num"], ascending=False)
        """
        if ascending is not None and operator is not None:
            raise Exception(
                "Could not use 'ascending' and 'operator' at the same time to order by one column"
            )
        return OrderedTable(
            self._table,
            self._ordering_list + [sort_expr],
            self._ascending_list + [ascending],
            self._nulls_first_list + [nulls_first],
            self._operator_list + [operator],
        )

    # FIXME : Not sure about return type
    def head(self, num: int) -> "Table":
        """
        Returns a :Table that contains the first :num rows in order.

        Args:
            num: int: number of first rows

        Returns:
            Table: Ordered by table

        Example:
            .. code-block::  Python

                t.order_by(t["id"]).head(5)
        """
        from greenplumpython.table import Table

        return Table(
            f"""
                SELECT * FROM {self._table.name}
                {self._make_order_by_clause()}
                LIMIT {num}
            """,
            parents=[self._table],
        )

    @property
    def table(self) -> "Table":
        """
        Returns table associated for GROUP BY

        Returns:
            Table
        """
        return self._table

    def _make_order_by_clause(self) -> str:
        # FIXME : If user define ascending and operator, will get syntax error
        order_by_str = ",".join(
            [
                (
                    f"""
            {self._ordering_list[i]} {"" if self._ascending_list[i] is None else "ASC" if self._ascending_list[i] else "DESC"}
            {"" if self._operator_list[i] is None else ("USING " + self._operator_list[i])}
            {"" if self._nulls_first_list[i] is None else "NULLs FIRST" if self._nulls_first_list[i] else "NULLs LAST"}
            """
                )
                for i in range(len(self._ordering_list))
            ]
        )
        return "ORDER BY " + order_by_str
