from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .expr import Expr
    from .table import Table


class OrderedTable:
    def __init__(
        self,
        table: "Table",
        ordering_list: List["Expr"],
        ascending_list: List[bool],
        nulls_first_list: List[bool],
        operator_list: List[str],
    ) -> None:
        self._table = table
        self._ordering_list = ordering_list
        self._ascending_list = ascending_list
        self._nulls_first_list = nulls_first_list
        self._operator_list = operator_list

    def order_by(
        self,
        order_col: "Expr",
        ascending: Optional[bool] = None,
        nulls_first: Optional[bool] = None,
        operator: Optional[str] = None,
    ) -> "OrderedTable":
        return OrderedTable(
            self._table,
            self._ordering_list + [order_col],
            self._ascending_list + [ascending],
            self._nulls_first_list + [nulls_first],
            self._operator_list + [operator],
        )

    # FIXME : Not sure about return type
    def head(self, num: int) -> "Table":
        """
        Returns a Table
        """
        from .table import Table

        return Table(
            f"""
                SELECT * FROM {self._table.name}
                {self.make_order_by_clause()}
                LIMIT {num}
            """,
            parents=[self._table],
        )

    @property
    def table(self) -> "Table":
        return self._table

    def make_order_by_clause(self) -> str:
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
