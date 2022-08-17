"""
This module creates a Python object TableRowGroup for group by table.
"""
from typing import TYPE_CHECKING, Callable, Iterable, List, MutableSet

if TYPE_CHECKING:
    from .expr import Expr
    from .func import FunctionExpr
    from .table import Table


class TableRowGroup:
    """
    Represents a group of rows in a :Table generated by group_by().
    """

    def __init__(self, table: "Table", grouping_sets: List[List["Expr"]]) -> None:
        self._table = table
        self._grouping_sets = grouping_sets

    def apply(self, func: Callable[["Table"], "FunctionExpr"]) -> "FunctionExpr":
        """
        Apply a function to the row group.

        Args:
            func: Callable[["Table"], "FunctionExpr"]: a lambda function of a FunctionExpr

        Returns:
            FunctionExpr: a callable

        Example:
            .. code-block::  python

                numbers.group_by("is_even").apply(lambda row: count(row["*"]))
        """
        return func(self._table)(group_by=self)

    def __or__(self, other: "TableRowGroup") -> "TableRowGroup":
        """
        Returns the union of the two row groups.

        This does not merge the two groups into one. Instead, it means applying
        any operation to the union is equivalent to applying the operation to
        each group and union the result sets.
        """
        assert self._table == other._table
        return TableRowGroup(self._table, self._grouping_sets + other._grouping_sets)

    # FIXME: Make this function package-private
    def get_targets(self) -> Iterable["Expr"]:
        item_set: MutableSet[Expr] = set()
        for grouping_set in self._grouping_sets:
            for group_by_item in grouping_set:
                item_set.add(group_by_item)
        return item_set

    @property
    def table(self) -> "Table":
        """
        Returns table associated for GROUP BY

        Returns:
            Table
        """
        return self._table

    # FIXME: Make this function package-private
    def make_group_by_clause(self) -> str:
        grouping_sets_str = [
            f"({','.join([str(item) for item in grouping_set])})"
            for grouping_set in self._grouping_sets
        ]
        return "GROUP BY GROUPING SETS " + f"({','.join(grouping_sets_str)})"
