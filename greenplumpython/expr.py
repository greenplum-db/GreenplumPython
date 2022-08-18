"""
This module creates a Python object Expr.
"""
import copy
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Optional, Union, overload

from greenplumpython.db import Database

if TYPE_CHECKING:
    from greenplumpython.table import Table
    from greenplumpython.type import Type


class Expr:
    """
    Representation of Expr. It can be

        - a Column
        - an Operator
        - a Function
    """

    def __init__(
        self,
        as_name: Optional[str] = None,
        table: Optional["Table"] = None,
        db: Optional[Database] = None,
    ) -> None:
        self._as_name = as_name
        self._table = table
        self._db = table.db if table is not None else db
        assert self._db is not None

    def __and__(self, other: "Expr") -> "Expr":
        """
        Operator **&**

        Returns a Binary Expression AND between self and another Expr

        Example:
            .. code-block::  Python

                t["type"] == "type_1" & t["val"] > 0

        """
        return BinaryExpr("AND", self, other)

    def __or__(self, other: "Expr") -> "Expr":
        """
        Operator **|**

        Returns a Binary Expression OR between self and another Expr

        Example:
            .. code-block::  Python

                t["type"] == "type_1" | t["type"] == "type_2"
        """
        return BinaryExpr("OR", self, other)

    def __eq__(self, other: "Expr") -> "Expr":
        """
        Operator **==**

        Returns a Binary Expression EQUAL between self and another Expr

        Example:
            .. code-block::  Python

                t["type"] == "type_1"
        """
        if isinstance(other, type(None)):
            return BinaryExpr("is", self, other)
        return BinaryExpr("=", self, other)

    def __lt__(self, other: "Expr") -> "Expr":
        """
        Operator **<**

        Returns a Binary Expression LESS THAN between self and another Expr

        Example:
            .. code-block::  Python

                t["val"] < 0
        """
        return BinaryExpr("<", self, other)

    def __le__(self, other: "Expr") -> "Expr":
        """
        Operator **<=**

        Returns a Binary Expression LESS EQUAL between self and another Expr

        Example:
            .. code-block::  Python

                t["val"] <= 0
        """
        return BinaryExpr("<=", self, other)

    def __gt__(self, other: "Expr") -> "Expr":
        """
        Operator **>**

        Returns a Binary Expression GREATER THAN between self and another Expr

        Example:
            .. code-block::  Python

                t["val"] > 0
        """
        return BinaryExpr(">", self, other)

    def __ge__(self, other: "Expr") -> "Expr":
        """
        Operator **>=**

        Returns a Binary Expression GREATER EQUAL between self and another Expr

        Example:
            .. code-block::  Python

                t["val"] >= 0
        """
        return BinaryExpr(">=", self, other)

    def __ne__(self, other: "Expr") -> "Expr":
        """
        Operator **!=**

        Returns a Binary Expression NOT EQUAL between self and another Expr

        Example:
            .. code-block::  Python

                t["val"] != 0
        """
        return BinaryExpr("!=", self, other)

    def __mod__(self, other: Union[int, "Expr"]) -> "Expr":
        """
        Operator **%**

        Returns a Binary Expression Modulo between an Expr and an integer or an Expr

        Example:
            .. code-block::  Python

                t["val"] % 2
        """
        return BinaryExpr("%", self, other)

    def __pos__(self) -> "Expr":
        """
        Operator **+**

        Returns a Unary Expression POSITIVE of self

        Example:
            .. code-block::  Python

                +t["val"]
        """
        return UnaryExpr("+", self)

    def __neg__(self) -> "Expr":
        """
        Operator **-**

        Returns a Unary Expression NEGATIVE of self

        Example:
            .. code-block::  Python

                -t["val"]
        """
        return UnaryExpr("-", self)

    def __abs__(self) -> "Expr":
        """
        Operator **abs()**

        Returns a Unary Expression ABSOLUTE of self

        Example:
            .. code-block::  Python

                abs(t["val"])
        """
        return UnaryExpr("ABS", self)

    def __invert__(self) -> "Expr":
        """
        Operator **~**

        Returns a Unary Expression NOT of self

        Example:
            .. code-block::  Python

                not(t["val"])
        """
        return UnaryExpr("NOT", self)

    def like(self, pattern: str) -> "Expr":
        """
        Returns BinaryExpr in order to apply LIKE statement on self with pattern

        Args:
            pattern: str: regex pattern

        Returns:
            Expr

        Example:
            Select rows where id begins with "a"

            .. code-block::  Python

                t[t["id"].like(r"a%")]

        """
        return BinaryExpr("LIKE", self, pattern)

    def __str__(self) -> str:
        """
        Returns string statement of Expr, ie : name + AS (optional)
        """
        return self._serialize() + (" AS " + self._as_name if self._as_name is not None else "")

    def rename(self, new_name: str) -> "Expr":
        """
        Return copy of Expr with a new name

        Args:
            new_name: str: Expr's new name

        Returns:
            Expr: a new Object Expr with given new name
        """
        new_expr = copy.copy(self)  # Shallow copy
        new_expr._as_name = new_name
        return new_expr

    def _serialize(self) -> str:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        Returns name of Expr

        Returns:
            str: Expr name
        """
        raise NotImplementedError()

    @property
    def as_name(self) -> str:
        """
        Returns alias name of Expr

        Returns:
            str: Expr alias name
        """
        return self._as_name

    @property
    def db(self) -> Optional[Database]:
        """
        Returns Expr associated database

        Returns:
            Optional[Database]: database associated with Expr
        """
        return self._db

    @property
    def table(self) -> Optional["Table"]:
        """
        Returns Expr associated table

        Returns:
        Optional[Table]: table associated with Expr
        """
        return self._table

    def to_table(self) -> "Table":
        """
        Returns a Table

        Method for Function object
        """
        from greenplumpython.table import Table

        from_clause = f"FROM {self.table.name}" if self.table is not None else ""
        parents = [self.table] if self.table is not None else []
        return Table(f"SELECT {str(self)} {from_clause}", parents=parents, db=self._db)


class BinaryExpr(Expr):
    """
    Inherited from Expr.

    Representation of a Binary Expression
    """

    @singledispatchmethod
    def _init(
        self,
        operator: str,
        left: Any,
        right: Any,
        as_name: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        table: Optional[Table] = None
        if isinstance(left, Expr) and left.table is not None:
            table = left.table
        if isinstance(right, Expr) and right.table is not None:
            table = right.table
        super().__init__(as_name=as_name, table=table, db=db)
        self.operator = operator
        self.left = left
        self.right = right

    @overload
    def __init__(
        self,
        operator: str,
        left: "Expr",
        right: "Expr",
        as_name: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        ...

    @overload
    def __init__(
        self,
        operator: str,
        left: "Expr",
        right: int,
        as_name: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        ...

    @overload
    def __init__(
        self,
        operator: str,
        left: "Expr",
        right: str,
        as_name: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        ...

    def __init__(
        self,
        operator: str,
        left: Any,
        right: Any,
        as_name: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        """

        Args:
            left: Any : could be an Expr object or object in primitive types (int, str, etc)
            right: Any : could be an Expr object or object in primitive types (int, str, etc)
        """
        self._init(operator, left, right, as_name, db)

    def _serialize(self) -> str:
        from greenplumpython.type import to_pg_const

        left_str = str(self.left) if isinstance(self.left, Expr) else to_pg_const(self.left)
        right_str = str(self.right) if isinstance(self.right, Expr) else to_pg_const(self.right)
        return f"({left_str} {self.operator} {right_str})"


class UnaryExpr(Expr):
    """
    Inherited from Expr.

    Representation of a Unary Expression.
    """

    def __init__(
        self,
        operator: str,
        right: Expr,
        as_name: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        """

        Args:
            right: Expr
        """
        table = right.table
        super().__init__(as_name=as_name, table=table, db=db)
        self.operator = operator
        self.right = right

    def _serialize(self) -> str:
        right_str = str(self.right)
        return f"{self.operator}({right_str})"


class Column(Expr):
    """
    Inherited from Expr.

    Representation of a Python object Column.
    """

    def __init__(self, name: str, table: "Table", as_name: Optional[str] = None) -> None:
        super().__init__(as_name=as_name, table=table)
        self._name = name
        self._type: Optional[Type] = None  # TODO: Add type inference

    def _serialize(self) -> str:
        assert self.table is not None
        return self.table.name + "." + self.name

    @property
    def name(self) -> str:
        """
        Returns column name

        Returns:
            str: column name
        """
        return self._name

    @property
    def table(self) -> Optional["Table"]:
        """
        Returns column associated table

        Returns:
            Optional[Table]: table associated with column
        """
        return self._table
