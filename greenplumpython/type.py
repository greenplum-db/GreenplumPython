# noqa: D100
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_type_hints,
)
from uuid import uuid4

from greenplumpython.db import Database
from greenplumpython.expr import Expr, _serialize_to_expr

if TYPE_CHECKING:
    from greenplumpython.dataframe import DataFrame


class TypeCast(Expr):
    """
    An expression of type casting.

    Example:
            .. highlight:: python
            .. code-block::  Python

                >>> rows = [("01-01-1990",), ("05-01-98",)]
                >>> df = db.create_dataframe(rows=rows, column_names=["date_str"])
                >>> date_type = gp.type_("date")
                >>> result = df.assign(date=lambda t: date_type(t["date_str"]))
                >>> result
                -------------------------
                 date_str   | date
                ------------+------------
                 01-01-1990 | 1990-01-01
                 05-01-98   | 1998-05-01
                -------------------------
                (2 rows)
    """

    def __init__(self, obj: object, qualified_type_name: str) -> None:
        # noqa: D205 D400
        """
        Args:
            obj: object : which will be applied type casting
            qualified_type_name : str : qualified name of type which object will be cast
        """
        dataframe = obj._dataframe if isinstance(obj, Expr) else None
        super().__init__(dataframe)
        self._obj = obj
        self._qualified_type_name = qualified_type_name

    def _serialize(self, db: Optional[Database] = None) -> str:
        obj_str = _serialize_to_expr(self._obj, db=db)
        return f"({obj_str}::{self._qualified_type_name})"


class DataType:
    """
    Represents a type of values in a :class:`~dataframe.DataFrame`.

    It is mapped to a type in database, including

        - A predefined type, such as :code:`integer` for :class:`int` and\
            :code:`text` for :class:`str` in Python. In this case,\
            `name` is specified for its name in database.
        - A user-defined composite type, for Python :code:`class`. In this\
            case, a type annotation object is provided such as the defined\
            :code:`class`.

    A :class:`~type.DataType` object is callable. when called, it casts the object in
    the argument to the mapped type in database.
    """

    def __init__(
        self,
        name: str,
        annotation: Optional[type] = None,
        schema: Optional[str] = None,
        modifier: Optional[int] = None,
    ) -> None:
        # noqa: D107
        self._name = name
        self._annotation = annotation
        self._created_in_dbs: Optional[Set[Database]] = set() if annotation is not None else None
        self._schema = schema
        self._modifier = modifier
        self._qualified_name_str = f'"{self._name}"'
        if self._schema is not None:
            self._qualified_name_str = f'"{self._schema}".' + self._qualified_name_str
        if self._modifier is not None:
            self._qualified_name_str += f"({self._modifier})"

    def _serialize(self, db: Database) -> str:
        if self._annotation is None:
            raise Exception("No type annotation to serialize")
        members = get_type_hints(self._annotation)
        if len(members) == 0:
            raise Exception(f"Failed to get annotations for type {self._annotation}")
        members_str = ",\n".join(
            [f"{name} {_serialize_to_type_name(type_t, db)}" for name, type_t in members.items()]
        )
        return f"({members_str})"

    # -- Creation of a composite type in Greenplum corresponding to the class_type given
    def _create_in_db(self, db: Database):
        # noqa: D400
        """
        :meta private:

        Create a new composite type in database and returns its name.

        Args:
            db : :class:`~db.Database` : where the type will be created

        Returns:
            str: name of the created composite type

        """
        if self._created_in_dbs is None or db in self._created_in_dbs:
            return
        assert isinstance(
            self._annotation, type
        ), "Only composite data types can be created in database."
        schema = "pg_temp"

        db._execute(
            f'CREATE TYPE "{schema}"."{self._name}" AS {self._serialize(db=db)};',
            has_results=False,
        )
        self._created_in_dbs.add(db)

    def __call__(self, obj: Any) -> TypeCast:
        """
        Cast the argument :code:`obj` to the corresponding type in database.

        Args:
            obj: the object to be cast. It can be one of the following

                - Any adaptable Python object, or
                - A :class:`Column` of a :class:`DataFrame`, or
                - Any :class:`Expr` consisting of adaptable Python objects and
                    :class:`Column`s of a :class:`DataFrame`.
        """
        return TypeCast(obj, self._qualified_name_str)

    @property
    def _qualified_name(self) -> Tuple[Optional[str], str]:
        """
        Return the schema name and name of :class:`~type.DataType`.

        Returns:
            Tuple[str, str]: schema name and :class:`~type.DataType`'s name.
        """
        return self._schema, self._name


# -- Map between Python and Greenplum primitive types
_defined_types: Dict[Optional[type], DataType] = {
    None: DataType(name="void"),
    int: DataType(name="int4"),
    float: DataType(name="float8"),
    bool: DataType(name="bool"),
    str: DataType(name="text"),
    bytes: DataType(name="bytea"),
}


def type_(name: str, schema: Optional[str] = None, modifier: Optional[int] = None) -> DataType:
    """
    Get access to a type predefined in database.

    Args:
        name: str: name of type
        schema: Optional[str]: name of schema
        modifier: Optional[int]: variable or fixed length depending on type

    Returns:
        The predefined type as a :class:`~type.Type` object.
    """
    return DataType(name, schema=schema, modifier=modifier)


def _serialize_to_type_name(
    annotation: Union[DataType, type],
    db: Database,
    for_return: bool = False,
) -> str:
    # noqa: D400
    """
    :meta private:

    Convert a Python type annotation to a SQL type.

    Args:
        annotation: type annotation in Python
        schema: schema name
        db: database that the type will be created in if not present.
        for_return: if the type is used as a function's return type

    Returns:
        str: name of type in SQL
    """
    if hasattr(annotation, "__origin__"):
        # The `or` here is to make the function work on Python 3.6.
        # Python 3.6 is the default Python version on CentOS 7 and Ubuntu 18.04
        if annotation.__origin__ == list or annotation.__origin__ == List:
            args: Tuple[type, ...] = annotation.__args__
            if for_return:
                return f"SETOF {_serialize_to_type_name(args[0], db)}"  # type: ignore
            else:
                return f"{_serialize_to_type_name(args[0], db)}[]"  # type: ignore
        raise NotImplementedError("Only list is supported as generic data type")
    else:
        if isinstance(annotation, DataType):
            return annotation._qualified_name_str
        assert db is not None, "Database is required to create type"
        if annotation not in _defined_types:
            type_name = "type_" + uuid4().hex
            _defined_types[annotation] = DataType(name=type_name, annotation=annotation)
        _defined_types[annotation]._create_in_db(db)
        return _defined_types[annotation]._qualified_name_str
