"""To create and call Greenplum/PostgreSQL UDFs or UDAs."""
import ast
import functools
import inspect
import json
import sys
import sysconfig
from textwrap import dedent
from typing import Any, Callable, List, Literal, Optional, Set, Tuple
from uuid import uuid4

import dill  # type: ignore reportMissingTypeStubs

dill.settings["recurse"] = True

from greenplumpython.col import Column
from greenplumpython.dataframe import DataFrame
from greenplumpython.db import Database
from greenplumpython.expr import Expr, _serialize_to_expr
from greenplumpython.group import DataFrameGroupingSet
from greenplumpython.type import _defined_types, _serialize_to_type_name


class FunctionExpr(Expr):
    """
    Inherited from :class:`~expr.Expr`.

    A Function Expression object associated with a Greenplum/PostgreSQL function which can be called
    and applied to the data in the database.
    """

    def __init__(
        self,
        func: "_AbstractFunction",
        args: Tuple[Any] = [],
        group_by: Optional[DataFrameGroupingSet] = None,
        dataframe: Optional[DataFrame] = None,
        distinct: bool = False,
    ) -> None:
        # noqa D400
        if dataframe is None and group_by is not None:
            dataframe = group_by._dataframe
        for arg in args:
            if isinstance(arg, Expr) and arg._dataframe is not None:
                if dataframe is None:
                    dataframe = arg._dataframe
                elif dataframe._name != arg._dataframe._name:
                    raise Exception("Cannot pass arguments from more than one dataframes")
        super().__init__(dataframe=dataframe)
        self._func = func
        self._args = args
        self._group_by = group_by
        self._distinct = distinct

    def _bind(
        self,
        group_by: Optional[DataFrameGroupingSet] = None,
        dataframe: Optional[DataFrame] = None,
    ):
        # noqa D400
        """:meta private:"""
        return FunctionExpr(
            self._func,
            self._args,
            group_by=group_by,
            dataframe=dataframe,
            distinct=self._distinct,
        )

    def _serialize(self, db: Optional[Database] = None) -> str:
        # noqa D400
        """:meta private:"""
        if db is not None:
            self._function._create_in_db(db)
        distinct = "DISTINCT" if self._distinct else ""
        args_string = (
            ",".join([_serialize_to_expr(arg, db=db) for arg in self._args])
            if any(self._args)
            else ""
            if not isinstance(self._func, AggregateFunction)
            else "*"
        )
        return f"{self._function._qualified_name_str}({distinct} {args_string})"

    def apply(
        self, expand: bool = False, column_name: Optional[str] = None, db: Optional[Database] = None
    ) -> DataFrame:
        # noqa D400
        """
        :meta private:

        Returns the resulting :class:`DataFrame` of the self function applied
        to args, with potential GROUP BY if it is an Aggregation function.
        """
        assert not (
            expand and column_name is not None
        ), "Cannot assign single column name when expanding multi-valued results."
        self._function._create_in_db(db=db)
        from_clause = f"FROM {self._dataframe._name}" if self._dataframe is not None else ""
        group_by_clause = self._group_by._clause() if self._group_by is not None else ""
        if expand and column_name is None:
            column_name = "func_" + uuid4().hex
        parents = [self._dataframe] if self._dataframe is not None else []
        grouping_col_names = self._group_by._flatten() if self._group_by is not None else None
        # FIXME: The names of GROUP BY exprs can collide with names of fields in
        # the comosite type, making the column names of the resulting dataframe not
        # unique. This can be mitigated after we implement dataframe column
        # inference by raising an error when the function gets called.
        grouping_cols = (
            [Column(name, self._dataframe)._serialize(db=None) for name in grouping_col_names]
            if grouping_col_names is not None and len(grouping_col_names) != 0
            else None
        )
        if (
            isinstance(self._function, NormalFunction)
            and self._function._language_handler == "plcontainer"
        ):
            return_annotation = inspect.signature(self._function._wrapped_func).return_annotation  # type: ignore reportUnknownArgumentType
            _serialize_to_type_name(return_annotation, db=db, for_return=True)
            input_args = self._args
            if len(input_args) == 0:
                raise Exception("No input data specified, please specify a DataFrame or Columns")
            input_clause = (
                "*"
                if (len(input_args) == 1 and isinstance(input_args[0], DataFrame))
                else ",".join([arg._serialize(db=db) for arg in input_args])
            )
            return DataFrame(
                f"""
                SELECT * FROM plcontainer_apply(TABLE(
                    SELECT {input_clause} {from_clause}), '{self._function._qualified_name_str}', 4096) AS
                    {_defined_types[return_annotation.__args__[0]]._serialize(db=db)}
                """,
                db=db,
                parents=parents,
            )
        else:
            unexpanded_dataframe = DataFrame(
                " ".join(
                    [
                        f"SELECT {_serialize_to_expr(self, db=db)} {'AS ' + column_name if column_name is not None else ''}",
                        ("," + ",".join(grouping_cols)) if (grouping_cols is not None) else "",
                        from_clause,
                        group_by_clause,
                    ]
                ),
                db=db,
                parents=parents,
            )
            # We use 2 `DataFrame`s here because on GPDB 6X and PostgreSQL <= 9.6, a
            # function returning records that contains more than one attributes
            # will be called multiple times if we do
            # ```sql
            # SELECT (func(a, b)).* FROM t;
            # ```
            # which might cause performance issue. To workaround we need to do
            # ```sql
            # WITH func_call AS (
            #     SELECT func(a, b) AS result FROM t
            # )
            # SELECT (result).* FROM func_call;
            # ```
            rebased_grouping_cols = (
                [
                    _serialize_to_expr(unexpanded_dataframe[name], db=db)
                    for name in grouping_col_names
                ]
                if grouping_col_names is not None
                else None
            )
            result_cols = (
                _serialize_to_expr(unexpanded_dataframe["*"], db=db)
                if not expand
                else _serialize_to_expr(unexpanded_dataframe[column_name]["*"], db=db)
                # `len(rebased_grouping_cols) == 0` means `GROUP BY GROUPING SETS (())`
                if rebased_grouping_cols is None or len(rebased_grouping_cols) == 0
                else f"({unexpanded_dataframe._name}).*"
                if not expand
                else f"{','.join(rebased_grouping_cols)}, {_serialize_to_expr(unexpanded_dataframe[column_name]['*'], db=db)}"
            )

            return DataFrame(
                f"SELECT {result_cols} FROM {unexpanded_dataframe._name}",
                db=db,
                parents=[unexpanded_dataframe],
            )

    @property
    def _function(self) -> "_AbstractFunction":
        return self._func


class ArrayFunctionExpr(FunctionExpr):
    """
    Inherited from :class:`~func.FunctionExpr`.

    Specialized for an Array Function.
    It will array aggregate all the columns given by the user.
    """

    def _serialize(self, db: Optional[Database] = None) -> str:
        # noqa D400
        """:meta private:"""
        if db is not None:
            self._function._create_in_db(db)
        args_string_list = []
        args_string = ""
        grouping_col_names = self._group_by._flatten() if self._group_by is not None else None
        grouping_cols = (
            {Column(name, self._dataframe) for name in grouping_col_names}
            if grouping_col_names is not None
            else None
        )
        if any(self._args):
            for i in range(len(self._args)):
                if self._args[i] is None:
                    continue
                if isinstance(self._args[i], Expr):
                    if grouping_cols is None or self._args[i] not in grouping_cols:
                        s = f"array_agg({_serialize_to_expr(self._args[i], db=db)})"  # type: ignore
                    else:
                        s = _serialize_to_expr(self._args[i], db=db)  # type: ignore
                else:
                    s = _serialize_to_expr(self._args[i], db=db)
                args_string_list.append(s)
            args_string = ",".join(args_string_list)
        return f"{self._function._qualified_name_str}({args_string})"

    def _bind(
        self,
        group_by: Optional[DataFrameGroupingSet] = None,
        dataframe: Optional[DataFrame] = None,
    ):
        # noqa D400
        """:meta private:"""
        return ArrayFunctionExpr(
            self._func,
            self._args,
            group_by=group_by if group_by else self._group_by,
            dataframe=dataframe,
        )


# The parent class for all database functions.
# It is not a Callable by design to prevent misuse.
class _AbstractFunction:
    def __init__(
        self,
        wrapped_func: Optional[Callable[..., Any]],
        name: Optional[str],
        schema: Optional[str],
    ) -> None:
        # if wrapped_func is None, the function object is obtained by
        # gp.function() rather than gp.create_function(). Otherwise a
        # Python function will be passed to wrapped_func.
        self._name = "func_" + uuid4().hex if wrapped_func is not None else name
        assert self._name is not None
        self._schema = "pg_temp" if wrapped_func is not None else None if schema is None else schema
        self._qualified_name_str = (
            f'"{self._name}"' if self._schema is None else f'"{self._schema}"."{self._name}"'
        )

    @property
    def _qualified_name(self) -> Tuple[Optional[str], str]:
        return self._schema, self._name

    def _create_in_db(self, _: Database) -> None:
        # noqa D400
        """:meta private:"""
        raise NotImplementedError("Cannot create abstract function in database")


class NormalFunction(_AbstractFunction):
    """
    Represent a (normal) dataframe function.

    The function can be applied to:

    - a :class:`~dataframe.DataFrame` with :meth:`~dataframe.DataFrame.assign` or
      :meth:`~dataframe.DataFrame.apply`;
    - a :class:`~db.Database` when no :class:`~dataframe.DataFrame` is involved with
      :meth:`~db.Database.assign` or :meth:`~db.Database.apply`.

    A :class:`~func.NormalFunction` is mapped to a User-Defined Function (UDF) in
    database.

    When called, the arguments of an :class:`~func.AggregateFunction` can be

    - :class:`~col.Column` of a :class:`~dataframe.DataFrame`; or
    - constant values represented as Python objects

    and the :class:`~func.AggregateFunction` returns one value of the return type
    for each row of values in its arguments.
    """

    def __init__(
        self,
        wrapped_func: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        schema: Optional[str] = None,
        language_handler: Literal["plpython3u"] = "plpython3u",
        runtime: Optional[str] = None,
    ) -> None:
        # noqa D107
        super().__init__(wrapped_func, name, schema)
        self._created_in_dbs: Optional[Set[Database]] = set() if wrapped_func is not None else None
        self._wrapped_func = wrapped_func
        self._language_handler = language_handler
        self._runtime = runtime

    def unwrap(self) -> Callable[..., Any]:
        """Get the wrapped Python function in the database function."""
        assert self._wrapped_func is not None, "No Python function is wrapped inside."
        return self._wrapped_func

    def _serialize(self, db: Database) -> str:
        # tricky way to run if the code run in doctest
        # if it runs in doctest, there is bug about dill
        # we need to pass it
        if "doctest" in sys.modules:
            func_src: str = inspect.getsource(self._wrapped_func)
        else:
            func_src: str = dill.source.getsource(self._wrapped_func)
            assert isinstance(func_src, str)
        func_ast: ast.FunctionDef = ast.parse(dedent(func_src)).body[0]
        # TODO: Lambda expressions are NOT supported since inspect.signature()
        # does not work as expected.
        assert isinstance(
            func_ast, ast.FunctionDef
        ), f"{self._wrapped_func} is not a function. (lambda is not supported.)"
        func_sig = inspect.signature(self._wrapped_func)
        func_args = ",".join(
            [
                f'"{param.name}" {_serialize_to_type_name(param.annotation, db=db)}'
                for param in func_sig.parameters.values()
            ]
        )
        func_arg_names = ",".join(
            [f"{param.name}={param.name}" for param in func_sig.parameters.values()]
        )
        return_type = (
            _serialize_to_type_name(func_sig.return_annotation, db=db, for_return=True)
            if self._language_handler != "plcontainer"
            else "SETOF record"
        )
        func_pickled: bytes = dill.dumps(self._wrapped_func)
        _, func_name = self._qualified_name
        # Modify the AST of the wrapped function to minify dependency: (1-3)
        # 1. Apply random renaming to avoid name conflict. (TODO: Support
        #    calling another UDF in the current UDF directly.)
        func_ast.name = "__" + func_name
        # 2. Clear decorators and type annotations to avoid import.
        func_ast.decorator_list.clear()
        for arg in func_ast.args.args:
            arg.annotation = None
        func_ast.returns = None
        # 3. Prepend imports for modules referred to in the body.
        global_objects: List[Any] = dill.detect.globalvars(self._wrapped_func).values()
        importables: List[str] = [dill.source.getimportable(obj) for obj in global_objects]
        importables_ast: List[ast.Import] = ast.parse(dedent("".join(importables))).body
        func_ast.body = importables_ast + func_ast.body

        pickle_lib_name: str = "__lib_" + uuid4().hex
        sysconfig_lib_name: str = "__lib_" + uuid4().hex
        python_version = sysconfig.get_python_version()
        sys_lib_name: str = "__lib_" + uuid4().hex
        return (
            f"CREATE FUNCTION {self._qualified_name_str} ({func_args}) "
            f"RETURNS {return_type} "
            f"AS $$\n"
            f"# container: {self._runtime}\n"
            f"try:\n"
            f"    return GD['{func_ast.name}']({func_arg_names})\n"
            f"except KeyError:\n"
            f"    try:\n"
            f"        import dill as {pickle_lib_name}\n"
            f"        import sysconfig as {sysconfig_lib_name}\n"
            f"        import sys as {sys_lib_name}\n"
            f"        if {sysconfig_lib_name}.get_python_version() != '{python_version}':\n"
            f"            raise ModuleNotFoundError\n"
            f"        {sys_lib_name}.modules['plpy']=plpy\n"
            f"        setattr({sys_lib_name}.modules['plpy'], '_SD', SD)\n"
            f"        GD['{func_ast.name}'] = {pickle_lib_name}.loads({func_pickled})\n"
            f"    except ModuleNotFoundError:\n"
            f"        exec({json.dumps(ast.unparse(func_ast))}, globals())\n"
            f"        GD['{func_ast.name}'] = globals()['{func_ast.name}']\n"
            f"    return GD['{func_ast.name}']({func_arg_names})\n"
            f"$$ LANGUAGE {self._language_handler};"
        )

    def _create_in_db(self, db: Database) -> None:
        if self._wrapped_func is None:  # Function has already existed.
            return
        assert self._created_in_dbs is not None
        if db not in self._created_in_dbs:
            assert db._execute(self._serialize(db=db), has_results=False) == -1
            self._created_in_dbs.add(db)

    def __call__(self, *args: Any) -> FunctionExpr:
        """Call the dataframe function with the given arguments."""
        return FunctionExpr(self, args)


def function(name: str, schema: Optional[str] = None) -> NormalFunction:
    """
    Get access to a predefined dataframe :class:`~func.NormalFunction` from database.

    Args:
        name: Name of the function.
        schema: Schema (a.k.a namespace) of the function in database.

    Returns
        The :class:`~func.NormalFunction` with the specified :code:`name`
        and :code:`schema`.

    Example:
        .. code-block::  Python

            >>> generate_series = gp.function("generate_series")
            >>> db.apply(lambda: generate_series(0, 2))
            -----------------
             generate_series
            -----------------
                           0
                           1
                           2
            -----------------
            (3 rows)

    """
    return NormalFunction(name=name, schema=schema)


class AggregateFunction(_AbstractFunction):
    """
    Represent an aggregate function.

    The function can be applied to:

    - a :class:`~dataframe.DataFrame` with :meth:`~dataframe.DataFrame.apply`, where the function\
        will aggregate data in the entire dataframe;
    - a :class:`~group.DataFrameGroupingSet` with :meth:`~group.DataFrameGroupingSet.assign`\
        or :meth:`~group.DataFrameGroupingSet.apply`, where the function will\
        aggregate each group of data.

    An :class:`~func.AggregateFunction` is mapped to a User-Defined Aggregate (UDA)
    function in database.

    When called, the arguments of an :class:`~func.AggregateFunction` can be

    - :class:`~col.Column` of a :class:`~dataframe.DataFrame`; or

    - constant values represented as Python objects.

    And the :class:`~func.AggregateFunction` returns one value aggregating data in all
    rows of the :class:`~dataframe.DataFrame` or a group in the
    :class:`~group.DataFrameGroupingSet`.
    """

    def __init__(
        self,
        transition_func: Optional[NormalFunction] = None,
        name: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        # noqa D107
        super().__init__(
            transition_func.unwrap() if transition_func is not None else None,
            name,
            schema,
        )
        self._transition_func = transition_func
        self._created_in_dbs: Optional[Set[Database]] = (
            set() if transition_func is not None else None
        )

    @property
    def transition_function(self) -> NormalFunction:
        """Return the transition function of the aggregate function."""
        assert (
            self._transition_func is not None
        ), f"Transition function of the aggregate function {self._qualified_name_str} is unknown."
        return self._transition_func

    def _create_in_db(self, db: Database) -> None:
        # If self._transition_func is None, then the aggregate function is not
        # created with gp.create_aggregate(), but only refers to an existing
        # aggregate function.
        if self._transition_func is None:
            return
        assert self._created_in_dbs is not None
        if db not in self._created_in_dbs:
            self._transition_func._create_in_db(db)
            sig = inspect.signature(self.transition_function.unwrap())
            param_list = iter(sig.parameters.values())
            state_param = next(param_list)
            args_string = ",".join(
                [
                    f"{param.name} {_serialize_to_type_name(param.annotation, db=db)}"
                    for param in param_list
                ]
            )
            # -- Creation of UDA in Greenplum
            db._execute(
                (
                    f"CREATE AGGREGATE {self._qualified_name_str} ({args_string}) (\n"
                    f"    SFUNC = {self.transition_function._qualified_name_str},\n"
                    f"    STYPE = {_serialize_to_type_name(state_param.annotation, db=db)}\n"
                    f");\n"
                ),
                has_results=False,
            )
            self._created_in_dbs.add(db)

    def distinct(self, *args: Any) -> FunctionExpr:
        """
        Apply the current aggregate function to each distinct set of the arguments.

        Args:
            args: Argument of the aggregate function.

        Returns:
            FunctionExpr: An expression represents the function call.


        Example:

            .. highlight:: python
            .. code-block::  Python

                >>> rows = [(1,), (2,), (2,), (3,), (3,), (4,)]
                >>> numbers = db.create_dataframe(rows=rows, column_names=["val"])
                >>> count = gp.aggregate_function("count")
                >>> results = numbers.group_by().assign(
                ...     unique_numbers_count=lambda t: count.distinct(t["val"]))
                >>> results
                ----------------------
                 unique_numbers_count
                ----------------------
                                    4
                ----------------------
                (1 row)
        """
        return FunctionExpr(self, args, distinct=True)

    def __call__(self, *args: Any) -> FunctionExpr:
        """Call the dataframe function with the given arguments."""
        return FunctionExpr(self, args)


def aggregate_function(name: str, schema: Optional[str] = None) -> AggregateFunction:
    """
    Get access to a predefined :class:`~func.AggregateFunction` from the database.

    Args:
        name: Name of the aggregate function.
        schema: Schema (a.k.a namespace) of the aggregate function in database.

    Returns:
        The :class:`~func.AggregateFunction` with the specified :code:`name`
        and :code:`schema`.

    Example:
        .. highlight:: python
        .. code-block::  Python

            >>> array_agg = gp.aggregate_function("array_agg")
            >>> df = db.create_dataframe(columns={"i": range(3)})
            >>> result = df.apply(lambda t: array_agg(t['i']), column_name="aggregate_result")
            >>> result
            ------------------
             aggregate_result
            ------------------
             [0, 1, 2]
            ------------------
            (1 row)
    """
    return AggregateFunction(name=name, schema=schema)


# FIXME: Add test cases for optional parameters
def create_function(
    wrapped_func: Optional[Callable[..., Any]] = None,
    language_handler: Literal["plpython3u"] = "plpython3u",
    runtime: Optional[str] = None,
) -> NormalFunction:
    """
    Create a :class:`~func.NormalFunction` from the given Python function.

    Args:
        wrapped_func: the Python function carrying out the computation. Its
            definition need to follow the conventions below:

            - The function needs to be defined with the :code:`def` keyword.\
                Lambda expressions as the wrapped function are not supported\
                yet.
            - Each parameter and the return value needs to be annotated with\
                native Python type. The type annotations will be mapped to\
                the types in database automatically.
            - A :class:`~func.NormalFunction` can return multiple values. In that\
                case, the return type of the wrapped Python function needs\
                to be a Python :code:`class` with members annotated. It is\
                recommended to use :class:`dataclasses.dataclass` as return\
                type.

        language_handler: language handler to run the function in database,
            defaults to plpython3u, will also support plcontainer later.

        schema: schema name

    Returns:
        The newly created :class:`~func.NormalFunction`.

    Note:
        The created function is actually executed on the remote database
        server. To send it to the server, when creating the dataframe function,

        - Package `dill <https://dill.readthedocs.io/en/latest/>`_, by the\
            Uncertainty Quantification Foundation, is used to serialize the\
            wrapped Python function and its dependencies when applicable.\
            Therefore, it is recommended to install dill on the host of the\
            backing database server.
        - If dill is not installed on the server, or the Python versions\
            between client and server does not match, the source code of\
            the wrapped Python function will be transmitted to the server,\
            along with all the import statements for dependencies used by the\
            function. In that case, the modules imported need to be installed\
            on server in advance.

    Example:
        .. highlight:: python
        .. code-block::  Python

            >>> @gp.create_function
            ... def multiply(a: int, b: int) -> int:
            ...    return a * b

            >>> db.assign(result=lambda: multiply(1, 2))
            --------
             result
            --------
                  2
            --------
            (1 row)

    """
    # If user needs extra parameters when creating a function
    if wrapped_func is None:
        return functools.partial(
            create_function, language_handler=language_handler, runtime=runtime
        )
    return NormalFunction(
        wrapped_func=wrapped_func, language_handler=language_handler, runtime=runtime
    )


# FIXME: Add test cases for optional parameters
def create_aggregate(
    transition_func: Optional[Callable[..., Any]] = None,
    language_handler: Literal["plpython3u"] = "plpython3u",
) -> AggregateFunction:
    """
    Create an :class:`~func.AggregateFunction` from the given Python function.

    Args:
        transition_func : the wrapped Python function carrying out the state
            transition. It needs to follow the same convention as the
            :code:`wrapped_func` parameter of :func:`~func.create_function`, and the
            notes on serialization also applied here.

        language_handler : language handler to run the function in database,
            defaults to plpython3u, will also support plcontainer later.

    Returns:
        The newly created :class:`~func.AggregateFunction`.

    Example:
        .. highlight:: python
        .. code-block::  Python

            >>> @gp.create_aggregate
            ... def my_sum(cur_sum: int, val: int) -> int:
            ...     if cur_sum is None:
            ...         return val
            ...     return cur_sum + val

            >>> rows = [(1,) for _ in range(10)]
            >>> numbers = db.create_dataframe(rows=rows, column_names=["val"])
            >>> results = numbers.group_by().assign(result=lambda t: my_sum(t["val"]))
            >>> results
            --------
             result
            --------
                 10
            --------
            (1 row)
    """
    # If user needs extra parameters when creating a function
    if transition_func is None:
        return functools.partial(create_aggregate, language_handler=language_handler)
    return AggregateFunction(
        transition_func=NormalFunction(
            transition_func,
            name="func_" + uuid4().hex,
            schema="pg_temp",
            language_handler=language_handler,
        )
    )


class ColumnFunction(NormalFunction):
    """
    Represent a dataframe column function.

    - a :class:`~dataframe.DataFrame` with :meth:`~dataframe.DataFrame.apply`, where the function\
        will operate on columns in the entire dataframe;
    - a :class:`~group.DataFrameGroupingSet` with :meth:`~group.DataFrameGroupingSet.assign`\
        or :meth:`~group.DataFrameGroupingSet.apply`, where the function will operate\
        on columns of each group of data.

    As :class:`~func.NormalFunction`, a :class:`~func.ColumnFunction` is mapped to a UDF in
    database.

    The calling convention of a :class:`~func.ColumnFunction` is the same as a
    :class:`~func.AggregateFunction`. However, rather than operating on one row at a
    time, all rows of the entire column are aggregated into a :code:`list`
    before passing to :class:`~func.ColumnFunction` as argument, except when the
    column is used as the grouping attribute in :meth:`~dataframe.DataFrame.group_by`.

    A :class:`~func.ColumnFunction` returns

    - One value of the return type when applied to a :class:`~dataframe.DataFrame`; or
    - One value for each group when applied to a\
        :class:`~group.DataFrameGroupingSet`.

    Note:
        The primary use case for column function is to implement complex
        analytics such as machine learning using your favorite Python packages.

        Inside a column function, the user can operate on all the data, rather
        than only part of it. As a result, the operation does **not** to satisfy
        certain restrictions such as `Additivity`_.
        This makes it possible to implement complex functions.

    Warning:
        However, such good usability comes at the cost of scalability.

        - Gathering data into one place makes it hard to exploit inter-machine\
            parallelism when backed by an MPP database system like Greenplum,\
            especially when the number of groups is small. Fortunately, this\
            can be alleviated because many Python packages are SIMD optimized.

        - When the backing database system is PostgreSQL-derived, such as\
            Greenplum, the size of one value cannot be larger than 1 GB. This\
            limits the size of problems column functions can solve. Currently,\
            one way to mitigate this issue is to break a large\
            :class:`~dataframe.DataFrame` into smaller groups and somehow combine\
            the results of the column function for all groups.

        .. _Additivity: https://en.wikipedia.org/wiki/Sigma-additive_set_function
    """

    def __call__(self, *args: Any) -> ArrayFunctionExpr:
        """Call the dataframe function with the given arguments."""
        return ArrayFunctionExpr(self, args=args)


# FIXME: Add test cases for optional parameters
def create_column_function(
    wrapped_func: Optional[Callable[..., Any]] = None,
    language_handler: Literal["plpython3u"] = "plpython3u",
) -> ColumnFunction:
    """
    Create an :class:`~func.ColumnFunction` from the given Python function.

    Args:
        wrapped_func: the wrapped Python function carrying out computation on
            columns. It needs to follow the same convention as the
            :code:`wrapped_func` parameter of :func:`~func.create_function`, and the
            notes on serialization also applied here.

        language_handler : language handler to run the function in database,
            defaults to plpython3u, will also support plcontainer later.

    Returns:
        The newly created :class:`~func.ColumnFunction`.

    Example:
            .. highlight:: python
            .. code-block::  Python

                >>> @gp.create_column_function
                ... def my_array_summary(val_list: List[int]) -> str:
                ...     return f'Length: {len(val_list)}, Sum: {sum(val_list)}'

                >>> rows = [(1,), (2,), (3,)]
                >>> numbers = db.create_dataframe(rows=rows, column_names=["val"])
                >>> results = numbers.group_by().assign(
                ...     summary=lambda t: my_array_summary(t["val"]))
                >>> results
                -------------------
                 summary
                -------------------
                 Length: 3, Sum: 6
                -------------------
                (1 row)
    """
    # If user needs extra parameters when creating a function
    if wrapped_func is None:
        return functools.partial(create_column_function, language_handler=language_handler)
    return ColumnFunction(wrapped_func=wrapped_func, language_handler=language_handler)
