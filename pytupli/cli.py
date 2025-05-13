"""Command-line interface for the Tupli API client.

This module provides a command-line interface for interacting with the Tupli API,
offering formatted output of benchmark, episode, and artifact data. It uses the
Python Fire library to automatically generate CLI commands from the TupliAPIClient
class methods.
"""

from typing import Any, Optional, Union
from pydantic import BaseModel
from pytupli.storage import TupliAPIClient
from fire import Fire
import tabulate


def pretty_printer(result: Any) -> Optional[Union[str, Any]]:
    """Formats API results for display in the terminal.

    This function handles different types of results and formats them appropriately:
    - Lists of dictionaries or BaseModel instances are displayed as tables
    - Strings are wrapped with newlines
    - Other types are passed through to Fire's default formatting

    Args:
        result: The data to format. Can be None, a list of dicts/BaseModels, or any other type.

    Returns:
        Optional[Union[str, Any]]: The formatted string representation of the result,
            or None if the input was None.
    """
    if result is None:
        return

    # display a list of dicts as a table
    if isinstance(result, (list, tuple)) and (
        all(isinstance(x, dict) for x in result) or all(isinstance(x, BaseModel) for x in result)
    ):
        return tabulate.tabulate(
            [
                {
                    col: cell_format(value)
                    for col, value in (
                        row.items() if isinstance(row, dict) else row.model_dump().items()
                    )
                }
                for row in result
            ],
            headers='keys',
        )

    return (
        '\n' + result + '\n' if isinstance(result, str) else result
    )  # otherwise, let fire handle it


def cell_format(value: Any, decimals: int = 3, bool: tuple[str, str] = ('✅', '❌')) -> str:
    """Formats individual cell values for table display.

    Args:
        value: The value to format. Can be a boolean, float, or any other type.
        decimals (int, optional): Number of decimal places to show for float values.
            Defaults to 3.
        bool (tuple[str, str], optional): Unicode characters to use for True and False
            values. Defaults to ('✅', '❌').

    Returns:
        str: The formatted string representation of the value.
    """
    if value is True:
        return bool[0]
    if value is False:
        return bool[1]
    if isinstance(value, float):
        return '{:.{}f}'.format(value, decimals)
    return value


def main() -> None:
    """Entry point for the Tupli CLI.

    Creates a command-line interface using Python Fire, exposing all TupliAPIClient
    methods with formatted output through pretty_printer.
    """
    Fire(TupliAPIClient, name='pytupli', serialize=pretty_printer)


if __name__ == '__main__':
    main()
