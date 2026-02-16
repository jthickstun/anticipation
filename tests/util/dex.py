"""Dev Ex stuff."""


class Printer:
    _COLOR_END = "\033[0m"
    _RED = "\033[0;31m"
    _GREEN = "\033[0;32m"
    _YELLOW = "\033[1;33m"

    @staticmethod
    def _print_col(my_statement: str, color_code: str) -> None:
        s = f"{color_code}{my_statement}{Printer._COLOR_END}"
        print(s)

    @staticmethod
    def red(my_statement: str) -> None:
        Printer._print_col(my_statement, Printer._RED)

    @staticmethod
    def green(my_statement: str) -> None:
        Printer._print_col(my_statement, Printer._GREEN)

    @staticmethod
    def yellow(my_statement: str) -> None:
        Printer._print_col(my_statement, Printer._YELLOW)
