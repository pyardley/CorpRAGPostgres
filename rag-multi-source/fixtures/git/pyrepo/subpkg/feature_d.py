from ..shared import utils
from . import sibling


def run(value: int) -> str:
    return f"{utils.helper(value)}-{sibling.marker()}"
