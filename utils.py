from dataclasses import dataclass
from typing import Dict

@dataclass
class ReceiptData:
    receipt_path: str
    schema: Dict
    output: Dict