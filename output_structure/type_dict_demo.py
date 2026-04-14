from typing import TypedDict

class product_info(TypedDict):
    name: str
    cost: float
    quantity: int
new_product = product_info(name="laptop", cost=1000.9, quantity=10)
print(new_product)