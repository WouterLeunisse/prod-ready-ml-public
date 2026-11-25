from unittest import mock

# Uncomment this
from app import Item, app, get_item, items, update_items
from fastapi.testclient import TestClient


# Test the program logic
def test_update_items():
    item = Item(name="candy", description="nice chocolates", price=5, tax=1)

    res = update_items(item)
    assert res["price_with_tax"] == 6


def test_get_item():
    item = Item(name="candy", description="nice chocolates", price=5, tax=1)
    update_items(item)

    res = get_item("candy")
    assert res == {
        "name": "candy",
        "description": "nice chocolates",
        "price": 5,
        "tax": 1,
        "price_with_tax": 6,
    }


# Test the endpoints
client = TestClient(app)


def test_get_item_endpoint():
    items["foo"] = {"my": "fake item"}

    res = client.get("/item/foo")
    assert res.status_code == 200
    assert res.json() == {"my": "fake item"}


# Test a mocked-up endpoint
def test_get_item_mocked():
    with mock.patch("app.get_item", return_value={"my": "faked fake item"}):
        res = client.get("/item/foo")
        assert res.status_code == 200
        assert res.json() == {"my": "faked fake item"}


def test_get_item_endpoint_filters_sensitive_fields():
    items["bar"] = {
        "name": "bar",
        "description": "desc",
        "price": 10,
        "tax": 2,
        "price_with_tax": 12,
    }
    res = client.get("/item/bar")
    assert res.status_code == 200
    body = res.json()
    assert body == {"price_with_tax": 12}
    assert "price" not in body
    assert "tax" not in body
