import base64
import contextvars
import json
import os
import sqlite3
from typing import Optional

from dotenv import load_dotenv 
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating, get_ratings_for_products

load_dotenv()

# Support both .env (local) and Streamlit secrets (cloud)
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# Identifies the current shopper's session. The UI (app.py) sets this before each
# agent.invoke via set_session_id(); tools read it so orders are scoped per session
# instead of leaking across everyone who ever used the app.
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_session_id", default=None
)


def set_session_id(session_id: Optional[str]) -> None:
    """Set the active session id used to scope orders. Call before agent.invoke()."""
    _current_session_id.set(session_id)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
vision_llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status. Ratings are already attached
    to every result (computed from the reviews table; products with no reviews are rating 0),
    and results are sorted from highest to lowest rated. Do NOT call get_rating for these
    results — the average_rating is already correct.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic, average_rating, review_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = {
        row[0]: {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows
    }

    for r in get_ratings_for_products(list(products.keys())):
        products[r["product_id"]]["average_rating"] = r["average_rating"]
        products[r["product_id"]]["review_count"] = r["review_count"]

    result = sorted(
        products.values(),
        key=lambda p: (p["average_rating"], p["review_count"]),
        reverse=True,
    )
    return json.dumps(result)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def list_categories() -> str:
    """
    List all distinct product categories available in the store, with a count of
    products in each. Use this when the user wants to browse by category or asks
    what kinds of products are available.
    Returns a JSON array of objects, each with: category, product_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY category"
    )
    rows = cursor.fetchall()
    conn.close()
    categories = [{"category": row[0], "product_count": row[1]} for row in rows]
    return json.dumps(categories)


@tool
def list_products_by_category(
    category: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
) -> str:
    """
    List ALL products in a given category, with ratings already attached, sorted from
    highest to lowest rated. Use this when the user wants to see every product in a
    category (e.g. "what snacks do you have?", "show me all the honey").

    Optionally filter by maximum price and/or organic status. Ratings come from the
    reviews table; products with no reviews are treated as rating 0. Returns a JSON array
    of products, each with: id, name, category, price, description, is_organic,
    average_rating, review_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE category = ?"
    params: list = [category]
    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)
    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return json.dumps([])

    products = {
        row[0]: {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "description": row[4],
            "is_organic": bool(row[5]),
        }
        for row in rows
    }

    for r in get_ratings_for_products(list(products.keys())):
        products[r["product_id"]]["average_rating"] = r["average_rating"]
        products[r["product_id"]]["review_count"] = r["review_count"]

    result = sorted(
        products.values(),
        key=lambda p: (p["average_rating"], p["review_count"]),
        reverse=True,
    )
    return json.dumps(result)


@tool
def top_rated_by_category(
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
) -> str:
    """
    Return the single highest-rated product for each category, with ratings already
    attached. Use this when the user wants to narrow down by category and see the best
    (highest rated) product per category, or the best product in one specific category.

    Optionally restrict to one category, and/or filter by maximum price and organic status.
    Ratings are computed from the reviews table; products with no reviews are treated as
    rating 0. Returns a JSON array with one entry per category, each with: id, name,
    category, price, is_organic, average_rating, review_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, is_organic FROM products WHERE 1=1"
    params: list = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)
    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return json.dumps([])

    products = {
        row[0]: {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "is_organic": bool(row[4]),
        }
        for row in rows
    }

    ratings = get_ratings_for_products(list(products.keys()))
    for r in ratings:
        products[r["product_id"]]["average_rating"] = r["average_rating"]
        products[r["product_id"]]["review_count"] = r["review_count"]

    # Pick the highest-rated product per category (break ties by more reviews, then id).
    best_per_category: dict = {}
    for prod in products.values():
        cat = prod["category"]
        current = best_per_category.get(cat)
        if current is None or (
            prod["average_rating"],
            prod["review_count"],
            -prod["id"],
        ) > (
            current["average_rating"],
            current["review_count"],
            -current["id"],
        ):
            best_per_category[cat] = prod

    result = sorted(best_per_category.values(), key=lambda p: p["category"])
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns 
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    session_id = _current_session_id.get()
    cursor.execute(
        "INSERT INTO orders (session_id, product_id, product_name, price) VALUES (?, ?, ?, ?)",
        (session_id, product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def get_orders(limit: Optional[int] = 20) -> str:
    """
    Retrieve past orders from the database, most recent first. Use this whenever the user
    asks about their order(s) — e.g. "what is my final order?", "what did I order?",
    "show my orders", "is that all?". Do NOT reconstruct orders from the conversation;
    always call this tool so the answer reflects what was actually saved.

    Returns a JSON object with: order_count, total_spent, and orders (an array where each
    item has order_id, product_id, product_name, price, ordered_at). The most recent order
    is first in the array. Only orders placed in the current shopping session are returned.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    session_id = _current_session_id.get()
    sql = (
        "SELECT id, product_id, product_name, price, ordered_at "
        "FROM orders WHERE session_id IS ? ORDER BY id DESC"
    )
    params: list = [session_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    orders = [
        {
            "order_id": row[0],
            "product_id": row[1],
            "product_name": row[2],
            "price": row[3],
            "ordered_at": row[4],
        }
        for row in rows
    ]
    total = round(sum(o["price"] for o in orders), 2)
    return json.dumps(
        {"order_count": len(orders), "total_spent": total, "orders": orders}
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return f"Error: Image file not found at path: {image_path}"

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[   
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    try:
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_react_agent(
    tools = [search_products, get_rating, list_categories, list_products_by_category, top_rated_by_category, checkout, get_orders, describe_product_image],
    model = llm, 
    prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING BY CATEGORY — when the user wants to browse or narrow down by category, "
        "asks what categories/kinds of products exist, or asks for the best/highest-rated "
        "product per category (or the best in one category):\n"
        "1. If the user wants to know what categories exist, call list_categories.\n"
        "2. If the user wants to see ALL products in a category (e.g. 'what snacks do you "
        "   have?', 'show me all the honey'), call list_products_by_category ONCE with that "
        "   category (pass max_price/is_organic only if specified). It returns every product "
        "   in the category with ratings already attached — do NOT call get_rating for these.\n"
        "3. If the user wants the highest-rated product per category (or per one category), "
        "   call top_rated_by_category ONCE (pass category/max_price/is_organic only if the "
        "   user specified them). This returns one product per category with ratings already "
        "   attached — do NOT call get_rating again for these.\n"
        "4. Present the results using the same numbered product format described in the "
        "   BROWSING section below.\n"
        "5. Do NOT call checkout at this stage.\n\n"
        "BROWSING — when the user describes a specific product they want to buy:\n"
        "1. Call search_products to find matching items (apply any price/organic filters given). "
        "   The results already include average_rating and review_count and are sorted best-first "
        "   — use those values directly and do NOT call get_rating.\n"
        "2. Filter by the user's minimum rating if specified.\n"
        "3. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "4. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "5. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message.\n\n"
        "ORDER HISTORY — when the user asks about what they have ordered (e.g. 'what is my "
        "final order?', 'what did I order?', 'show my orders', 'is that all?', 'my order "
        "total'):\n"
        "1. ALWAYS call get_orders — never answer from memory or from the chat history, and "
        "   never claim there is only one order without checking.\n"
        "2. If order_count is 0, tell the user they have no orders yet.\n"
        "3. Otherwise list every order in plain text, most recent first, e.g. "
        "   'Order #<id> — <product_name> — $<price>', then state the total: "
        "   'Total: $<total_spent> across <order_count> order(s).'\n"
        "4. Answer 'is that all?' based on the full get_orders result, not on the last "
        "   message you sent."
    ),
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    )
                }
            ]
        }
    )
    print(result["messages"][-1].content)