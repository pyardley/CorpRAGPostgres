"""
One-off seed data generator for the RetailReportingDemo SQL Server fixture
(see fixtures/sql-server-impact-analysis-prompt.md). Not part of the app —
run manually against a local SQL Server instance:

    python fixtures/seed_data.py
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import pyodbc
from faker import Faker

CONN_STR = (
    r"DRIVER={ODBC Driver 18 for SQL Server};"
    r"SERVER=localhost\MSSQLSERVER01;"
    r"DATABASE=RetailReportingDemo;"
    r"Trusted_Connection=yes;"
    r"TrustServerCertificate=yes"
)

fake = Faker()
Faker.seed(42)
random.seed(42)

REGIONS = [
    ("Pacific Northwest", "USA"),
    ("Southeast", "USA"),
    ("Midwest", "USA"),
    ("Northeast", "USA"),
    ("Ontario", "Canada"),
    ("Southern England", "UK"),
]

CATEGORIES = [
    "Electronics", "Office Supplies", "Furniture", "Apparel", "Home & Kitchen",
    "Sporting Goods", "Toys & Games", "Health & Beauty", "Automotive Parts",
    "Garden & Outdoor",
]

PRODUCT_NOUNS = {
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Laptop Stand",
                    "USB-C Hub", "Portable Charger", "Noise-Cancelling Headphones",
                    "4K Webcam", "Mechanical Keyboard", "Wireless Mouse"],
    "Office Supplies": ["Stapler", "Desk Organizer", "Notebook Set", "Ballpoint Pen Pack",
                         "Sticky Notes", "Whiteboard", "Filing Cabinet", "Paper Shredder",
                         "Label Maker", "Desk Lamp"],
    "Furniture": ["Office Chair", "Standing Desk", "Bookshelf", "Conference Table", "Sofa",
                  "Coffee Table", "Desk Drawer Unit", "Monitor Arm", "Ergonomic Footrest",
                  "Bar Stool"],
    "Apparel": ["Cotton T-Shirt", "Denim Jacket", "Running Shoes", "Wool Sweater",
                "Baseball Cap", "Rain Jacket", "Yoga Pants", "Polo Shirt", "Winter Gloves",
                "Canvas Backpack"],
    "Home & Kitchen": ["Blender", "Coffee Maker", "Air Fryer", "Cutlery Set", "Non-Stick Pan",
                       "Toaster", "Vacuum Cleaner", "Bedding Set", "Storage Bin",
                       "Cutting Board"],
    "Sporting Goods": ["Yoga Mat", "Dumbbell Set", "Tennis Racket", "Soccer Ball",
                        "Camping Tent", "Hiking Backpack", "Resistance Bands",
                        "Bicycle Helmet", "Water Bottle", "Jump Rope"],
    "Toys & Games": ["Board Game", "Building Blocks", "Puzzle Set", "Remote Control Car",
                      "Action Figure", "Plush Toy", "Card Game", "Model Kit", "Drone",
                      "Art Supply Kit"],
    "Health & Beauty": ["Electric Toothbrush", "Hair Dryer", "Skincare Set", "Vitamin Pack",
                         "Massage Gun", "Shaving Kit", "Face Mask Set", "Body Lotion",
                         "Hair Straightener", "Sunscreen"],
    "Automotive Parts": ["Brake Pads", "Air Filter", "Motor Oil", "Spark Plug Set",
                          "Windshield Wipers", "Car Battery", "Floor Mats",
                          "Tire Pressure Gauge", "Seat Cover", "Dash Cam"],
    "Garden & Outdoor": ["Garden Hose", "Lawn Mower", "Pruning Shears", "Patio Umbrella",
                          "Outdoor Grill", "Planter Box", "Bird Feeder", "Solar Lights",
                          "Wheelbarrow", "Garden Gloves"],
}

BRANDS = ["Nova", "Apex", "Zenith", "Everline", "TrueNorth", "Summit", "Vertex", "Aurora",
          "Pioneer", "Crestline", "Meridian", "Ironclad", "BlueSky", "Northgate"]
BRAND_SUFFIX = ["", "", "", " Pro", " Max", " Lite", " 2.0", " XL"]

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Consumer"]
COMMISSION_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
PAYMENT_METHODS = ["CreditCard", "ACH", "Wire", "Check", "PayPal"]

N_REGIONS = len(REGIONS)
N_WAREHOUSES = 8
N_CATEGORIES = len(CATEGORIES)
N_PRODUCTS = 150
N_CUSTOMERS = 500
N_EMPLOYEES = 40
N_ORDERS = 8000
RETURN_RATE = 0.05

TODAY = date.today()
START_DATE = TODAY - timedelta(days=730)


def insert_chunked(cursor, table, columns, rows, chunk_size=200):
    if not rows:
        return
    col_list = ", ".join(columns)
    row_placeholder = "(" + ", ".join(["?"] * len(columns)) + ")"
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        sql = f"INSERT INTO {table} ({col_list}) VALUES " + ", ".join([row_placeholder] * len(chunk))
        params = [v for row in chunk for v in row]
        cursor.execute(sql, params)


def main():
    conn = pyodbc.connect(CONN_STR, autocommit=False)
    cur = conn.cursor()

    print("Clearing existing data...")
    for tbl in ["AuditLog", "Payments", "Returns", "OrderLines", "Orders", "Inventory",
                "Employees", "Customers", "Products", "ProductCategories", "Warehouses",
                "Regions"]:
        cur.execute(f"DELETE FROM dbo.{tbl}")
    conn.commit()

    # Tables are freshly (re)created by fixtures/sql/01_tables.sql before this
    # script runs, so identity columns are guaranteed to start at 1 — no need
    # to fight DBCC CHECKIDENT's inconsistent "table has never had rows" rule.

    # ---- Regions ----
    print("Seeding Regions...")
    insert_chunked(cur, "dbo.Regions", ["RegionName", "Country"], REGIONS)
    conn.commit()

    # ---- Warehouses ----
    print("Seeding Warehouses...")
    warehouse_rows = [
        (f"{fake.city()} Distribution Center", random.randint(1, N_REGIONS))
        for _ in range(N_WAREHOUSES)
    ]
    insert_chunked(cur, "dbo.Warehouses", ["WarehouseName", "RegionID"], warehouse_rows)
    conn.commit()

    # ---- Product Categories ----
    print("Seeding ProductCategories...")
    insert_chunked(cur, "dbo.ProductCategories", ["CategoryName"], [(c,) for c in CATEGORIES])
    conn.commit()

    # ---- Products ----
    print("Seeding Products...")
    product_rows = []
    product_list_prices = []
    for i in range(N_PRODUCTS):
        category_idx = (i % N_CATEGORIES)
        category_name = CATEGORIES[category_idx]
        noun = random.choice(PRODUCT_NOUNS[category_name])
        brand = random.choice(BRANDS)
        suffix = random.choice(BRAND_SUFFIX)
        product_name = f"{brand} {noun}{suffix}"
        unit_cost = round(random.uniform(5, 300), 2)
        list_price = round(unit_cost * random.uniform(1.4, 2.5), 2)
        lead_time = random.choice([3, 5, 7, 10, 14, 21, 30])
        product_rows.append((product_name, category_idx + 1, unit_cost, list_price, lead_time))
        product_list_prices.append(list_price)
    insert_chunked(
        cur, "dbo.Products",
        ["ProductName", "CategoryID", "UnitCost", "ListPrice", "SupplierLeadTimeDays"],
        product_rows,
    )
    conn.commit()

    # ---- Inventory ----
    print("Seeding Inventory...")
    inventory_rows = []
    for product_id in range(1, N_PRODUCTS + 1):
        n_warehouses_for_product = random.randint(2, 6)
        warehouse_ids = random.sample(range(1, N_WAREHOUSES + 1), n_warehouses_for_product)
        for wh_id in warehouse_ids:
            qty_on_hand = random.randint(0, 500)
            reorder_point = random.randint(20, 150)
            restock_date = fake.date_between(start_date=START_DATE, end_date=TODAY)
            inventory_rows.append((product_id, wh_id, qty_on_hand, reorder_point, restock_date))
    insert_chunked(
        cur, "dbo.Inventory",
        ["ProductID", "WarehouseID", "QuantityOnHand", "ReorderPoint", "LastRestockDate"],
        inventory_rows,
    )
    conn.commit()

    # ---- Customers ----
    print("Seeding Customers...")
    customer_rows = []
    for _ in range(N_CUSTOMERS):
        segment = random.choices(SEGMENTS, weights=[15, 25, 30, 30])[0]
        name = fake.name() if segment == "Consumer" else fake.company()
        email = fake.company_email() if segment != "Consumer" else fake.email()
        region_id = random.randint(1, N_REGIONS)
        signup_date = fake.date_between(start_date=START_DATE - timedelta(days=365), end_date=TODAY)
        is_active = 1 if random.random() > 0.08 else 0
        customer_rows.append((name, email, segment, region_id, signup_date, is_active))
    insert_chunked(
        cur, "dbo.Customers",
        ["CustomerName", "Email", "Segment", "RegionID", "SignupDate", "IsActive"],
        customer_rows,
    )
    conn.commit()

    # ---- Employees ----
    print("Seeding Employees...")
    employee_rows = []
    n_managers = 5
    for i in range(N_EMPLOYEES):
        full_name = fake.name()
        hire_date = fake.date_between(start_date="-8y", end_date="-30d")
        region_id = random.randint(1, N_REGIONS)
        tier = random.choices(COMMISSION_TIERS, weights=[35, 30, 25, 10])[0]
        manager_id = None if i < n_managers else random.randint(1, n_managers)
        employee_rows.append((full_name, hire_date, region_id, tier, manager_id))
    insert_chunked(
        cur, "dbo.Employees",
        ["FullName", "HireDate", "RegionID", "CommissionTier", "ManagerID"],
        employee_rows,
    )
    conn.commit()

    # ---- Orders + OrderLines (built together so line data can reference orders) ----
    print("Generating Orders + OrderLines in memory...")
    statuses = ["Completed", "Shipped", "Pending", "Cancelled"]
    status_weights = [70, 12, 10, 8]

    order_rows = []
    order_dates = []
    for _ in range(N_ORDERS):
        customer_id = random.randint(1, N_CUSTOMERS)
        employee_id = random.randint(1, N_EMPLOYEES)
        order_date = fake.date_between(start_date=START_DATE, end_date=TODAY)
        status = random.choices(statuses, weights=status_weights)[0]
        order_rows.append((customer_id, employee_id, order_date, status))
        order_dates.append(order_date)

    print("Seeding Orders...")
    insert_chunked(cur, "dbo.Orders", ["CustomerID", "EmployeeID", "OrderDate", "Status"], order_rows)
    conn.commit()

    print("Generating OrderLines in memory...")
    order_line_rows = []
    order_line_meta = []  # (order_id, order_date, status, quantity) for downstream Returns/Payments
    for order_id in range(1, N_ORDERS + 1):
        n_lines = random.randint(1, 4)
        for _ in range(n_lines):
            product_id = random.randint(1, N_PRODUCTS)
            list_price = product_list_prices[product_id - 1]
            unit_price = round(list_price * random.uniform(0.95, 1.05), 2)
            quantity = random.randint(1, 8)
            discount_pct = round(random.choice([0, 0, 0, 5, 10, 15, 20]), 2)
            order_line_rows.append((order_id, product_id, quantity, unit_price, discount_pct))
            order_line_meta.append((order_id, order_dates[order_id - 1], quantity, unit_price, discount_pct))

    print(f"Seeding OrderLines ({len(order_line_rows)} rows)...")
    insert_chunked(
        cur, "dbo.OrderLines",
        ["OrderID", "ProductID", "Quantity", "UnitPrice", "DiscountPct"],
        order_line_rows,
        chunk_size=200,
    )
    conn.commit()

    # ---- Returns (~5% of order lines) ----
    print("Seeding Returns...")
    n_lines = len(order_line_meta)
    return_candidates = random.sample(range(1, n_lines + 1), int(n_lines * RETURN_RATE))
    return_rows = []
    for line_id in return_candidates:
        order_id, order_date, quantity, _unit_price, _discount = order_line_meta[line_id - 1]
        return_qty = random.randint(1, quantity)
        min_return_date = order_date + timedelta(days=1)
        if min_return_date >= TODAY:
            continue
        return_date = fake.date_between(start_date=min_return_date, end_date=TODAY)
        reason = random.choice([
            "Defective item", "Wrong size", "No longer needed", "Arrived damaged",
            "Changed mind", "Ordered by mistake",
        ])
        return_rows.append((line_id, return_date, return_qty, reason))
    insert_chunked(
        cur, "dbo.Returns",
        ["OrderLineID", "ReturnDate", "Quantity", "Reason"],
        return_rows,
    )
    conn.commit()

    # ---- Payments (1 per non-cancelled order) ----
    print("Seeding Payments...")
    order_totals: dict[int, float] = {}
    for order_id, _order_date, quantity, unit_price, discount_pct in order_line_meta:
        net = quantity * unit_price * (1 - discount_pct / 100.0)
        order_totals[order_id] = order_totals.get(order_id, 0.0) + net

    payment_rows = []
    for order_id, _employee_id, order_date, status in [
        (i + 1, order_rows[i][1], order_rows[i][2], order_rows[i][3]) for i in range(N_ORDERS)
    ]:
        if status == "Cancelled":
            continue
        payment_date = order_date + timedelta(days=random.randint(0, 5))
        if payment_date > TODAY:
            payment_date = order_date
        amount = round(order_totals.get(order_id, 0.0), 2)
        method = random.choice(PAYMENT_METHODS)
        payment_rows.append((order_id, payment_date, amount, method))
    insert_chunked(
        cur, "dbo.Payments",
        ["OrderID", "PaymentDate", "Amount", "Method"],
        payment_rows,
    )
    conn.commit()

    print("Done seeding.")
    cur.execute("SELECT COUNT(*) FROM dbo.Orders")
    print("Orders:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM dbo.OrderLines")
    print("OrderLines:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM dbo.Returns")
    print("Returns:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM dbo.Payments")
    print("Payments:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM dbo.Inventory")
    print("Inventory:", cur.fetchone()[0])

    conn.close()


if __name__ == "__main__":
    main()
