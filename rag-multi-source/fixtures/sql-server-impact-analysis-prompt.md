# Prompt: SQL Server impact-analysis test fixture

Hand this prompt to a session with SQL Server access to build a test database
for exercising CorporateRAGPostgres's impact analysis over ingested SQL Server
schema (tables, views, stored procedures, functions).

Context: `app/ingestion/sql_ingestor.py` ingests routines via
`INFORMATION_SCHEMA.ROUTINES` (procs/functions), views via
`INFORMATION_SCHEMA.VIEWS`, table DDL reconstructed from
`INFORMATION_SCHEMA.COLUMNS`, and triggers via `sys.triggers` /
`sys.sql_modules` (triggers aren't exposed through `INFORMATION_SCHEMA` at
all, so that one needed SQL Server's own catalog views). Nothing below
uses `WITH ENCRYPTION`, since that nulls out the definition column for
whichever catalog view is being read and breaks ingestion entirely.

---

```
Create a realistic SQL Server test fixture for exercising impact analysis in a RAG
system that ingests SQL Server schema/objects (tables, views, stored procedures,
functions, and triggers). The goal is a database with enough real dependency
chains that "what breaks if I change X" has a non-trivial, multi-hop answer.

## Domain

A regional retail sales & fulfillment reporting system. Database name:
`RetailReportingDemo`, schema `dbo` throughout.

## Base tables (with PKs/FKs/check constraints as appropriate)

- Regions(RegionID PK, RegionName, Country)
- Warehouses(WarehouseID PK, WarehouseName, RegionID FK)
- ProductCategories(CategoryID PK, CategoryName)
- Products(ProductID PK, ProductName, CategoryID FK, UnitCost, ListPrice,
  SupplierLeadTimeDays)
- Inventory(ProductID FK, WarehouseID FK, QuantityOnHand, ReorderPoint,
  LastRestockDate) — composite PK
- Customers(CustomerID PK, CustomerName, Email, Segment, RegionID FK,
  SignupDate, IsActive)
- Employees(EmployeeID PK, FullName, HireDate, RegionID FK, CommissionTier,
  ManagerID FK self-ref)
- Orders(OrderID PK, CustomerID FK, EmployeeID FK, OrderDate, Status,
  ModifiedDate)
- OrderLines(OrderLineID PK, OrderID FK, ProductID FK, Quantity, UnitPrice,
  DiscountPct)
- Returns(ReturnID PK, OrderLineID FK, ReturnDate, Quantity, Reason)
- Payments(PaymentID PK, OrderID FK, PaymentDate, Amount, Method)
- AuditLog(AuditID PK, TableName, Operation, RecordID, ChangedAt, ChangedBy)
  — write target for triggers only, nothing else writes to it

## Triggers

- trg_Orders_SetModifiedDate — AFTER UPDATE on Orders, sets ModifiedDate
- trg_Orders_Audit — AFTER INSERT/UPDATE/DELETE on Orders → AuditLog
- trg_Customers_Audit — AFTER UPDATE on Customers → AuditLog
- trg_OrderLines_DecrementInventory — AFTER INSERT on OrderLines, decrements
  matching Inventory.QuantityOnHand
- trg_Returns_RestockInventory — AFTER INSERT on Returns, increments
  Inventory.QuantityOnHand back

## Views (layer at least one view on top of another)

- vw_OrderLineDetail — base join: OrderLines + Orders + Products + Customers
  + Employees
- vw_CustomerOrderSummary — built ON TOP OF vw_OrderLineDetail, aggregated
  per customer
- vw_ActiveInventoryStatus — Inventory + Products + Warehouses + Regions,
  flags below-reorder-point rows
- vw_EmployeeRegionMap — Employees + Regions
- vw_ReturnsDetail — Returns + OrderLines + Products

## Shared logic used by MULTIPLE report procedures (this is the important part)

- fn_FiscalPeriod(@date DATE) — scalar function returning a fiscal
  year/period string
- fn_NetLineAmount(@qty, @unitPrice, @discountPct) — scalar function
- usp_StageCompletedOrderLines(@StartDate, @EndDate) — filters
  vw_OrderLineDetail down to Status='Completed' in the date range, writes
  into a persisted staging table dbo.StagingCompletedOrderLines (truncate +
  insert). This one staging procedure must be called by at least THREE of
  the five report procedures below — that shared fan-in is the main thing
  impact analysis should be able to discover.
- usp_LookupCustomerSegment — enrichment step, called by at least two report
  procedures.

## Five report procedures, each populating its OWN physical output table
(not just a result set — INSERT into a real table so it shows up as a
table resource too). Each should visibly move through filter → aggregate →
lookup → calculate → insert stages as separate steps/queries, not one
giant flat query:

1. usp_BuildReport_MonthlySalesByRegion → Report_MonthlySalesByRegion
   filter: usp_StageCompletedOrderLines
   aggregate: SUM(net amount) by Region + fiscal period
   lookup: region names via vw_EmployeeRegionMap/Regions
   calculate: month-over-month growth % (LAG window function)

2. usp_BuildReport_ProductPerformance → Report_ProductPerformance
   filter: usp_StageCompletedOrderLines
   aggregate: qty/revenue by product
   lookup: category + unit cost from Products
   calculate: margin %

3. usp_BuildReport_EmployeeCommission → Report_EmployeeCommission
   filter: usp_StageCompletedOrderLines
   aggregate: sales by employee + fiscal period (fn_FiscalPeriod)
   lookup: CommissionTier from Employees
   calculate: tiered commission amount

4. usp_BuildReport_CustomerChurnRisk → Report_CustomerChurnRisk
   filter: active customers, directly off Orders (not the shared staging —
   this one should be independent)
   aggregate: recency/frequency/monetary (RFM) per customer
   lookup: usp_LookupCustomerSegment
   calculate: churn risk score

5. usp_BuildReport_InventoryReplenishment → Report_InventoryReplenishment
   filter: Inventory where QuantityOnHand < ReorderPoint threshold
   aggregate: recent demand from OrderLines (last 30/60 days)
   lookup: SupplierLeadTimeDays from Products
   calculate: suggested reorder quantity

## Constraints on how objects are written

- No `WITH ENCRYPTION` on any procedure/view/function — the consuming tool
  reads definitions out of INFORMATION_SCHEMA.ROUTINES /
  INFORMATION_SCHEMA.VIEWS, which return NULL for encrypted objects.
- Prefer `CREATE OR ALTER` for procs/views/functions so the scripts are
  re-runnable. Triggers don't support that syntax — guard with
  `IF OBJECT_ID(...) IS NOT NULL DROP TRIGGER ...` before CREATE.
- Organize as numbered, idempotent .sql files: tables → triggers →
  functions → views → staging/shared procs → report procs+tables → seed
  data, so they can be run in order against a fresh instance via sqlcmd.

## Data volume (realistic-looking fake data, not "Test1/Test2")

Regions ~6, Warehouses ~8, ProductCategories ~10, Products ~150,
Customers ~500, Employees ~40, Orders ~8,000 spread over the last 24
months, OrderLines ~20,000 (2-3 per order), Returns ~400 (~5% of order
lines), Payments ~1 per order, Inventory rows for each Product×Warehouse
combination that's stocked (~600 rows). Use realistic names/emails/dates
(e.g. via Faker if generating through Python, or a reasonable T-SQL
name-part table if pure T-SQL). Skew OrderDate/Status/Region distributions
so the five report tables produce non-trivial, differentiated output when
the procs are run — not all identical values.

## Final steps

1. Run all five `usp_BuildReport_*` procedures once so the five
   Report_* tables are populated, not empty.
2. Sanity-check: SELECT TOP 20 from each Report_* table and confirm the
   numbers look plausible (no all-zero/all-null columns).
3. Summarize the final object inventory (table/view/proc/function/trigger
   counts) and confirm no routine/view has a NULL ROUTINE_DEFINITION /
   VIEW_DEFINITION (query INFORMATION_SCHEMA directly to check).
```
