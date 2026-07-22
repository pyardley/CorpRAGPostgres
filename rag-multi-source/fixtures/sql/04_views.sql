USE RetailReportingDemo;
GO

CREATE OR ALTER VIEW dbo.vw_OrderLineDetail
AS
SELECT
    ol.OrderLineID,
    o.OrderID,
    o.OrderDate,
    o.Status,
    c.CustomerID,
    c.CustomerName,
    c.Segment,
    c.RegionID,
    e.EmployeeID,
    e.FullName        AS EmployeeName,
    e.CommissionTier,
    p.ProductID,
    p.ProductName,
    p.CategoryID,
    p.UnitCost,
    ol.Quantity,
    ol.UnitPrice,
    ol.DiscountPct,
    dbo.fn_NetLineAmount(ol.Quantity, ol.UnitPrice, ol.DiscountPct) AS NetAmount
FROM dbo.OrderLines ol
JOIN dbo.Orders o     ON o.OrderID = ol.OrderID
JOIN dbo.Products p   ON p.ProductID = ol.ProductID
JOIN dbo.Customers c  ON c.CustomerID = o.CustomerID
JOIN dbo.Employees e  ON e.EmployeeID = o.EmployeeID;
GO

CREATE OR ALTER VIEW dbo.vw_CustomerOrderSummary
AS
SELECT
    CustomerID,
    CustomerName,
    Segment,
    COUNT(DISTINCT OrderID)        AS TotalOrders,
    SUM(NetAmount)                 AS TotalNetAmount,
    MIN(OrderDate)                 AS FirstOrderDate,
    MAX(OrderDate)                 AS LastOrderDate
FROM dbo.vw_OrderLineDetail
GROUP BY CustomerID, CustomerName, Segment;
GO

CREATE OR ALTER VIEW dbo.vw_ActiveInventoryStatus
AS
SELECT
    i.ProductID,
    p.ProductName,
    i.WarehouseID,
    w.WarehouseName,
    w.RegionID,
    r.RegionName,
    i.QuantityOnHand,
    i.ReorderPoint,
    i.LastRestockDate,
    CASE WHEN i.QuantityOnHand < i.ReorderPoint THEN 1 ELSE 0 END AS IsBelowReorderPoint
FROM dbo.Inventory i
JOIN dbo.Products p    ON p.ProductID = i.ProductID
JOIN dbo.Warehouses w  ON w.WarehouseID = i.WarehouseID
JOIN dbo.Regions r     ON r.RegionID = w.RegionID;
GO

CREATE OR ALTER VIEW dbo.vw_EmployeeRegionMap
AS
SELECT
    e.EmployeeID,
    e.FullName,
    e.CommissionTier,
    e.HireDate,
    e.RegionID,
    r.RegionName,
    r.Country
FROM dbo.Employees e
JOIN dbo.Regions r ON r.RegionID = e.RegionID;
GO

CREATE OR ALTER VIEW dbo.vw_ReturnsDetail
AS
SELECT
    rt.ReturnID,
    rt.ReturnDate,
    rt.Quantity      AS ReturnedQuantity,
    rt.Reason,
    ol.OrderLineID,
    ol.OrderID,
    p.ProductID,
    p.ProductName,
    p.CategoryID
FROM dbo.Returns rt
JOIN dbo.OrderLines ol ON ol.OrderLineID = rt.OrderLineID
JOIN dbo.Products p    ON p.ProductID = ol.ProductID;
GO
