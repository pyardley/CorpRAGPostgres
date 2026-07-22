USE RetailReportingDemo;
GO

IF OBJECT_ID('dbo.trg_Orders_SetModifiedDate', 'TR') IS NOT NULL DROP TRIGGER dbo.trg_Orders_SetModifiedDate;
GO
CREATE TRIGGER dbo.trg_Orders_SetModifiedDate
ON dbo.Orders
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    IF UPDATE(ModifiedDate) RETURN;

    UPDATE o
    SET o.ModifiedDate = GETDATE()
    FROM dbo.Orders o
    JOIN inserted i ON i.OrderID = o.OrderID;
END;
GO

IF OBJECT_ID('dbo.trg_Orders_Audit', 'TR') IS NOT NULL DROP TRIGGER dbo.trg_Orders_Audit;
GO
CREATE TRIGGER dbo.trg_Orders_Audit
ON dbo.Orders
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @Operation VARCHAR(10);

    IF EXISTS(SELECT 1 FROM inserted) AND EXISTS(SELECT 1 FROM deleted)
        SET @Operation = 'UPDATE';
    ELSE IF EXISTS(SELECT 1 FROM inserted)
        SET @Operation = 'INSERT';
    ELSE
        SET @Operation = 'DELETE';

    INSERT INTO dbo.AuditLog (TableName, Operation, RecordID)
    SELECT 'Orders', @Operation, OrderID FROM inserted
    UNION ALL
    SELECT 'Orders', @Operation, OrderID FROM deleted WHERE @Operation = 'DELETE';
END;
GO

IF OBJECT_ID('dbo.trg_Customers_Audit', 'TR') IS NOT NULL DROP TRIGGER dbo.trg_Customers_Audit;
GO
CREATE TRIGGER dbo.trg_Customers_Audit
ON dbo.Customers
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    INSERT INTO dbo.AuditLog (TableName, Operation, RecordID)
    SELECT 'Customers', 'UPDATE', CustomerID FROM inserted;
END;
GO

IF OBJECT_ID('dbo.trg_OrderLines_DecrementInventory', 'TR') IS NOT NULL DROP TRIGGER dbo.trg_OrderLines_DecrementInventory;
GO
CREATE TRIGGER dbo.trg_OrderLines_DecrementInventory
ON dbo.OrderLines
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;

    ;WITH TargetWarehouse AS (
        SELECT
            i.ProductID,
            i.WarehouseID,
            ins.Quantity,
            ROW_NUMBER() OVER (PARTITION BY ins.OrderLineID ORDER BY i.QuantityOnHand DESC) AS rn
        FROM inserted ins
        JOIN dbo.Orders o ON o.OrderID = ins.OrderID
        JOIN dbo.Customers c ON c.CustomerID = o.CustomerID
        JOIN dbo.Warehouses w ON w.RegionID = c.RegionID
        JOIN dbo.Inventory i ON i.ProductID = ins.ProductID AND i.WarehouseID = w.WarehouseID
    )
    UPDATE inv
    SET inv.QuantityOnHand = CASE
            WHEN inv.QuantityOnHand - tw.Quantity < 0 THEN 0
            ELSE inv.QuantityOnHand - tw.Quantity
        END
    FROM dbo.Inventory inv
    JOIN TargetWarehouse tw
      ON tw.ProductID = inv.ProductID
     AND tw.WarehouseID = inv.WarehouseID
     AND tw.rn = 1;
END;
GO

IF OBJECT_ID('dbo.trg_Returns_RestockInventory', 'TR') IS NOT NULL DROP TRIGGER dbo.trg_Returns_RestockInventory;
GO
CREATE TRIGGER dbo.trg_Returns_RestockInventory
ON dbo.Returns
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;

    ;WITH TargetWarehouse AS (
        SELECT
            i.ProductID,
            i.WarehouseID,
            ins.Quantity,
            ROW_NUMBER() OVER (PARTITION BY ins.ReturnID ORDER BY i.QuantityOnHand ASC) AS rn
        FROM inserted ins
        JOIN dbo.OrderLines ol ON ol.OrderLineID = ins.OrderLineID
        JOIN dbo.Orders o ON o.OrderID = ol.OrderID
        JOIN dbo.Customers c ON c.CustomerID = o.CustomerID
        JOIN dbo.Warehouses w ON w.RegionID = c.RegionID
        JOIN dbo.Inventory i ON i.ProductID = ol.ProductID AND i.WarehouseID = w.WarehouseID
    )
    UPDATE inv
    SET inv.QuantityOnHand = inv.QuantityOnHand + tw.Quantity
    FROM dbo.Inventory inv
    JOIN TargetWarehouse tw
      ON tw.ProductID = inv.ProductID
     AND tw.WarehouseID = inv.WarehouseID
     AND tw.rn = 1;
END;
GO
