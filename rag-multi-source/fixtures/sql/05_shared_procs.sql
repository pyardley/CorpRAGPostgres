USE RetailReportingDemo;
GO

IF OBJECT_ID('dbo.StagingCompletedOrderLines', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.StagingCompletedOrderLines (
        OrderLineID     INT NOT NULL PRIMARY KEY,
        OrderID         INT NOT NULL,
        OrderDate       DATE NOT NULL,
        FiscalPeriod    VARCHAR(10) NOT NULL,
        CustomerID      INT NOT NULL,
        CustomerName    VARCHAR(150) NOT NULL,
        RegionID        INT NOT NULL,
        EmployeeID      INT NOT NULL,
        EmployeeName    VARCHAR(150) NOT NULL,
        CommissionTier  VARCHAR(20) NOT NULL,
        ProductID       INT NOT NULL,
        ProductName     VARCHAR(150) NOT NULL,
        CategoryID      INT NOT NULL,
        UnitCost        DECIMAL(10,2) NOT NULL,
        Quantity        INT NOT NULL,
        UnitPrice       DECIMAL(10,2) NOT NULL,
        DiscountPct     DECIMAL(5,2) NOT NULL,
        NetAmount       DECIMAL(12,2) NOT NULL
    );
END
GO

IF OBJECT_ID('dbo.StagingCustomerSegment', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.StagingCustomerSegment (
        CustomerID      INT NOT NULL PRIMARY KEY,
        CustomerName    VARCHAR(150) NOT NULL,
        Segment         VARCHAR(30) NOT NULL,
        RegionID        INT NOT NULL
    );
END
GO

-- Shared filter stage: narrows the order-line universe down to completed
-- orders in a date range. Used by usp_BuildReport_MonthlySalesByRegion,
-- usp_BuildReport_ProductPerformance, and usp_BuildReport_EmployeeCommission.
CREATE OR ALTER PROCEDURE dbo.usp_StageCompletedOrderLines
    @StartDate DATE,
    @EndDate   DATE
AS
BEGIN
    SET NOCOUNT ON;

    TRUNCATE TABLE dbo.StagingCompletedOrderLines;

    INSERT INTO dbo.StagingCompletedOrderLines (
        OrderLineID, OrderID, OrderDate, FiscalPeriod, CustomerID, CustomerName,
        RegionID, EmployeeID, EmployeeName, CommissionTier, ProductID, ProductName,
        CategoryID, UnitCost, Quantity, UnitPrice, DiscountPct, NetAmount
    )
    SELECT
        OrderLineID, OrderID, OrderDate, dbo.fn_FiscalPeriod(OrderDate), CustomerID, CustomerName,
        RegionID, EmployeeID, EmployeeName, CommissionTier, ProductID, ProductName,
        CategoryID, UnitCost, Quantity, UnitPrice, DiscountPct, NetAmount
    FROM dbo.vw_OrderLineDetail
    WHERE Status = 'Completed'
      AND OrderDate BETWEEN @StartDate AND @EndDate;
END;
GO

-- Shared lookup/enrichment stage: refreshes a denormalized customer ->
-- segment/region snapshot. Used by usp_BuildReport_CustomerChurnRisk and
-- usp_BuildReport_MonthlySalesByRegion.
CREATE OR ALTER PROCEDURE dbo.usp_LookupCustomerSegment
AS
BEGIN
    SET NOCOUNT ON;

    TRUNCATE TABLE dbo.StagingCustomerSegment;

    INSERT INTO dbo.StagingCustomerSegment (CustomerID, CustomerName, Segment, RegionID)
    SELECT CustomerID, CustomerName, Segment, RegionID
    FROM dbo.Customers;
END;
GO
