USE RetailReportingDemo;
GO

-- ============================================================
-- Report 1: Monthly Sales By Region
-- ============================================================
IF OBJECT_ID('dbo.Report_MonthlySalesByRegion', 'U') IS NULL
CREATE TABLE dbo.Report_MonthlySalesByRegion (
    RegionID            INT NOT NULL,
    RegionName          VARCHAR(60) NOT NULL,
    FiscalPeriod        VARCHAR(10) NOT NULL,
    TotalNetAmount      DECIMAL(14,2) NOT NULL,
    PriorPeriodAmount   DECIMAL(14,2) NULL,
    GrowthPct           DECIMAL(7,2) NULL,
    TopSegment          VARCHAR(30) NULL,
    GeneratedAt         DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT PK_Report_MonthlySalesByRegion PRIMARY KEY (RegionID, FiscalPeriod)
);
GO

CREATE OR ALTER PROCEDURE dbo.usp_BuildReport_MonthlySalesByRegion
    @StartDate DATE = NULL,
    @EndDate   DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    IF @StartDate IS NULL SET @StartDate = DATEADD(MONTH, -24, CAST(GETDATE() AS DATE));
    IF @EndDate IS NULL SET @EndDate = CAST(GETDATE() AS DATE);

    -- Filter stage (shared with ProductPerformance / EmployeeCommission)
    EXEC dbo.usp_StageCompletedOrderLines @StartDate = @StartDate, @EndDate = @EndDate;

    -- Lookup stage (shared with CustomerChurnRisk): refresh customer segment snapshot
    EXEC dbo.usp_LookupCustomerSegment;

    -- Aggregate stage: revenue per region/period
    IF OBJECT_ID('tempdb..#RegionPeriodAgg') IS NOT NULL DROP TABLE #RegionPeriodAgg;
    SELECT
        RegionID,
        FiscalPeriod,
        MIN(OrderDate)  AS PeriodStart,
        SUM(NetAmount)  AS TotalNetAmount
    INTO #RegionPeriodAgg
    FROM dbo.StagingCompletedOrderLines
    GROUP BY RegionID, FiscalPeriod;

    -- Aggregate stage: revenue per region/period/segment, ranked to find the top segment
    IF OBJECT_ID('tempdb..#RegionSegmentAgg') IS NOT NULL DROP TABLE #RegionSegmentAgg;
    SELECT
        s.RegionID,
        x.FiscalPeriod,
        s.Segment,
        SUM(x.NetAmount) AS SegmentAmount,
        ROW_NUMBER() OVER (PARTITION BY s.RegionID, x.FiscalPeriod ORDER BY SUM(x.NetAmount) DESC) AS rn
    INTO #RegionSegmentAgg
    FROM dbo.StagingCompletedOrderLines x
    JOIN dbo.StagingCustomerSegment s ON s.CustomerID = x.CustomerID
    GROUP BY s.RegionID, x.FiscalPeriod, s.Segment;

    -- Lookup stage: region names + top segment + prior-period amount
    IF OBJECT_ID('tempdb..#Enriched') IS NOT NULL DROP TABLE #Enriched;
    SELECT
        a.RegionID,
        r.RegionName,
        a.FiscalPeriod,
        a.TotalNetAmount,
        LAG(a.TotalNetAmount) OVER (PARTITION BY a.RegionID ORDER BY a.PeriodStart) AS PriorPeriodAmount,
        seg.Segment AS TopSegment
    INTO #Enriched
    FROM #RegionPeriodAgg a
    JOIN dbo.Regions r ON r.RegionID = a.RegionID
    LEFT JOIN #RegionSegmentAgg seg
      ON seg.RegionID = a.RegionID AND seg.FiscalPeriod = a.FiscalPeriod AND seg.rn = 1;

    -- Calculate stage: month-over-month growth %
    TRUNCATE TABLE dbo.Report_MonthlySalesByRegion;
    INSERT INTO dbo.Report_MonthlySalesByRegion (
        RegionID, RegionName, FiscalPeriod, TotalNetAmount, PriorPeriodAmount, GrowthPct, TopSegment
    )
    SELECT
        RegionID,
        RegionName,
        FiscalPeriod,
        TotalNetAmount,
        PriorPeriodAmount,
        CASE
            WHEN PriorPeriodAmount IS NULL OR PriorPeriodAmount = 0 THEN NULL
            ELSE ROUND((TotalNetAmount - PriorPeriodAmount) / PriorPeriodAmount * 100, 2)
        END AS GrowthPct,
        TopSegment
    FROM #Enriched;
END;
GO

-- ============================================================
-- Report 2: Product Performance
-- ============================================================
IF OBJECT_ID('dbo.Report_ProductPerformance', 'U') IS NULL
CREATE TABLE dbo.Report_ProductPerformance (
    ProductID       INT NOT NULL PRIMARY KEY,
    ProductName     VARCHAR(150) NOT NULL,
    CategoryID      INT NOT NULL,
    CategoryName    VARCHAR(60) NOT NULL,
    TotalQuantity   INT NOT NULL,
    TotalRevenue    DECIMAL(14,2) NOT NULL,
    TotalCost       DECIMAL(14,2) NOT NULL,
    MarginPct       DECIMAL(7,2) NULL,
    GeneratedAt     DATETIME NOT NULL DEFAULT GETDATE()
);
GO

CREATE OR ALTER PROCEDURE dbo.usp_BuildReport_ProductPerformance
    @StartDate DATE = NULL,
    @EndDate   DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    IF @StartDate IS NULL SET @StartDate = DATEADD(MONTH, -24, CAST(GETDATE() AS DATE));
    IF @EndDate IS NULL SET @EndDate = CAST(GETDATE() AS DATE);

    -- Filter stage (shared with MonthlySalesByRegion / EmployeeCommission)
    EXEC dbo.usp_StageCompletedOrderLines @StartDate = @StartDate, @EndDate = @EndDate;

    -- Aggregate stage: quantity/revenue/cost per product
    IF OBJECT_ID('tempdb..#ProductAgg') IS NOT NULL DROP TABLE #ProductAgg;
    SELECT
        ProductID,
        SUM(Quantity)           AS TotalQuantity,
        SUM(NetAmount)          AS TotalRevenue,
        SUM(Quantity * UnitCost) AS TotalCost
    INTO #ProductAgg
    FROM dbo.StagingCompletedOrderLines
    GROUP BY ProductID;

    -- Lookup stage: category name + product name
    IF OBJECT_ID('tempdb..#Enriched') IS NOT NULL DROP TABLE #Enriched;
    SELECT
        a.ProductID,
        p.ProductName,
        p.CategoryID,
        cat.CategoryName,
        a.TotalQuantity,
        a.TotalRevenue,
        a.TotalCost
    INTO #Enriched
    FROM #ProductAgg a
    JOIN dbo.Products p ON p.ProductID = a.ProductID
    JOIN dbo.ProductCategories cat ON cat.CategoryID = p.CategoryID;

    -- Calculate stage: margin %
    TRUNCATE TABLE dbo.Report_ProductPerformance;
    INSERT INTO dbo.Report_ProductPerformance (
        ProductID, ProductName, CategoryID, CategoryName, TotalQuantity, TotalRevenue, TotalCost, MarginPct
    )
    SELECT
        ProductID, ProductName, CategoryID, CategoryName, TotalQuantity, TotalRevenue, TotalCost,
        CASE WHEN TotalRevenue = 0 THEN NULL ELSE ROUND((TotalRevenue - TotalCost) / TotalRevenue * 100, 2) END
    FROM #Enriched;
END;
GO

-- ============================================================
-- Report 3: Employee Commission
-- ============================================================
IF OBJECT_ID('dbo.Report_EmployeeCommission', 'U') IS NULL
CREATE TABLE dbo.Report_EmployeeCommission (
    EmployeeID          INT NOT NULL,
    EmployeeName        VARCHAR(150) NOT NULL,
    FiscalPeriod        VARCHAR(10) NOT NULL,
    CommissionTier      VARCHAR(20) NOT NULL,
    TotalSales          DECIMAL(14,2) NOT NULL,
    CommissionRate      DECIMAL(5,2) NOT NULL,
    CommissionAmount    DECIMAL(14,2) NOT NULL,
    GeneratedAt         DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT PK_Report_EmployeeCommission PRIMARY KEY (EmployeeID, FiscalPeriod)
);
GO

CREATE OR ALTER PROCEDURE dbo.usp_BuildReport_EmployeeCommission
    @StartDate DATE = NULL,
    @EndDate   DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    IF @StartDate IS NULL SET @StartDate = DATEADD(MONTH, -24, CAST(GETDATE() AS DATE));
    IF @EndDate IS NULL SET @EndDate = CAST(GETDATE() AS DATE);

    -- Filter stage (shared with MonthlySalesByRegion / ProductPerformance)
    EXEC dbo.usp_StageCompletedOrderLines @StartDate = @StartDate, @EndDate = @EndDate;

    -- Aggregate stage: sales per employee/period
    IF OBJECT_ID('tempdb..#EmployeeAgg') IS NOT NULL DROP TABLE #EmployeeAgg;
    SELECT
        EmployeeID,
        FiscalPeriod,
        SUM(NetAmount) AS TotalSales
    INTO #EmployeeAgg
    FROM dbo.StagingCompletedOrderLines
    GROUP BY EmployeeID, FiscalPeriod;

    -- Lookup stage: current commission tier, looked up fresh from Employees
    -- rather than trusting the staged snapshot (tiers can change after staging).
    IF OBJECT_ID('tempdb..#Enriched') IS NOT NULL DROP TABLE #Enriched;
    SELECT
        a.EmployeeID,
        e.FullName AS EmployeeName,
        a.FiscalPeriod,
        e.CommissionTier,
        a.TotalSales
    INTO #Enriched
    FROM #EmployeeAgg a
    JOIN dbo.Employees e ON e.EmployeeID = a.EmployeeID;

    -- Calculate stage: tiered commission rate/amount
    TRUNCATE TABLE dbo.Report_EmployeeCommission;
    INSERT INTO dbo.Report_EmployeeCommission (
        EmployeeID, EmployeeName, FiscalPeriod, CommissionTier, TotalSales, CommissionRate, CommissionAmount
    )
    SELECT
        EmployeeID, EmployeeName, FiscalPeriod, CommissionTier, TotalSales,
        CASE CommissionTier
            WHEN 'Platinum' THEN 8.00
            WHEN 'Gold'     THEN 6.00
            WHEN 'Silver'   THEN 4.00
            ELSE 2.50
        END AS CommissionRate,
        TotalSales * CASE CommissionTier
            WHEN 'Platinum' THEN 0.08
            WHEN 'Gold'     THEN 0.06
            WHEN 'Silver'   THEN 0.04
            ELSE 0.025
        END AS CommissionAmount
    FROM #Enriched;
END;
GO

-- ============================================================
-- Report 4: Customer Churn Risk
-- ============================================================
IF OBJECT_ID('dbo.Report_CustomerChurnRisk', 'U') IS NULL
CREATE TABLE dbo.Report_CustomerChurnRisk (
    CustomerID          INT NOT NULL PRIMARY KEY,
    CustomerName        VARCHAR(150) NOT NULL,
    Segment             VARCHAR(30) NOT NULL,
    RegionID            INT NOT NULL,
    LastOrderDate       DATE NULL,
    DaysSinceLastOrder  INT NULL,
    TotalOrders         INT NOT NULL,
    TotalNetAmount      DECIMAL(14,2) NOT NULL,
    ChurnRiskScore      DECIMAL(6,2) NOT NULL,
    RiskBand            VARCHAR(10) NOT NULL,
    GeneratedAt         DATETIME NOT NULL DEFAULT GETDATE()
);
GO

CREATE OR ALTER PROCEDURE dbo.usp_BuildReport_CustomerChurnRisk
AS
BEGIN
    SET NOCOUNT ON;

    -- Filter stage: active customers, queried directly off Orders/OrderLines
    -- (deliberately independent of the shared completed-order-lines staging table)
    IF OBJECT_ID('tempdb..#ActiveCustomerOrders') IS NOT NULL DROP TABLE #ActiveCustomerOrders;
    SELECT
        c.CustomerID,
        o.OrderID,
        o.OrderDate,
        dbo.fn_NetLineAmount(ol.Quantity, ol.UnitPrice, ol.DiscountPct) AS NetAmount
    INTO #ActiveCustomerOrders
    FROM dbo.Customers c
    JOIN dbo.Orders o ON o.CustomerID = c.CustomerID AND o.Status = 'Completed'
    JOIN dbo.OrderLines ol ON ol.OrderID = o.OrderID
    WHERE c.IsActive = 1;

    -- Aggregate stage: recency/frequency/monetary
    IF OBJECT_ID('tempdb..#RFM') IS NOT NULL DROP TABLE #RFM;
    SELECT
        CustomerID,
        MAX(OrderDate)              AS LastOrderDate,
        COUNT(DISTINCT OrderID)     AS TotalOrders,
        SUM(NetAmount)              AS TotalNetAmount
    INTO #RFM
    FROM #ActiveCustomerOrders
    GROUP BY CustomerID;

    -- Lookup stage (shared with MonthlySalesByRegion): customer segment/region snapshot
    EXEC dbo.usp_LookupCustomerSegment;

    IF OBJECT_ID('tempdb..#Enriched') IS NOT NULL DROP TABLE #Enriched;
    SELECT
        s.CustomerID,
        s.CustomerName,
        s.Segment,
        s.RegionID,
        r.LastOrderDate,
        DATEDIFF(DAY, r.LastOrderDate, GETDATE())  AS DaysSinceLastOrder,
        ISNULL(r.TotalOrders, 0)                   AS TotalOrders,
        ISNULL(r.TotalNetAmount, 0)                AS TotalNetAmount
    INTO #Enriched
    FROM dbo.StagingCustomerSegment s
    LEFT JOIN #RFM r ON r.CustomerID = s.CustomerID
    WHERE EXISTS (SELECT 1 FROM dbo.Customers c WHERE c.CustomerID = s.CustomerID AND c.IsActive = 1);

    -- Calculate stage: churn risk score (recency-weighted + frequency-weighted)
    TRUNCATE TABLE dbo.Report_CustomerChurnRisk;
    ;WITH Scored AS (
        SELECT
            CustomerID, CustomerName, Segment, RegionID, LastOrderDate, DaysSinceLastOrder, TotalOrders, TotalNetAmount,
            (
                (CASE
                    WHEN DaysSinceLastOrder IS NULL THEN 100
                    WHEN DaysSinceLastOrder > 365 THEN 100
                    WHEN DaysSinceLastOrder > 180 THEN 70
                    WHEN DaysSinceLastOrder > 90  THEN 40
                    ELSE 10
                 END) * 0.6
                +
                (CASE
                    WHEN TotalOrders = 0     THEN 100
                    WHEN TotalOrders <= 2    THEN 60
                    WHEN TotalOrders <= 5    THEN 30
                    ELSE 5
                 END) * 0.4
            ) AS ChurnRiskScore
        FROM #Enriched
    )
    INSERT INTO dbo.Report_CustomerChurnRisk (
        CustomerID, CustomerName, Segment, RegionID, LastOrderDate, DaysSinceLastOrder,
        TotalOrders, TotalNetAmount, ChurnRiskScore, RiskBand
    )
    SELECT
        CustomerID, CustomerName, Segment, RegionID, LastOrderDate, DaysSinceLastOrder,
        TotalOrders, TotalNetAmount, ROUND(ChurnRiskScore, 2),
        CASE
            WHEN ChurnRiskScore >= 70 THEN 'High'
            WHEN ChurnRiskScore >= 40 THEN 'Medium'
            ELSE 'Low'
        END
    FROM Scored;
END;
GO

-- ============================================================
-- Report 5: Inventory Replenishment
-- ============================================================
IF OBJECT_ID('dbo.Report_InventoryReplenishment', 'U') IS NULL
CREATE TABLE dbo.Report_InventoryReplenishment (
    ProductID               INT NOT NULL,
    ProductName             VARCHAR(150) NOT NULL,
    WarehouseID             INT NOT NULL,
    WarehouseName           VARCHAR(100) NOT NULL,
    QuantityOnHand          INT NOT NULL,
    ReorderPoint            INT NOT NULL,
    RecentDemand            INT NOT NULL,
    SupplierLeadTimeDays    INT NOT NULL,
    SuggestedReorderQty     INT NOT NULL,
    GeneratedAt             DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT PK_Report_InventoryReplenishment PRIMARY KEY (ProductID, WarehouseID)
);
GO

CREATE OR ALTER PROCEDURE dbo.usp_BuildReport_InventoryReplenishment
AS
BEGIN
    SET NOCOUNT ON;

    -- Filter stage: inventory at or below its reorder point
    IF OBJECT_ID('tempdb..#LowStock') IS NOT NULL DROP TABLE #LowStock;
    SELECT ProductID, WarehouseID, QuantityOnHand, ReorderPoint
    INTO #LowStock
    FROM dbo.Inventory
    WHERE QuantityOnHand <= ReorderPoint;

    -- Aggregate stage: recent demand (last 60 days) by product
    IF OBJECT_ID('tempdb..#RecentDemand') IS NOT NULL DROP TABLE #RecentDemand;
    SELECT
        ol.ProductID,
        SUM(ol.Quantity) AS RecentDemand
    INTO #RecentDemand
    FROM dbo.OrderLines ol
    JOIN dbo.Orders o ON o.OrderID = ol.OrderID
    WHERE o.OrderDate >= DATEADD(DAY, -60, CAST(GETDATE() AS DATE))
    GROUP BY ol.ProductID;

    -- Lookup stage: product name/lead time, warehouse name
    IF OBJECT_ID('tempdb..#Enriched') IS NOT NULL DROP TABLE #Enriched;
    SELECT
        ls.ProductID,
        p.ProductName,
        ls.WarehouseID,
        w.WarehouseName,
        ls.QuantityOnHand,
        ls.ReorderPoint,
        ISNULL(rd.RecentDemand, 0) AS RecentDemand,
        p.SupplierLeadTimeDays
    INTO #Enriched
    FROM #LowStock ls
    JOIN dbo.Products p ON p.ProductID = ls.ProductID
    JOIN dbo.Warehouses w ON w.WarehouseID = ls.WarehouseID
    LEFT JOIN #RecentDemand rd ON rd.ProductID = ls.ProductID;

    -- Calculate stage: suggested reorder qty (projected lead-time demand + reorder buffer - on hand)
    TRUNCATE TABLE dbo.Report_InventoryReplenishment;
    INSERT INTO dbo.Report_InventoryReplenishment (
        ProductID, ProductName, WarehouseID, WarehouseName, QuantityOnHand, ReorderPoint,
        RecentDemand, SupplierLeadTimeDays, SuggestedReorderQty
    )
    SELECT
        ProductID, ProductName, WarehouseID, WarehouseName, QuantityOnHand, ReorderPoint,
        RecentDemand, SupplierLeadTimeDays,
        CASE
            WHEN (CEILING((RecentDemand / 60.0) * SupplierLeadTimeDays) + ReorderPoint - QuantityOnHand) < 0 THEN 0
            ELSE CAST(CEILING((RecentDemand / 60.0) * SupplierLeadTimeDays) + ReorderPoint - QuantityOnHand AS INT)
        END AS SuggestedReorderQty
    FROM #Enriched;
END;
GO
