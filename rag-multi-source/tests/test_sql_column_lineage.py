"""
Unit tests for core.sql_column_lineage.column_lineage_findings.

sqlglot-only capability -- no legacy engine to mirror. Uses literal
procedure bodies from fixtures/sql/06_reports.sql, same convention as
tests/test_sql_dependency_extraction.py and tests/test_sql_join_shape.py.
Built up in risk order: a simple 2-hop case first (sanity check), then
the harder 3-stage-plus-CTE CustomerChurnRisk case, then gating.
"""

from core.retriever import RetrievedChunk
from core.sql_column_lineage import column_lineage_findings

_TRACING_QUESTION = (
    "Show how Report_CustomerChurnRisk.TotalNetAmount is derived. Go "
    "right back to original source tables."
)
_NON_TRACING_QUESTION = "What does this procedure do?"

# Literal body from fixtures/sql/06_reports.sql (lines 112-159) -- a
# simple single-temp-table-stage procedure, no CTE.
_PRODUCT_PERFORMANCE_PROC = """
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
"""

# Literal body from fixtures/sql/06_reports.sql (lines 256-340).
_CUSTOMER_CHURN_RISK_PROC = """
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
"""


def _chunk(resource_id, object_name, object_type, text, score=1.0):
    return RetrievedChunk(
        resource_id=resource_id,
        source="sql",
        chunk_index=0,
        score=score,
        text=text,
        title=object_name,
        url="",
        metadata={"object_name": object_name, "object_type": object_type},
    )


def test_simple_two_hop_case():
    """Sanity check on a single-temp-table-stage procedure before the
    harder multi-stage case: dbo.Report_ProductPerformance.TotalRevenue
    <- #ProductAgg.TotalRevenue (SUM(NetAmount)) <-
    dbo.StagingCompletedOrderLines.NetAmount."""
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_ProductPerformance",
            "dbo.usp_BuildReport_ProductPerformance",
            "procedure",
            _PRODUCT_PERFORMANCE_PROC,
            score=0.9,
        )
    ]
    question = "Show how TotalRevenue is derived. Go right back to original source tables."
    findings = column_lineage_findings(question, hits)
    assert findings == [
        "`dbo.Report_ProductPerformance.TotalRevenue` ← "
        "`#Enriched.TotalRevenue` ← "
        "`#ProductAgg.TotalRevenue` (SUM(NetAmount)) ← "
        "`dbo.StagingCompletedOrderLines.NetAmount`"
    ]


def test_customer_churn_risk_full_chain_through_cte_and_three_stages():
    """The hard case: chains through the CTE (Scored), #Enriched, #RFM,
    and #ActiveCustomerOrders, ending at the real base-table columns
    behind the fn_NetLineAmount(...) call -- exact-string assertion since
    this is the flagship capability this whole engine exists to add."""
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
            "dbo.usp_BuildReport_CustomerChurnRisk",
            "procedure",
            _CUSTOMER_CHURN_RISK_PROC,
            score=0.9,
        )
    ]
    findings = column_lineage_findings(_TRACING_QUESTION, hits)
    assert findings == [
        "`dbo.Report_CustomerChurnRisk.TotalNetAmount` ← "
        "`Scored.TotalNetAmount` ← "
        "`#Enriched.TotalNetAmount` (ISNULL(r.TotalNetAmount, 0)) ← "
        "`#RFM.TotalNetAmount` (SUM(NetAmount)) ← "
        "`#ActiveCustomerOrders.NetAmount` "
        "(dbo.fn_NetLineAmount(ol.Quantity, ol.UnitPrice, ol.DiscountPct)) ← "
        "`dbo.OrderLines.DiscountPct`, `dbo.OrderLines.Quantity`, "
        "`dbo.OrderLines.UnitPrice`"
    ]


def test_falls_back_to_first_output_columns_when_question_names_none():
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
            "dbo.usp_BuildReport_CustomerChurnRisk",
            "procedure",
            _CUSTOMER_CHURN_RISK_PROC,
            score=0.9,
        )
    ]
    question = "Trace this report back to its source tables."
    findings = column_lineage_findings(question, hits)
    assert len(findings) == 3
    assert findings[0].startswith("`dbo.Report_CustomerChurnRisk.CustomerID` ←")
    assert findings[1].startswith("`dbo.Report_CustomerChurnRisk.CustomerName` ←")
    assert findings[2].startswith("`dbo.Report_CustomerChurnRisk.Segment` ←")


def test_gated_on_tracing_question():
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
            "dbo.usp_BuildReport_CustomerChurnRisk",
            "procedure",
            _CUSTOMER_CHURN_RISK_PROC,
            score=0.9,
        )
    ]
    assert column_lineage_findings(_NON_TRACING_QUESTION, hits) == []


def test_no_anchor_or_empty_hits():
    assert column_lineage_findings(_TRACING_QUESTION, []) == []


def test_unparseable_text_is_silent_not_a_crash():
    hits = [
        _chunk(
            "sql:x.dbo.usp_Garbage",
            "dbo.usp_Garbage",
            "procedure",
            "this is not valid T-SQL at all {{{ ] ) (",
            score=0.9,
        )
    ]
    assert column_lineage_findings(_TRACING_QUESTION, hits) == []


def test_no_temp_table_stages_is_silent():
    """A trivial procedure with no INTO stages at all -- e.g. a bare
    passthrough SELECT with no derivation to trace -- should produce no
    findings rather than an empty-chain artifact."""
    text = """
CREATE OR ALTER PROCEDURE dbo.usp_Trivial
AS
BEGIN
    SELECT 1 AS X;
END;
"""
    hits = [_chunk("sql:x.dbo.usp_Trivial", "dbo.usp_Trivial", "procedure", text, 0.9)]
    assert column_lineage_findings(_TRACING_QUESTION, hits) == []
