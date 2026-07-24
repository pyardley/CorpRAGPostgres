"""
Unit tests for core.sql_join_shape.join_shape_findings.

Uses literal procedure bodies from the RetailReportingDemo fixture
(fixtures/sql/06_reports.sql) — CustomerChurnRisk, InventoryReplenishment,
and MonthlySalesByRegion — the same "ground truth against real fixture
text" convention test_sql_dependency_extraction.py uses, so wording/
parsing regressions are caught without a live SQL Server connection.
"""

from core.retriever import RetrievedChunk
from core.sql_join_shape import join_shape_findings

_TRACING_QUESTION = (
    "Show how Report_CustomerChurnRisk.TotalNetAmount is derived. Go "
    "right back to original source tables."
)
_NON_TRACING_QUESTION = "What does this procedure do?"

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

# The procedure's own trailing CTE, isolated (from the ';WITH Scored' tail
# onward) — used by the CTE-skip test to confirm it produces no finding in
# isolation, as a baseline before test (b) below proves the skip is what's
# actually suppressing it (see that test's docstring).
_CTE_TAIL_ONLY = _CUSTOMER_CHURN_RISK_PROC[_CUSTOMER_CHURN_RISK_PROC.index(";WITH Scored") :]

# Literal body from fixtures/sql/06_reports.sql (lines 21-92).
_MONTHLY_SALES_BY_REGION_PROC = """
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
"""

# Literal body from fixtures/sql/06_reports.sql (lines 362-416).
_INVENTORY_REPLENISHMENT_PROC = """
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


def _churn_risk_hit(score=0.95):
    return _chunk(
        "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
        "dbo.usp_BuildReport_CustomerChurnRisk",
        "procedure",
        _CUSTOMER_CHURN_RISK_PROC,
        score=score,
    )


def test_flagship_left_join_finding_exact_wording():
    """Exact-string assertion (not just presence) — catches wording
    regressions, matching test_sql_dependency_extraction.py's convention.
    No siblings: candidates come entirely from the anchor's own temp-table
    tokens (#ActiveCustomerOrders, #RFM, #Enriched), so only the #Enriched
    chain (driving dbo.StagingCustomerSegment, LEFT-joined to #RFM) is
    eligible — the #ActiveCustomerOrders chain's tables (dbo.Customers/
    dbo.Orders/dbo.OrderLines) never became candidates without siblings."""
    hits = [_churn_risk_hit()]
    findings = join_shape_findings(_TRACING_QUESTION, hits)
    assert findings == [
        "`dbo.StagingCustomerSegment` LEFT-joins `#RFM` when building "
        "`#Enriched` — every `dbo.StagingCustomerSegment` row is kept "
        "even without a match, but a `#RFM` row with no matching key in "
        "`dbo.StagingCustomerSegment` is silently excluded from the result"
    ]


def test_inner_join_category_with_sibling_candidates():
    """Sibling hits for dbo.Customers/dbo.Orders/dbo.OrderLines (each
    independently retrieved, each name present in the anchor's own text)
    promote them to candidates, making the #ActiveCustomerOrders 2-JOIN
    chain eligible too."""
    hits = [
        _churn_risk_hit(),
        _chunk("sql:x.dbo.Customers", "dbo.Customers", "table", "CREATE TABLE dbo.Customers (...)", 0.5),
        _chunk("sql:x.dbo.Orders", "dbo.Orders", "table", "CREATE TABLE dbo.Orders (...)", 0.4),
        _chunk("sql:x.dbo.OrderLines", "dbo.OrderLines", "table", "CREATE TABLE dbo.OrderLines (...)", 0.3),
    ]
    findings = join_shape_findings(_TRACING_QUESTION, hits)
    assert (
        "`dbo.Orders` is INNER-joined to `dbo.Customers` when building "
        "`#ActiveCustomerOrders` — a row in either table with no matching "
        "join key is silently excluded (dropped, not nulled) from what "
        "continues downstream"
    ) in findings
    assert (
        "`dbo.OrderLines` is INNER-joined to `dbo.Orders` when building "
        "`#ActiveCustomerOrders` — a row in either table with no matching "
        "join key is silently excluded (dropped, not nulled) from what "
        "continues downstream"
    ) in findings


def test_mixed_join_types_in_one_chain_correctly_attributed():
    """usp_BuildReport_InventoryReplenishment's #Enriched build: 2 INNER
    JOINs + 1 LEFT JOIN in a single chain. Both categories must appear,
    each naming the correct pair of tables."""
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_InventoryReplenishment",
            "dbo.usp_BuildReport_InventoryReplenishment",
            "procedure",
            _INVENTORY_REPLENISHMENT_PROC,
            score=0.9,
        )
    ]
    findings = join_shape_findings(_TRACING_QUESTION, hits)
    assert any(
        "`dbo.Products` is INNER-joined to `#LowStock` when building `#Enriched`" in f
        for f in findings
    )
    assert any("`dbo.Warehouses` is INNER-joined to `dbo.Products`" in f for f in findings)
    assert any(
        "`dbo.Warehouses` LEFT-joins `#RecentDemand` when building `#Enriched`" in f
        for f in findings
    )


def test_window_functions_do_not_break_or_spuriously_trigger():
    """usp_BuildReport_MonthlySalesByRegion's #Enriched build mixes a
    ROW_NUMBER()/LAG() OVER (PARTITION BY ...) window function into the
    same statement as a real INNER+LEFT join chain. The real LEFT finding
    must still fire, and nothing should hallucinate a window-function
    column (e.g. PriorPeriodAmount, rn) as a table name."""
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_MonthlySalesByRegion",
            "dbo.usp_BuildReport_MonthlySalesByRegion",
            "procedure",
            _MONTHLY_SALES_BY_REGION_PROC,
            score=0.9,
        )
    ]
    findings = join_shape_findings(_TRACING_QUESTION, hits)
    assert any(
        "`dbo.Regions` LEFT-joins `#RegionSegmentAgg` when building `#Enriched`" in f
        for f in findings
    )
    joined_text = " ".join(findings)
    assert "PriorPeriodAmount" not in joined_text
    assert "PartitionBy" not in joined_text.replace(" ", "")


def test_cte_tail_in_isolation_produces_no_finding():
    """The literal ;WITH Scored AS (...) tail of CustomerChurnRisk, fed
    as its own standalone anchor text — the CTE's own FROM #Enriched has
    no JOIN, and the outer FROM Scored has no JOIN either, so this is
    empty regardless of the CTE-skip rule (see the synthetic case below
    for proof the skip itself is active, not just structurally moot)."""
    hits = [
        _chunk(
            "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
            "dbo.usp_BuildReport_CustomerChurnRisk",
            "procedure",
            _CTE_TAIL_ONLY,
            score=0.9,
        )
    ]
    assert join_shape_findings(_TRACING_QUESTION, hits) == []


def test_cte_skip_is_deliberate_not_coincidental():
    """A synthetic CTE-consuming SELECT whose own top-level FROM/LEFT JOIN
    *would* produce a finding if the CTE-skip rule weren't applied (both
    #Sibling and CteX's driving role are structurally reachable — nothing
    here is hidden inside an opaque Parenthesis/Where the way a CTE's own
    internal definition would be). Proven active by flipping the same
    shape without the leading `;WITH` and confirming it *does* fire."""
    with_cte = """
CREATE OR ALTER PROCEDURE dbo.usp_SyntheticCte
AS
BEGIN
    ;WITH CteX AS (
        SELECT ID FROM dbo.Seed
    )
    SELECT x.ID
    INTO #Result
    FROM CteX x
    LEFT JOIN #Sibling s ON s.ID = x.ID;
END;
"""
    without_cte = """
CREATE OR ALTER PROCEDURE dbo.usp_SyntheticNoCte
AS
BEGIN
    SELECT x.ID
    INTO #Result
    FROM CteX x
    LEFT JOIN #Sibling s ON s.ID = x.ID;
END;
"""
    hits_with = [
        _chunk("sql:x.dbo.usp_SyntheticCte", "dbo.usp_SyntheticCte", "procedure", with_cte, 0.9)
    ]
    hits_without = [
        _chunk("sql:x.dbo.usp_SyntheticNoCte", "dbo.usp_SyntheticNoCte", "procedure", without_cte, 0.9)
    ]
    assert join_shape_findings(_TRACING_QUESTION, hits_with) == []
    assert join_shape_findings(_TRACING_QUESTION, hits_without) != []


def test_where_exists_stays_opaque():
    """dbo.Customers appears both (a) as a normal top-level FROM target in
    the #ActiveCustomerOrders chain, and (b) only inside the nested
    `WHERE EXISTS (SELECT ... FROM dbo.Customers ...)` guarding the
    #Enriched chain. The #Enriched/LEFT finding (driving
    dbo.StagingCustomerSegment, joined #RFM) must never mention Customers
    — proving the nested WHERE EXISTS's own FROM was never walked into."""
    hits = [
        _churn_risk_hit(),
        _chunk("sql:x.dbo.Customers", "dbo.Customers", "table", "CREATE TABLE dbo.Customers (...)", 0.5),
    ]
    findings = join_shape_findings(_TRACING_QUESTION, hits)
    enriched_finding = next(f for f in findings if "StagingCustomerSegment" in f)
    assert "Customers" not in enriched_finding


def test_gated_on_tracing_question():
    hits = [_churn_risk_hit()]
    assert join_shape_findings(_NON_TRACING_QUESTION, hits) == []


def test_no_anchor_or_empty_hits():
    assert join_shape_findings(_TRACING_QUESTION, []) == []


def test_comma_join_unrecognized_shape_is_silent_not_a_crash():
    text = """
CREATE OR ALTER PROCEDURE dbo.usp_CommaJoin
AS
BEGIN
    SELECT a.ID
    INTO #Result
    FROM dbo.A a, dbo.B b
    WHERE a.ID = b.ID;
END;
"""
    hits = [_chunk("sql:x.dbo.usp_CommaJoin", "dbo.usp_CommaJoin", "procedure", text, 0.9)]
    assert join_shape_findings(_TRACING_QUESTION, hits) == []


def test_cap_enforcement():
    text = """
CREATE OR ALTER PROCEDURE dbo.usp_ManyJoins
AS
BEGIN
    SELECT a.ID
    INTO #Result
    FROM #TempA a
    JOIN #TempB b ON b.ID=a.ID
    JOIN #TempC c ON c.ID=a.ID
    JOIN #TempD d ON d.ID=a.ID
    JOIN #TempE e ON e.ID=a.ID
    JOIN #TempF f ON f.ID=a.ID
    JOIN #TempG g ON g.ID=a.ID
    JOIN #TempH h ON h.ID=a.ID;
END;
"""
    hits = [_chunk("sql:x.dbo.usp_ManyJoins", "dbo.usp_ManyJoins", "procedure", text, 0.9)]
    findings = join_shape_findings(_TRACING_QUESTION, hits)
    assert len(findings) == 5


def test_ungated_chain_touching_no_candidates_is_silent():
    text = """
CREATE OR ALTER PROCEDURE dbo.usp_Gated
AS
BEGIN
    SELECT a.ID
    INTO #Keep
    FROM dbo.Unrelated1 a
    JOIN dbo.Unrelated2 b ON b.ID = a.ID;
END;
"""
    hits = [_chunk("sql:x.dbo.usp_Gated", "dbo.usp_Gated", "procedure", text, 0.9)]
    assert join_shape_findings(_TRACING_QUESTION, hits) == []
