"""
Unit tests for core.sql_dependency_extraction.find_references.

Uses the literal body of dbo.usp_BuildReport_CustomerChurnRisk from the
RetailReportingDemo fixture (fixtures/sql/06_reports.sql) as the primary
case — this is the exact procedure whose derivation-tracing answer
originally missed the #Enriched/StagingCustomerSegment stage, motivating
the whole SQL dependency graph feature. Regex/matching regressions here
should be caught without needing a live SQL Server connection.
"""

from core.sql_dependency_extraction import find_references

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

_SELF_KEY = "dbo.usp_buildreport_customerchurnrisk"

_KNOWN_OBJECTS = {
    "dbo.customers": ("dbo.Customers", "table"),
    "dbo.orders": ("dbo.Orders", "table"),
    "dbo.orderlines": ("dbo.OrderLines", "table"),
    "dbo.fn_netlineamount": ("dbo.fn_NetLineAmount", "function"),
    "dbo.usp_lookupcustomersegment": ("dbo.usp_LookupCustomerSegment", "procedure"),
    "dbo.stagingcustomersegment": ("dbo.StagingCustomerSegment", "table"),
    "dbo.report_customerchurnrisk": ("dbo.Report_CustomerChurnRisk", "table"),
    _SELF_KEY: ("dbo.usp_BuildReport_CustomerChurnRisk", "procedure"),
    # Distractors that must NOT appear as edges — these are real fixture
    # objects that a naive "any retrieved object" check once flagged as
    # false-positive "missing" dependencies (see core.trace_completeness's
    # commit history), even though this procedure's body never mentions
    # them at all.
    "dbo.usp_buildreport_monthlysalesbyregion": (
        "dbo.usp_BuildReport_MonthlySalesByRegion",
        "procedure",
    ),
    "dbo.vw_customerordersummary": ("dbo.vw_CustomerOrderSummary", "view"),
    "dbo.usp_stagecompletedorderlines": ("dbo.usp_StageCompletedOrderLines", "procedure"),
}


def test_customer_churn_risk_proc_edges_match_exactly():
    edges = find_references(_CUSTOMER_CHURN_RISK_PROC, _SELF_KEY, _KNOWN_OBJECTS)

    subject = "dbo.usp_BuildReport_CustomerChurnRisk"
    expected = {
        (subject, "references", "dbo.Customers"),
        (subject, "references", "dbo.Orders"),
        (subject, "references", "dbo.OrderLines"),
        (subject, "calls", "dbo.fn_NetLineAmount"),
        (subject, "calls", "dbo.usp_LookupCustomerSegment"),
        (subject, "references", "dbo.StagingCustomerSegment"),
        (subject, "writes_to", "dbo.Report_CustomerChurnRisk"),
    }

    assert set(edges) == expected


def test_distractor_objects_never_appear():
    """Objects genuinely absent from the proc body must never be flagged,
    even though they're valid entries in the known-object catalog."""
    edges = find_references(_CUSTOMER_CHURN_RISK_PROC, _SELF_KEY, _KNOWN_OBJECTS)
    referenced_objects = {obj for _subject, _predicate, obj in edges}

    assert "dbo.usp_BuildReport_MonthlySalesByRegion" not in referenced_objects
    assert "dbo.vw_CustomerOrderSummary" not in referenced_objects
    assert "dbo.usp_StageCompletedOrderLines" not in referenced_objects


def test_self_reference_excluded():
    """An object never references itself, even though its own name
    trivially appears in its own definition (the CREATE statement)."""
    edges = find_references(_CUSTOMER_CHURN_RISK_PROC, _SELF_KEY, _KNOWN_OBJECTS)
    subjects_and_objects = {s for s, _p, _o in edges} | {o for _s, _p, o in edges}

    assert "dbo.usp_BuildReport_CustomerChurnRisk" not in {
        o for _s, _p, o in edges
    }
    assert all(s == "dbo.usp_BuildReport_CustomerChurnRisk" for s, _p, _o in edges)
    assert subjects_and_objects  # sanity: the proc does reference something


def test_comment_only_mention_is_not_a_false_match_source():
    """A name that appears ONLY inside a comment, and is never otherwise
    referenced in real code, must not be reported — comments are
    stripped before matching."""
    definition = """
    CREATE PROCEDURE dbo.Foo AS
    BEGIN
        -- see dbo.Bar for a similar pattern, not used here
        SELECT 1;
    END;
    """
    known = {
        "dbo.foo": ("dbo.Foo", "procedure"),
        "dbo.bar": ("dbo.Bar", "table"),
    }
    edges = find_references(definition, "dbo.foo", known)
    assert edges == []


def test_bare_name_matches_only_for_dbo_schema():
    """An unqualified reference is only resolved against the `dbo`
    schema — a bare name colliding with a non-dbo-schema object must not
    produce a spurious edge."""
    definition = "CREATE PROCEDURE dbo.Foo AS BEGIN SELECT * FROM Widgets; END;"
    known = {
        "dbo.foo": ("dbo.Foo", "procedure"),
        "dbo.widgets": ("dbo.Widgets", "table"),
        "reporting.widgets": ("reporting.Widgets", "table"),
    }
    edges = find_references(definition, "dbo.foo", known)
    assert edges == [("dbo.Foo", "references", "dbo.Widgets")]


def test_writes_to_predicate_for_insert_and_update_targets():
    definition = (
        "CREATE PROCEDURE dbo.Foo AS BEGIN "
        "INSERT INTO dbo.Written (X) SELECT 1; "
        "UPDATE dbo.AlsoWritten SET X = 1; "
        "SELECT * FROM dbo.OnlyRead; "
        "END;"
    )
    known = {
        "dbo.foo": ("dbo.Foo", "procedure"),
        "dbo.written": ("dbo.Written", "table"),
        "dbo.alsowritten": ("dbo.AlsoWritten", "table"),
        "dbo.onlyread": ("dbo.OnlyRead", "table"),
    }
    edges = set(find_references(definition, "dbo.foo", known))
    assert edges == {
        ("dbo.Foo", "writes_to", "dbo.Written"),
        ("dbo.Foo", "writes_to", "dbo.AlsoWritten"),
        ("dbo.Foo", "references", "dbo.OnlyRead"),
    }
