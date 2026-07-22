USE RetailReportingDemo;
GO

CREATE OR ALTER FUNCTION dbo.fn_FiscalPeriod(@AsOfDate DATE)
RETURNS VARCHAR(10)
AS
BEGIN
    -- Fiscal year starts July 1st: a date in Jan-Jun belongs to the fiscal
    -- year that started the previous July.
    DECLARE @FiscalYear INT;
    IF MONTH(@AsOfDate) >= 7
        SET @FiscalYear = YEAR(@AsOfDate) + 1;
    ELSE
        SET @FiscalYear = YEAR(@AsOfDate);

    RETURN 'FY' + CAST(@FiscalYear AS VARCHAR(4)) + '-' + RIGHT('0' + CAST(MONTH(@AsOfDate) AS VARCHAR(2)), 2);
END;
GO

CREATE OR ALTER FUNCTION dbo.fn_NetLineAmount(@Quantity INT, @UnitPrice DECIMAL(10,2), @DiscountPct DECIMAL(5,2))
RETURNS DECIMAL(12,2)
AS
BEGIN
    RETURN CAST(@Quantity AS DECIMAL(12,2)) * @UnitPrice * (1 - (@DiscountPct / 100.0));
END;
GO
