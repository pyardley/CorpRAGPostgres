USE RetailReportingDemo;
GO

IF OBJECT_ID('dbo.AuditLog', 'U') IS NOT NULL DROP TABLE dbo.AuditLog;
IF OBJECT_ID('dbo.Payments', 'U') IS NOT NULL DROP TABLE dbo.Payments;
IF OBJECT_ID('dbo.Returns', 'U') IS NOT NULL DROP TABLE dbo.Returns;
IF OBJECT_ID('dbo.OrderLines', 'U') IS NOT NULL DROP TABLE dbo.OrderLines;
IF OBJECT_ID('dbo.Orders', 'U') IS NOT NULL DROP TABLE dbo.Orders;
IF OBJECT_ID('dbo.Inventory', 'U') IS NOT NULL DROP TABLE dbo.Inventory;
IF OBJECT_ID('dbo.Employees', 'U') IS NOT NULL DROP TABLE dbo.Employees;
IF OBJECT_ID('dbo.Customers', 'U') IS NOT NULL DROP TABLE dbo.Customers;
IF OBJECT_ID('dbo.Products', 'U') IS NOT NULL DROP TABLE dbo.Products;
IF OBJECT_ID('dbo.ProductCategories', 'U') IS NOT NULL DROP TABLE dbo.ProductCategories;
IF OBJECT_ID('dbo.Warehouses', 'U') IS NOT NULL DROP TABLE dbo.Warehouses;
IF OBJECT_ID('dbo.Regions', 'U') IS NOT NULL DROP TABLE dbo.Regions;
GO

CREATE TABLE dbo.Regions (
    RegionID    INT IDENTITY(1,1) PRIMARY KEY,
    RegionName  VARCHAR(60) NOT NULL,
    Country     VARCHAR(60) NOT NULL
);
GO

CREATE TABLE dbo.Warehouses (
    WarehouseID     INT IDENTITY(1,1) PRIMARY KEY,
    WarehouseName   VARCHAR(100) NOT NULL,
    RegionID        INT NOT NULL FOREIGN KEY REFERENCES dbo.Regions(RegionID)
);
GO

CREATE TABLE dbo.ProductCategories (
    CategoryID      INT IDENTITY(1,1) PRIMARY KEY,
    CategoryName    VARCHAR(60) NOT NULL
);
GO

CREATE TABLE dbo.Products (
    ProductID               INT IDENTITY(1,1) PRIMARY KEY,
    ProductName             VARCHAR(150) NOT NULL,
    CategoryID              INT NOT NULL FOREIGN KEY REFERENCES dbo.ProductCategories(CategoryID),
    UnitCost                DECIMAL(10,2) NOT NULL CHECK (UnitCost >= 0),
    ListPrice               DECIMAL(10,2) NOT NULL CHECK (ListPrice >= 0),
    SupplierLeadTimeDays    INT NOT NULL CHECK (SupplierLeadTimeDays >= 0)
);
GO

CREATE TABLE dbo.Inventory (
    ProductID       INT NOT NULL FOREIGN KEY REFERENCES dbo.Products(ProductID),
    WarehouseID     INT NOT NULL FOREIGN KEY REFERENCES dbo.Warehouses(WarehouseID),
    QuantityOnHand  INT NOT NULL CHECK (QuantityOnHand >= 0),
    ReorderPoint    INT NOT NULL CHECK (ReorderPoint >= 0),
    LastRestockDate DATE NOT NULL,
    CONSTRAINT PK_Inventory PRIMARY KEY (ProductID, WarehouseID)
);
GO

CREATE TABLE dbo.Customers (
    CustomerID      INT IDENTITY(1,1) PRIMARY KEY,
    CustomerName    VARCHAR(150) NOT NULL,
    Email           VARCHAR(200) NOT NULL,
    Segment         VARCHAR(30) NOT NULL CHECK (Segment IN ('Enterprise','Mid-Market','SMB','Consumer')),
    RegionID        INT NOT NULL FOREIGN KEY REFERENCES dbo.Regions(RegionID),
    SignupDate      DATE NOT NULL,
    IsActive        BIT NOT NULL DEFAULT 1
);
GO

CREATE TABLE dbo.Employees (
    EmployeeID      INT IDENTITY(1,1) PRIMARY KEY,
    FullName        VARCHAR(150) NOT NULL,
    HireDate        DATE NOT NULL,
    RegionID        INT NOT NULL FOREIGN KEY REFERENCES dbo.Regions(RegionID),
    CommissionTier  VARCHAR(20) NOT NULL CHECK (CommissionTier IN ('Bronze','Silver','Gold','Platinum')),
    ManagerID       INT NULL FOREIGN KEY REFERENCES dbo.Employees(EmployeeID)
);
GO

CREATE TABLE dbo.Orders (
    OrderID         INT IDENTITY(1,1) PRIMARY KEY,
    CustomerID      INT NOT NULL FOREIGN KEY REFERENCES dbo.Customers(CustomerID),
    EmployeeID      INT NOT NULL FOREIGN KEY REFERENCES dbo.Employees(EmployeeID),
    OrderDate       DATE NOT NULL,
    Status          VARCHAR(20) NOT NULL CHECK (Status IN ('Completed','Pending','Cancelled','Shipped')),
    ModifiedDate    DATETIME NOT NULL DEFAULT GETDATE()
);
GO

CREATE TABLE dbo.OrderLines (
    OrderLineID     INT IDENTITY(1,1) PRIMARY KEY,
    OrderID         INT NOT NULL FOREIGN KEY REFERENCES dbo.Orders(OrderID),
    ProductID       INT NOT NULL FOREIGN KEY REFERENCES dbo.Products(ProductID),
    Quantity        INT NOT NULL CHECK (Quantity > 0),
    UnitPrice       DECIMAL(10,2) NOT NULL CHECK (UnitPrice >= 0),
    DiscountPct     DECIMAL(5,2) NOT NULL DEFAULT 0 CHECK (DiscountPct BETWEEN 0 AND 100)
);
GO

CREATE TABLE dbo.Returns (
    ReturnID        INT IDENTITY(1,1) PRIMARY KEY,
    OrderLineID     INT NOT NULL FOREIGN KEY REFERENCES dbo.OrderLines(OrderLineID),
    ReturnDate      DATE NOT NULL,
    Quantity        INT NOT NULL CHECK (Quantity > 0),
    Reason          VARCHAR(200) NULL
);
GO

CREATE TABLE dbo.Payments (
    PaymentID       INT IDENTITY(1,1) PRIMARY KEY,
    OrderID         INT NOT NULL FOREIGN KEY REFERENCES dbo.Orders(OrderID),
    PaymentDate     DATE NOT NULL,
    Amount          DECIMAL(12,2) NOT NULL CHECK (Amount >= 0),
    Method          VARCHAR(30) NOT NULL CHECK (Method IN ('CreditCard','ACH','Wire','Check','PayPal'))
);
GO

CREATE TABLE dbo.AuditLog (
    AuditID         INT IDENTITY(1,1) PRIMARY KEY,
    TableName       VARCHAR(60) NOT NULL,
    Operation       VARCHAR(10) NOT NULL,
    RecordID        INT NOT NULL,
    ChangedAt       DATETIME NOT NULL DEFAULT GETDATE(),
    ChangedBy       VARCHAR(100) NOT NULL DEFAULT SUSER_SNAME()
);
GO
