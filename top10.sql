SELECT *FROM customers;
select COUNT(*) from transactions;
SELECT * from transactions;

CREATE TABLE IF NOT EXISTS products (
    StockCode VARCHAR(50),
    descripton TEXT
);

-- loading data
LOAD DATA INFILE 'ProductInfo.csv'
INTO TABLE products
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(StockCode, descripton);

-- checking
-- Block 1: Create a temporary table for top 10 products excluding exact 'NA' match
CREATE TABLE top_ten_products AS
SELECT 
    p.StockCode,
    p.descripton,
    SUM(t.Quantity) as total_quantity_sold
FROM 
    transactions t
    JOIN Products p ON t.StockCode = p.StockCode
WHERE 
    p.descripton != 'NA'  -- This will exclude the exact 'NA' match
GROUP BY 
    p.StockCode,
    p.descripton
ORDER BY 
    total_quantity_sold DESC
LIMIT 10;

-- query
SELECT * FROM top_ten_products;
-- main query

SELECT 
    t.Invoice,
    t.StockCode,
    p.descripton,
    t.Quantity,
    t.`customer ID`,
    t.InvoiceDate,
    c.Country
FROM 
    transactions t
    JOIN top_ten_products tp ON t.StockCode = tp.StockCode
    JOIN Products p ON t.StockCode = p.StockCode
    LEFT JOIN customers c ON t.`customer ID` = c.`Customer ID`
ORDER BY 
    tp.total_quantity_sold DESC, t.InvoiceDate DESC;

-- next, 
SELECT *FROM top_ten_products;