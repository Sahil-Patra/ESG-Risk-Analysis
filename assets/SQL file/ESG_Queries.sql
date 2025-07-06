
---Column Name and Their Datatypes

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'SP_500_ESG_Risk_Ratings'; 

---Cleaning and changing the datatypes of columns

UPDATE SP_500_ESG_Risk_Ratings
SET Environment_Risk_Score = NULLIF(LTRIM(RTRIM(REPLACE(Environment_Risk_Score, ',', ''))), '');;

--
UPDATE SP_500_ESG_Risk_Ratings
SET Governance_Risk_Score = NULLIF(LTRIM(RTRIM(REPLACE(Governance_Risk_Score, ',', ''))), '');

--
UPDATE SP_500_ESG_Risk_Ratings
SET Social_Risk_Score = NULLIF(LTRIM(RTRIM(REPLACE(Social_Risk_Score, ',', ''))), '');

--
UPDATE SP_500_ESG_Risk_Ratings
SET Total_ESG_Risk_score = NULLIF(LTRIM(RTRIM(REPLACE(Total_ESG_Risk_score, ',', ''))), '');

--
UPDATE SP_500_ESG_Risk_Ratings
SET Controversy_Score = NULLIF(LTRIM(RTRIM(REPLACE(Controversy_Score, ',', ''))), '');

UPDATE SP_500_ESG_Risk_Ratings
SET Controversy_Score = REPLACE(Controversy_Score, 'N/A', '0');

UPDATE SP_500_ESG_Risk_Ratings
SET Controversy_Score = REPLACE(Controversy_Score, 'null', '0');

UPDATE SP_500_ESG_Risk_Ratings
SET Controversy_Score = Coalesce(Controversy_Score, '0');

ALTER TABLE SP_500_ESG_Risk_Ratings
ALTER COLUMN Controversy_Score Integer;

-- Industry-wide average ESG component scores
SELECT
    Industry,
    Round(AVG(Environment_Risk_Score), 2) AS Avg_Environment,
    Round(AVG(Governance_Risk_Score), 2) AS Avg_Governance,
    Round(AVG(Social_Risk_Score), 2) AS Avg_Social
FROM SP_500_ESG_Risk_Ratings
GROUP BY Industry
ORDER BY Industry;

-- Count of companies by ESG risk level

SELECT ESG_Risk_Level, COUNT(*) AS Company_Count
FROM SP_500_ESG_Risk_Ratings
WHERE ESG_Risk_Level IS NOT NULL
GROUP BY ESG_Risk_Level
ORDER BY Company_Count DESC;

-- ESG score by risk category

SELECT ESG_Risk_Level,
       AVG(Total_ESG_Risk_score) AS Avg_Total_ESG
FROM SP_500_ESG_Risk_Ratings
GROUP BY ESG_Risk_Level;

-- Governance vs ESG score

SELECT ROUND(AVG(Governance_Risk_Score), 2) AS Avg_Gov,
       ROUND(AVG(Total_ESG_Risk_Score), 2) AS Avg_Total
FROM SP_500_ESG_Risk_Ratings;


-- No. of companies per each risk level

SELECT [ESG_Risk_Level], COUNT(*) AS Company_Count
FROM SP_500_ESG_Risk_Ratings
WHERE ESG_Risk_Level is not null
GROUP BY [ESG_Risk_Level]
ORDER BY Company_Count DESC;
