-- Explore data
SELECT *
FROM Housing;


-- Standerize Date
SELECT STRFTIME('%d-%m-%Y %H:%M:%S', SaleDate) AS CorrectDate
FROM Housing;
-- Add CorrectDate to the Housing table
ALTER TABLE Housing ADD COLUMN SaleDateCorrect DATETIME;
UPDATE Housing SET SaleDateCorrect = STRFTIME('%d-%m-%Y %H:%M:%S', SaleDate);


-- Populate Property Address data
SELECT *
FROM Housing
WHERE PropertyAddress IS NULL;

SELECT a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, IFNULL(a.PropertyAddress, b.PropertyAddress)
FROM Housing a JOIN Housing b
	ON a.ParcelID = b.ParcelID AND a.UniqueID != b.UniqueID
WHERE a.PropertyAddress IS NULL;

UPDATE Housing SET PropertyAddress = IFNULL(a.PropertyAddress, b.PropertyAddress)
	FROM Housing a JOIN Housing b
		ON a.ParcelID = b.ParcelID AND a.UniqueID != b.UniqueID
	WHERE a.PropertyAddress IS NULL;



-- Breaking out Address into Individual Columns (Address, City, State)
SELECT PropertyAddress, SUBSTR(PropertyAddress, 1, INSTR(PropertyAddress, ',') -1 ) as Address,
	SUBSTR(PropertyAddress, INSTR(PropertyAddress, ',')+1) AS City
FROM Housing;

ALTER TABLE Housing ADD COLUMN PropertySplitAddress VARCHAR(255);
Update Housing SET PropertySplitAddress = SUBSTR(PropertyAddress, 1, INSTR(PropertyAddress, ',') -1 );

ALTER TABLE Housing ADD COLUMN PropertySplitCity VARCHAR(255);
Update Housing SET PropertySplitCity = SUBSTR(PropertyAddress, INSTR(PropertyAddress, ',')+1);


-- Change Y and N to Yes and No in "Sold as Vacant" field
SELECT DISTINCT(SoldAsVacant), COUNT(SoldAsVacant)
FROM Housing
GROUP BY SoldAsVacant
ORDER BY 2;

Select SoldAsVacant, CASE
	WHEN SoldAsVacant = 'Y' THEN 'Yes'
	WHEN SoldAsVacant = 'N' THEN 'No'
	ELSE SoldAsVacant
	END AS Correct
From Housing
WHERE SoldAsVacant IN ('Y', 'N');

UPDATE Housing SET SoldAsVacant = CASE
	WHEN SoldAsVacant = 'Y' THEN 'Yes'
	WHEN SoldAsVacant = 'N' THEN 'No'
	ELSE SoldAsVacant
	END;


-- Remove Duplicates
WITH RowNumberCTE AS (
	SELECT *, ROW_NUMBER() OVER (
		PARTITION BY ParcelID, PropertyAddress,
				 SalePrice, SaleDate, LegalReference
		ORDER BY UniqueID) AS row_num
	FROM Housing
)
DELETE
From RowNumberCTE
Where row_num > 1;
-- Order by PropertyAddress;


-- Delete Unused Columns
ALTER TABLE Housing DROP COLUMN OwnerAddress;
ALTER TABLE Housing DROP COLUMN TaxDistrict;
ALTER TABLE Housing DROP COLUMN PropertyAddress;
ALTER TABLE Housing DROP COLUMN SaleDate;
