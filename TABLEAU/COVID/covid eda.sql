-- Inspect deaths
SELECT *
FROM CovidDeaths;


-- Inspect vaccinations
SELECT *
FROM CovidVaccinations;


-- Total Cases vs Total Deaths in Egypt
SELECT Location, date, total_cases, total_deaths,
	(CAST(total_deaths AS DOUBLE) / CAST(total_cases AS DOUBLE))*100 AS DeathPercentage
FROM CovidDeaths
WHERE location = 'Egypt' AND continent IS NOT NULL
ORDER BY 1,2;


-- Total Cases vs Population in Egypt
SELECT Location, date, Population, total_cases,
	(CAST(total_deaths AS DOUBLE) / CAST(population AS DOUBLE))*100 AS PercentPopulationInfected
FROM CovidDeaths
WHERE location = 'Egypt' AND continent IS NOT NULL
ORDER BY 1,2;


-- Countries with Highest Infection Rate compared to Population
SELECT Location, date, Population, total_cases,
	ROUND((CAST(total_deaths AS DOUBLE) / CAST(population AS DOUBLE)), 5) AS PercentPopulationInfected
FROM CovidDeaths
WHERE continent IS NOT NULL
ORDER BY PercentPopulationInfected DESC;


-- Countries with Highest Death Count per Population
SELECT Location, continent, MAX(CAST(Total_deaths AS INT)) AS TotalDeathCount
FROM CovidDeaths
WHERE continent IS NOT NULL
GROUP BY  Location, continent
ORDER BY TotalDeathCount DESC;


-- Showing contintents with the number of deaths
SELECT continent, SUM(CAST(Total_deaths AS INT)) AS TotalDeathCount
FROM CovidDeaths
WHERE continent IS NOT NULL
GROUP BY  continent
ORDER BY TotalDeathCount DESC;


-- GLOBAL NUMBERS (cases VS deaths)
SELECT SUM(new_cases) AS total_cases, SUM(CAST(Total_deaths AS INT)) AS total_deaths,
	ROUND(SUM(CAST(new_deaths AS DOUBLE))/SUM(CAST(new_cases AS DOUBLE))*100, 5) AS DeathPercentage
FROM CovidDeaths
WHERE continent IS NOT NULL;


-- Total Population vs Vaccinations
-- Percentage of Population that has recieved at least one Covid Vaccine
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
	SUM(CAST(v.new_vaccinations AS INT)) OVER (
		Partition BY d.Location ORDER BY d.location, d.Date
	) AS RollingVaccinatedPeople
FROM CovidDeaths d JOIN CovidVaccinations v
	ON d.location = v.location and d.date = v.date
WHERE d.continent IS NOT NULL
ORDER BY 2,3;


-- Using CTE to perform Calculation on Partition By in previous query
WITH VaccinatedPeople (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated) AS (
	SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
	SUM(CAST(v.new_vaccinations AS INT)) OVER (
		Partition BY d.Location ORDER BY d.location, d.Date
	) AS RollingPeopleVaccinated
	FROM CovidDeaths d JOIN CovidVaccinations v
		ON d.location = v.location and d.date = v.date
	WHERE d.continent IS NOT NULL
)

SELECT *
FROM VaccinatedPeople;


-- Using Temp Table to perform Calculation on Partition By in previous query
DROP TABLE IF EXISTS VaccinatedPeople;
Create Table VaccinatedPeople(
	Continent VARCHAR(255),
	Location VARCHAR(255),
	Date DATETIME,
	Population INTEGER,
	New_vaccinations INTEGER,
	RollingPeopleVaccinated INTEGER
	)

INSERT INTO VaccinatedPeople
	SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
		SUM(CAST(v.new_vaccinations AS INT)) OVER (
			Partition BY d.Location ORDER BY d.location, d.Date
		) AS RollingPeopleVaccinated
	FROM CovidDeaths d JOIN CovidVaccinations v
		ON d.location = v.location and d.date = v.date
	WHERE d.continent IS NOT NULL;

SELECT *
FROM VaccinatedPeople;


-- Creating View to store data for later visualizations
CREATE VIEW VaccinatedPeople AS
	SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
		SUM(CAST(v.new_vaccinations AS INT)) OVER (
			Partition BY d.Location ORDER BY d.location, d.Date
		) AS RollingPeopleVaccinated
	FROM CovidDeaths d JOIN CovidVaccinations v
		ON d.location = v.location and d.date = v.date
	WHERE d.continent IS NOT NULL;
