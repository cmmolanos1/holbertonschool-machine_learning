-- AVG TEMPERATURES
SELECT city, AVG(value) AS t_avg
FROM temperatures
GROUP BY city
ORDER BY t_avg DESC;