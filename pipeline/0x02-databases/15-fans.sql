-- ranks country origins of bands, ordered by the number of (non-unique) fans
SELECT origin, SUM(fans) as fans
FROM metal_bands
GROUP BY origin
ORDER BY fans DESC;