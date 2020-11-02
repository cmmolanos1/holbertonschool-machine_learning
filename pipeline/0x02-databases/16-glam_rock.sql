--  lists all bands with Glam Rock as their main style, ranked by their longevity
SELECT band_name, IF(split IS NULL, YEAR(CURDATE()), split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam Rock%'
ORDER BY lifespan DESC;