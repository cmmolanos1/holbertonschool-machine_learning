-- creates a function SafeDiv that divides (and returns) the first by the second
-- number or returns 0 if the second number is equal to 0
delimiter //
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
BEGIN
    DECLARE division FLOAT;
    SET division = 0;

    IF b <> 0 THEN
        SET division = a / b;
    end if;

    RETURN division;
END
//
delimiter ;