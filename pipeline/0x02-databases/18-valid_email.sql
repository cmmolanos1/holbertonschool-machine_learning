--  creates a trigger that resets the attribute valid_email only when the email has been changed
delimiter //
CREATE TRIGGER ResetValidEmail
    BEFORE UPDATE
    ON users
    FOR EACH ROW
        BEGIN
            IF OLD.email <> NEW.email THEN
                IF NEW.email REGEXP '^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9._-]@[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]\\.[a-zA-Z]{2,63}$' THEN
                    SET NEW.valid_email = 1;
                ELSE
                    SET NEW.valid_email = 0;
                END IF;
            END IF;
        END;
//
delimiter ;

