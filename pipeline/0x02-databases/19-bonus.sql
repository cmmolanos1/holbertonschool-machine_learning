-- creates a stored procedure AddBonus that adds a new correction for a student.
delimiter //
CREATE PROCEDURE AddBonus(IN user_id INT, project_name VARCHAR(255), score INT)
BEGIN
    SET @count = (SELECT COUNT(*) FROM projects WHERE projects.name LIKE project_name);
    IF @count = 0 THEN
        INSERT INTO projects(name)
        VALUES (project_name);
    END IF;

    -- SET @project_id = (SELECT id FROM projects WHERE name = project_name);

    INSERT INTO corrections
    VALUES (user_id, (SELECT id FROM projects WHERE name = project_name), score);
END
//
delimiter ;