-- creates a stored procedure AddBonus that adds a new correction for a student.
delimiter //
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    SET @average = (SELECT AVG(score)
                    FROM corrections
                    WHERE corrections.user_id = user_id);

    UPDATE users
        SET average_score = @average WHERE id = user_id;
END
//
delimiter ;