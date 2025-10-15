create database ble_receiver;

drop table probe_imu;

CREATE TABLE probe_imu (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    report_second INT,
    angle_report_loop_count int,
    transferred_loop_count int,
    yaw FLOAT,
    pitch FLOAT,
    roll FLOAT,
    quat_r FLOAT,
    quat_i FLOAT,
    quat_j FLOAT,
    quat_k FLOAT,
    accel_x FLOAT,
    accel_y FLOAT,
    accel_z FLOAT,
    status INT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

INSERT INTO probe_imu(ble_counter, report_second,angle_report_loop_count,transferred_loop_count, yaw, pitch, roll, quat_r, quat_i, quat_j, quat_k, accel_x, accel_y, accel_z, status)
VALUES (1, 0,0,0,0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);



drop table probe_imu_history;

CREATE TABLE probe_imu_history (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    report_second INT,
    angle_report_loop_count int,
    transferred_loop_count int,
    yaw FLOAT,
    pitch FLOAT,
    roll FLOAT,
    quat_r FLOAT,
    quat_i FLOAT,
    quat_j FLOAT,
    quat_k FLOAT,
    accel_x FLOAT,
    accel_y FLOAT,
    accel_z FLOAT,
    status INT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

SET SQL_SAFE_UPDATES = 0;
DELETE FROM probe_imu_history;
SET SQL_SAFE_UPDATES = 1;
