create database ble_receiver;

drop table probe_imu;

CREATE TABLE probe_imu (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    report_second INT,
    angle_report_loop_count int,
    transferred_loop_count int,
    yaw FLOAT,
    yaw_delta FLOAT,
    yaw_calibrated FLOAT,
    pitch FLOAT,
    roll FLOAT,
    quat_r FLOAT,
    quat_i FLOAT,
    quat_j FLOAT,
    quat_k FLOAT,
    status INT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

INSERT INTO probe_imu(ble_counter, report_second,angle_report_loop_count,transferred_loop_count, yaw, yaw_delta, yaw_calibrated,  pitch, roll, quat_r, quat_i, quat_j, quat_k, status)
VALUES (1, 0,0,0,0, 0, 0, 0, 0, 1, 0, 0, 0, 0);



drop table probe_imu_history;

CREATE TABLE probe_imu_history (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    report_second INT,
    angle_report_loop_count int,
    transferred_loop_count int,
    yaw FLOAT,
	yaw_delta FLOAT,
    yaw_calibrated FLOAT,
    pitch FLOAT,
    roll FLOAT,
    quat_r FLOAT,
    quat_i FLOAT,
    quat_j FLOAT,
    quat_k FLOAT,
    status INT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

SET SQL_SAFE_UPDATES = 0;
DELETE FROM probe_imu_history;
SET SQL_SAFE_UPDATES = 1;


drop table car_imu;

CREATE TABLE car_imu (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    report_second INT,
    angle_report_loop_count int,
    transferred_loop_count int,
    yaw FLOAT,
	yaw_delta FLOAT,
    yaw_calibrated FLOAT,
    pitch FLOAT,
    roll FLOAT,
    quat_r FLOAT,
    quat_i FLOAT,
    quat_j FLOAT,
    quat_k FLOAT,
    status INT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

INSERT INTO car_imu(ble_counter, report_second,angle_report_loop_count,transferred_loop_count, yaw, yaw_delta, yaw_calibrated, pitch, roll, quat_r, quat_i, quat_j, quat_k, status)
VALUES (1, 0,0,0,0, 0, 0, 0, 0, 1, 0, 0, 0, 0);



drop table car_imu_history;

CREATE TABLE car_imu_history (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    report_second INT,
    angle_report_loop_count int,
    transferred_loop_count int,
    yaw FLOAT,
	yaw_delta FLOAT,
    yaw_calibrated FLOAT,
    pitch FLOAT,
    roll FLOAT,
    quat_r FLOAT,
    quat_i FLOAT,
    quat_j FLOAT,
    quat_k FLOAT,
    status INT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

SET SQL_SAFE_UPDATES = 0;
DELETE FROM car_imu_history;
SET SQL_SAFE_UPDATES = 1;


drop table pressure_mat;

CREATE TABLE pressure_mat (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    x INT,
    y int,
    max_value int,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);


drop table pressure_mat;

CREATE TABLE pressure_mat_history (
    idx INT NOT NULL AUTO_INCREMENT,
    ble_counter INT,
    x INT,
    y int,
    max_value int,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);



drop table calibrated_imu;

CREATE TABLE calibrated_imu (
    idx INT NOT NULL AUTO_INCREMENT,
    yaw FLOAT,
    pitch FLOAT,
    roll FLOAT,
    yaw_2 FLOAT,
    pitch_2 FLOAT,
    roll_2 FLOAT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

INSERT INTO calibrated_imu(yaw, pitch, roll, yaw_2, pitch_2, roll_2)
VALUES (0,0,0, 0, 0, 0);
