create database ble_receiver;

drop table probe_imu;

CREATE TABLE probe_imu (
    idx INT NOT NULL AUTO_INCREMENT,
    counter int,
    yaw FLOAT,
    pitch FLOAT,
    roll FLOAT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);

insert into probe_imu(counter, roll, pitch, yaw) Value(1,0,0,0);



drop table probe_imu_history;

CREATE TABLE probe_imu_history (
    idx INT NOT NULL AUTO_INCREMENT,
    counter int,
    yaw FLOAT,
    pitch FLOAT,
    roll FLOAT,
    updated_at DATETIME(3),
    PRIMARY KEY (idx)
);