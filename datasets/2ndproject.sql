create database hairsalon;

use hairsalon;

show variables like 'secure_file_priv';
show variables like '%dir';

create table client_cancellations (
	CancellationID INT AUTO_INCREMENT PRIMARY KEY,
    CancelDate DATE NOT NULL,
    ClientCode VARCHAR(20) NOT NULL,
    ServiceCode VARCHAR(20) NOT NULL,
    Staff VARCHAR(50) NOT NULL,
    BookingDate DATE NOT NULL,
    CanceledBy VARCHAR(50) NOT NULL,
    Days INT
);

LOAD DATA INFILE 'C:/Workspace/Python/archive/Client Cancellations0.csv'
INTO TABLE Client_Cancellations
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(CancelDate, ClientCode, ServiceCode, Staff, BookingDate, CanceledBy, Days);

CREATE TABLE FutureBookings (
    BookingID INT AUTO_INCREMENT PRIMARY KEY,
    ClientCode VARCHAR(20),
    StaffCode VARCHAR(20),
    ServiceCode VARCHAR(20),
    BookingDate DATE,
    BookingTime TIME,
    TimeInt INT
);

LOAD DATA INFILE 'C:/workspace/Python/archive/Future Bookings (All Clients)0.csv'
INTO TABLE FutureBookings
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(ClientCode, StaffCode, ServiceCode, @BookingDate, @BookingTime, TimeInt)
SET
    BookingDate = STR_TO_DATE(@BookingDate, '%m/%d/%Y'),
    BookingTime = STR_TO_DATE(@BookingTime, '%h:%i:%s %p');
    

