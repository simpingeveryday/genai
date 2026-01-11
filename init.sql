CREATE DATABASE IF NOT EXISTS bank_loan_system;
USE bank_loan_system;

CREATE TABLE IF NOT EXISTS credit_scores (
    customer_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    credit_score INT
);

CREATE TABLE IF NOT EXISTS account_statuses (
    customer_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    nationality VARCHAR(50),
    email VARCHAR(100),
    account_status VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS pr_statuses (
    customer_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    pr_status BOOLEAN
);