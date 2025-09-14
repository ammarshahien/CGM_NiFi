# CGM Data Pipeline

A real-time Continuous Glucose Monitoring data processing pipeline using Apache NiFi and MiNiFi.

## Architecture

- **Data Generator**: Python script simulating CGM device data
- **MiNiFi Edge Agent**: Collects and forwards data from devices
- **Apache NiFi**: Central data processing with validation, alerting, and database storage

## Components

- Data Ingestion & Rate Limiting
- Data Validation & Quality Checks
- Alert Management (Warning & Critical alerts)
- Database Operations (SQL Server)
- Email Notifications

## Setup

1. Install Apache NiFi and MiNiFi
2. Import the flow templates
3. Configure database connection
4. Set up email credentials
5. Run data generator
