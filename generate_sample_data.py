"""Generate realistic dirty test datasets for CSV, Excel, and JSON."""
import os
import random
import pandas as pd
import numpy as np

os.makedirs('/home/muhammad-zain/generic_etl_pipeline/data/samples', exist_ok=True)

# 1. Pakistan Customers Dirty CSV
records = [
    {
        "cust_id": "CUST-1001",
        "full_name": "  Muhammad Zain  ",
        "mob_number": "03001234567",
        "user_email": "zain.dev@example.com",
        "patient_age": 28,
        "d_o_b": "1996-05-14",
        "salary_pkr": "PKR 145,000",
        "nic_number": "35201-1234567-1",
        "shipping_address": "House 12, Street 4, Phase 5 DHA, Lahore",
        "city_name": "Lahore",
        "is_active": "yes",
        "registered_datetime": "2024/01/10 10:15:30",
        "notes": "Premium enterprise customer with priority support.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1002",
        "full_name": "Ali Hassan",
        "mob_number": "+92 321 9876543",
        "user_email": "ali.hassan@domain.pk",
        "patient_age": 34,
        "d_o_b": "15/08/1990",
        "salary_pkr": "Rs. 95000",
        "nic_number": "3520298765431",
        "shipping_address": "Flat 3B, Sector F-8/3, Islamabad",
        "city_name": "Islamabad",
        "is_active": "True",
        "registered_datetime": "2023-11-20T14:45:00",
        "notes": "Requested SMS notification for all order dispatches.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1003",
        "full_name": "Fatima Noor",
        "mob_number": "00923335551234",
        "user_email": "fatima.noor@pkmail.org",
        "patient_age": 22,
        "d_o_b": "2002-12-01",
        "salary_pkr": "PKR 65,000",
        "nic_number": "42101-5544332-9",
        "shipping_address": "Plot 45, Block 6, PECHS, Karachi",
        "city_name": "Karachi",
        "is_active": "1",
        "registered_datetime": "12-05-2023 09:30:00",
        "notes": "Verified buyer on e-commerce portal.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1004",
        "full_name": "Usman Tariq",
        "mob_number": "3451122334",  # Missing 0 or 92
        "user_email": "usman.t@tech.com",
        "patient_age": 145, # Outlier age
        "d_o_b": "1978/04/22",
        "salary_pkr": "$2,500",
        "nic_number": "37405-1122334-5",
        "shipping_address": "Street 9, Satellite Town, Rawalpindi",
        "city_name": "Rawalpindi",
        "is_active": "active",
        "registered_datetime": "2024-02-01",
        "notes": "Regular high volume transactions recorded.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1005",
        "full_name": "Ayesha Khan",
        "mob_number": "0312-7788990",
        "user_email": None, # Missing email
        "patient_age": None, # Missing age
        "d_o_b": None,
        "salary_pkr": "120,000.00",
        "nic_number": None,
        "shipping_address": "House 99, Gulberg III, Lahore",
        "city_name": "Lahore",
        "is_active": "no",
        "registered_datetime": "2024-03-15 18:20:10",
        "notes": "Customer opted out of marketing emails.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1006",
        "full_name": "Bilal Ahmed",
        "mob_number": "92-301-4455667",
        "user_email": "bilal.ahmed@corporate.com",
        "patient_age": 41,
        "d_o_b": "1983-09-17",
        "salary_pkr": "PKR 250,000",
        "nic_number": "35201-9988776-3",
        "shipping_address": "Office 402, Eden Tower, Main Boulevard, Lahore",
        "city_name": "Lahore",
        "is_active": "true",
        "registered_datetime": "2023/12/30",
        "notes": "Corporate bulk order discount applies.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1007",
        "full_name": "Zubair Shah",
        "mob_number": None, # Missing phone
        "user_email": "zubair.shah@yahoo.com",
        "patient_age": 31,
        "d_o_b": "1993-01-25",
        "salary_pkr": "Rs. 82,000",
        "nic_number": "17301-3322114-7",
        "shipping_address": "University Town, Peshawar",
        "city_name": "Peshawar",
        "is_active": "0",
        "registered_datetime": "2024-01-28 11:00:00",
        "notes": "Account under periodic compliance review.",
        "mostly_empty_col": None
    },
    {
        "cust_id": "CUST-1008",
        "full_name": "Sana Sheikh",
        "mob_number": "0333 4445566",
        "user_email": "sana.s@consulting.pk",
        "patient_age": -10, # Invalid negative age
        "d_o_b": "1998-07-11",
        "salary_pkr": "PKR 110,000",
        "nic_number": "61101-8877665-2",
        "shipping_address": "Sector G-11/2, Islamabad",
        "city_name": "Islamabad",
        "is_active": "yes",
        "registered_datetime": "2024-02-14 16:40:00",
        "notes": "New registration via mobile app.",
        "mostly_empty_col": None
    },
    # Duplicate row for testing deduplicator
    {
        "cust_id": "CUST-1001",
        "full_name": "  Muhammad Zain  ",
        "mob_number": "03001234567",
        "user_email": "zain.dev@example.com",
        "patient_age": 28,
        "d_o_b": "1996-05-14",
        "salary_pkr": "PKR 145,000",
        "nic_number": "35201-1234567-1",
        "shipping_address": "House 12, Street 4, Phase 5 DHA, Lahore",
        "city_name": "Lahore",
        "is_active": "yes",
        "registered_datetime": "2024/01/10 10:15:30",
        "notes": "Premium enterprise customer with priority support.",
        "mostly_empty_col": None
    }
]

df_csv = pd.DataFrame(records)
df_csv.to_csv('/home/muhammad-zain/generic_etl_pipeline/data/samples/pakistan_customers_dirty.csv', index=False)

# 2. Sales Transactions Dirty Excel
sales_records = [
    {
        "transaction_id": f"TXN-{10000 + i}",
        "customer_phone": random.choice(["03001122334", "03219988776", "03335544332", "+92 345 6677889", "0312-3344556", None]),
        "customer_city": random.choice(["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", None]),
        "item_price": random.choice(["PKR 2,499.00", "Rs. 1500", "$45.00", "3200", "PKR 12,500.50"]),
        "order_qty": random.choice([1, 2, 5, 10, None, 150]), # 150 is outlier
        "discount_pct": random.choice(["5%", "10%", "15%", "0%", None]),
        "order_timestamp": random.choice(["2024-01-05 12:30:00", "05/01/2024", "2024/02/10", "1705300000"]),
        "is_fulfilled": random.choice(["yes", "no", "true", "false", "1", "0"])
    }
    for i in range(50)
]
df_sales = pd.DataFrame(sales_records)
try:
    df_sales.to_excel('/home/muhammad-zain/generic_etl_pipeline/data/samples/sales_inventory_dirty.xlsx', index=False)
except Exception:
    df_sales.to_csv('/home/muhammad-zain/generic_etl_pipeline/data/samples/sales_inventory_dirty.csv', index=False)

# 3. IoT Telemetry JSON
import json
telemetry_data = [
    {
        "device_id": f"DEV-NODE-{i:03d}",
        "client_ip": f"192.168.1.{10 + i}",
        "temperature_c": round(random.uniform(18.5, 95.0), 2),
        "humidity_pct": f"{random.randint(30, 90)}%",
        "reported_at": f"2024-03-{random.randint(1, 28):02d}T10:00:00Z",
        "emergency_contact": "03001234567" if i % 2 == 0 else "+92 321 5566778",
        "status": random.choice(["ONLINE", "OFFLINE", "DEGRADED", None])
    }
    for i in range(30)
]
with open('/home/muhammad-zain/generic_etl_pipeline/data/samples/iot_telemetry_corrupted.json', 'w') as f:
    json.dump(telemetry_data, f, indent=2)

print("Generated sample test files successfully!")
