import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import random

cred = credentials.Certificate('serviceAccountKey.json')

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def expected_orders(date):
    base = 40
    if date.weekday() == 4:  base += 25
    if date.weekday() == 5:  base += 55
    if date.weekday() == 6:  base += 45
    if date.month in [3, 4]: base += 20
    if date.month == 12:     base += 30
    if date.month in [1, 2]: base += 15
    return base + random.randint(-8, 8)

start_date = datetime.date.today() - datetime.timedelta(days=90)
orders_ref = db.collection('orders')
total = 0

for day_offset in range(90):
    order_date = start_date + datetime.timedelta(days=day_offset)
    num_orders = expected_orders(order_date)
    for i in range(num_orders):
        hour = random.choices(
            [8, 9, 10, 12, 13, 14, 18, 19, 20, 21],
            weights=[5, 8, 5, 20, 25, 15, 10, 15, 12, 5]
        )[0]
        minute = random.randint(0, 59)
        order_time = datetime.datetime.combine(order_date, datetime.time(hour, minute))
        orders_ref.add({
            'createdAt': order_time,
            'status': 'Delivered',
            'orderType': random.choice(['Dine-In', 'Takeaway', 'Delivery']),
            'totalAmount': round(random.uniform(8.0, 85.0), 2),
            'customerId': f'seed_customer_{random.randint(1, 50)}',
        })
        total += 1
    print(f'{order_date} ({order_date.strftime("%A")}): {num_orders} orders')

print(f'\nDone. {total} total orders inserted.')