import pyodbc

try:
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost\\SQLEXPRESS;'
        'DATABASE=ZAMANBANK;'
        'Trusted_Connection=yes;'
    )
    print("✅ Подключение успешно!")
except Exception as e:
    print("❌ Ошибка:", e)
