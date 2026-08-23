from sqlalchemy import URL


class Config:
    SQLALCHEMY_DATABASE_URI = URL.create(
        drivername="mysql+pymysql",
        username="root",
        password="Abhilasha@123",
        host="127.0.0.1",
        port=3306,
        database="ecommerce_analytics"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False