import pandas as pd
import numpy as np
import boto3
import json
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

load_dotenv()

#credieantials for AWS
access_key=os.getenv('access_key')
secret_key=os.getenv('secret_key')

#pushing data to AWS
s3=boto3.client('s3',aws_access_key_id=access_key,aws_secret_access_key=secret_key)
file_path={'file1.json':'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/raw data/file1.json',
           'file2.json':'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/raw data/file2.json',
           'file3.json':'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/raw data/file3.json',
           'file4.json':'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/raw data/file4.json',
           'file5.json':'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/raw data/file5.json',}

for file,path in file_path.items():
    s3.upload_file(path,'srvprojects','Final_project/'+file)
    

#converting data from json to csv 
def json_to_csv(json_data, output_csv_path):
    try:
        # Parse JSON data
        data = json.loads(json_data)

        # Extract restaurant details
        restaurants = []
        for entry in data:
            if 'restaurants' in entry and entry['restaurants']:
                for restaurant_entry in entry['restaurants']:
                    restaurant = restaurant_entry['restaurant']
                    restaurants.append({
                        'res_id': restaurant.get('R', {}).get('res_id'),
                        'name': restaurant.get('name'),
                        'cuisines': restaurant.get('cuisines'),
                        'average_cost_for_two': restaurant.get('average_cost_for_two'),
                        'currency': restaurant.get('currency'),
                        'price_range': restaurant.get('price_range'),
                        'location': restaurant.get('location', {}).get('address'),
                        'city': restaurant.get('location', {}).get('city'),
                        'latitude': restaurant.get('location', {}).get('latitude'),
                        'longitude': restaurant.get('location', {}).get('longitude'),
                        'rating': restaurant.get('user_rating', {}).get('aggregate_rating'),
                        'votes': restaurant.get('user_rating', {}).get('votes'),
                        'rating_text': restaurant.get('user_rating', {}).get('rating_text'),
                        'rating_color': restaurant.get('user_rating', {}).get('rating_color'),
                        'has_online_delivery': restaurant.get('has_online_delivery'),
                        'has_table_booking': restaurant.get('has_table_booking'),
                        'is_delivering_now': restaurant.get('is_delivering_now'),
                        'country_id': restaurant.get('location', {}).get('country_id'),
                        'locality': restaurant.get('location', {}).get('locality'),
                        'zipcode': restaurant.get('location', {}).get('zipcode'),
                        'photos_url': restaurant.get('photos_url'),
                        'menu_url': restaurant.get('menu_url'),
                    })

        # Convert to a Pandas DataFrame
        df = pd.DataFrame(restaurants)

        # Save DataFrame to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Data successfully saved to {output_csv_path}")

    except Exception as e:
        print(f"An error occurred while converting JSON to CSV: {e}")

#fetching data from AWS S3 server

# AWS S3 bucket and file details
bucket_name = 'srvprojects'
keys = [
    'Final_project/file1.json',
    'Final_project/file2.json',
    'Final_project/file3.json',
    'Final_project/file4.json',
    'Final_project/file5.json'
]

output_dir = 'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/converted_csv_data/'

# Fetch and process each file from S3
for key in keys:
    try:
        # Fetch JSON file from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        json_data = response['Body'].read().decode('utf-8')

        # Generate the local output CSV file path
        file_name = key.split('/')[-1].replace('.json', '.csv')
        output_csv_path = f"{output_dir}{file_name}"

        # Convert JSON to CSV
        json_to_csv(json_data, output_csv_path)

    except Exception as e:
        print(f"An error occurred while processing {key}: {e}")
    

#Establish DB connection AND create DATABASE

# Database connection details for the server (not the specific database)
server_config = {
    'host': 'myfinalproject.cls088i2o3dw.ap-south-1.rds.amazonaws.com',  # RDS endpoint or local host
    'port': 5432,
    'user': 'postgres',
    'password': 'admin123'
}

def create_database(db_name):
    conn = None
    cursor = None
    try:
        # Connect to the PostgreSQL server 
        conn = psycopg2.connect(**server_config)
        conn.autocommit = True  
        cursor = conn.cursor()

        # Create the database
        create_db_query = sql.SQL("CREATE DATABASE {db_name}").format(db_name=sql.Identifier(db_name))
        cursor.execute(create_db_query)
        print(f"Database '{db_name}' created successfully!")

    except psycopg2.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Usage
dbname='finalproject2db'
create_database(dbname)

#pushing all csv to RDS

# Database connection details
db_config = {
    'host': 'myfinalproject.cls088i2o3dw.ap-south-1.rds.amazonaws.com', 
    'port': 5432,
    'user': 'postgres',  
    'password': 'admin123',  
    'database': dbname  
}

def upload_csv_to_rds(csv_path, table_name):
    """
    Uploads a CSV file to a PostgreSQL RDS database table.

    Parameters:
        csv_path (str): Path to the CSV file.
        table_name (str): Name of the table to insert data into.
    """
    conn = None
    cursor = None
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(csv_path)

        # Establish database connection
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Create table (if not exists)
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (res_id BIGINT,name VARCHAR(255),cuisines TEXT,average_cost_for_two INT,
            currency VARCHAR(10),price_range INT,location TEXT,city VARCHAR(100),latitude FLOAT,longitude FLOAT,rating FLOAT,
            votes INT,rating_text VARCHAR(50),rating_color VARCHAR(20),has_online_delivery INT,has_table_booking INT,
            is_delivering_now INT,country_id INT,locality VARCHAR(255),zipcode VARCHAR(20),photos_url TEXT,menu_url TEXT);"""
            
        cursor.execute(create_table_query)

        # Insert DataFrame into PostgreSQL
        for _, row in df.iterrows():
            insert_query = f"""
            INSERT INTO {table_name} (res_id, name, cuisines, average_cost_for_two, currency, price_range, location,
                city, latitude, longitude, rating, votes, rating_text, rating_color,has_online_delivery, has_table_booking, is_delivering_now, country_id,
                locality, zipcode, photos_url, menu_url) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
                
            cursor.execute(insert_query, (
                row['res_id'], row['name'], row['cuisines'], row['average_cost_for_two'], 
                row['currency'], row['price_range'], row['location'], row['city'], 
                row['latitude'], row['longitude'], row['rating'], row['votes'], 
                row['rating_text'], row['rating_color'], row['has_online_delivery'], 
                row['has_table_booking'], row['is_delivering_now'], row['country_id'], 
                row['locality'], row['zipcode'], row['photos_url'], row['menu_url']
            ))

        # Commit the transaction
        conn.commit()
        print(f"Data from {csv_path} uploaded to table {table_name} successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Ensure cursor and connection are closed properly
        if cursor:
            cursor.close()
        if conn:
            conn.close()



csv_files = [
    'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/converted_csv_data/file1.csv',
    'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/converted_csv_data/file2.csv',
    'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/converted_csv_data/file3.csv',
    'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/converted_csv_data/file4.csv',
    'C:/Users/srvik/Desktop/github/ChefMate Restaurant Clustering & Cooking Guide Application/converted_csv_data/file5.csv'
]
for csv_file in csv_files:
    upload_csv_to_rds(csv_file, 'restaurants_data')


#fetching csv file from RDS

# Create SQLAlchemy engine
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

# Fetch data from the database
def fetch_data(table_name):
    try:
        engine = get_engine()
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
combined_df = fetch_data('restaurants_data')

#cleaning the csv data

print(combined_df.isnull().sum())
combined_df=combined_df.drop(columns=['zipcode'])
combined_df=combined_df.dropna()
print(combined_df.isnull().sum())
print(combined_df.dtypes)

#saving the cleaned data to csv file
combined_df.to_csv('clean_data.csv')




