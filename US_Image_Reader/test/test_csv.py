import csv
import re
import os
import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'D:/Work/Hackathon/High School Hackathon 2025/Student Score.csv'

data = []

try:
    # Open the CSV file for reading with UTF-8 encoding
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        # Create a CSV DictReader object
        csv_reader = csv.DictReader(file)

        # Iterate over each row in the CSV file
        for row_number, row in enumerate(csv_reader, start=1):

            print(f"Record {row_number}:")
            print(row)

            directory = f'./{row["grade"]}'
            os.makedirs(directory, exist_ok=True)

            data = [[row["first_name"]]] #

            output_file_path = f"{directory}/{row_number}.txt"
            with open(output_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(data)


            if row_number > 3:
                break


except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except UnicodeDecodeError as e:
    print(f"A Unicode decoding error occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

