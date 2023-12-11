import psycopg2
import os
from dotenv import load_dotenv
import pretty_errors
import openai
import timeit
import logging
import time
from openai.error import ServiceUnavailableError
import pandas as pd
from datetime import datetime, timedelta
from utils.handy import *
from utils.AsyncSummariseJob import async_summarise_job_gpt
from EmbeddingsOpenAI import embeddings_openai
from EmbeddingsE5 import embeddings_e5_base_v2_to_df

"""
Env variables
"""

load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
LOCAL_POSTGRE_URL = os.environ.get("LOCAL_POSTGRE_URL")
RENDER_POSTGRE_URL = os.environ.get("RENDER_POSTGRE_URL")
LOGGER_MAIN = os.environ.get("LOGGER_MAIN")
#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def PostgreToEmbeddings(pipeline: str, embedding_model: str):

	# Configure the logger with the custom format
	log_format = '%(asctime)s %(levelname)s: \n%(message)s\n'

	logging.basicConfig(filename=LOGGER_MAIN,
		level=logging.INFO,
		format=log_format)

	#DETERMINING WHICH DB & TABLE TO SEND THE EMBEDDINGS

	DB_URL = test_or_prod(pipeline=pipeline)

	# Check that DB_URL has an assigned valid value
	if DB_URL is None:
		logging.error("Error: URL_DB must be assigned valid values.")
		return

	#Uncomment after first call
	with open(SAVE_PATH + '/max_id.txt', 'r') as f:
		MAX_ID = int(f.read())

	logging.info(f"\n\nStarting PostgreToEmbeddings.\nSelecting new jobs to embed. Starting from ID: {MAX_ID}")

	def fetch_postgre_rows(db_url:str=DB_URL, max_id:int = 0) -> list:
		
		table = "main_jobs"
		
		logging.info(f"Calling fetch_postgre_rows() from {table} table")
		
		
		#Connect to local postgre db
		conn = psycopg2.connect(db_url)

		# Create a cursor object
		cur = conn.cursor()

		# Fetch new data from the table where id is greater than max_id
		cur.execute(f"SELECT id, title, description, location, timestamp FROM {table} WHERE id > {max_id}")
		new_data = cur.fetchall()

		# If new_data is not empty, update max_id with the maximum id from new_data
		if new_data:
			max_id = max(row[0] for row in new_data)

		# Close the database connection
		conn.commit()
		cur.close()
		conn.close()
		
		# Separate the columns into individual lists
		ids = [row[0] for row in new_data]
		titles = [row[1] for row in new_data]
		descriptions = [row[2] for row in new_data]
		locations = [row[3] for row in new_data]
		timestamp = [row[4] for row in new_data]

		return ids, titles, locations, descriptions, timestamp, max_id


	#Getting the rows 

	IDS, titles, locations, descriptions, TIMESTAMPS, max_id = fetch_postgre_rows(max_id=MAX_ID)

	logging.info(f"\nJobs selected for embedding: {len(IDS)}.\Temporal new max_id: {max_id}")


	def rows_to_nested_list(title_list: list, location_list: list, description_list: list) -> list:
		
		#Titles
		formatted_titles = ["#### title: {} ####".format(title) for title in title_list]
		cleaned_titles = [clean_rows(title) for title in formatted_titles]
		#Locations
		formatted_locations = ["#### location: {} ####".format(location) for location in location_list]
		cleaned_locations = [clean_rows(location) for location in formatted_locations]
		#Descriptions
		formatted_descriptions = ["#### description: {} ####".format(description) for description in description_list]
		cleaned_descriptions = [clean_rows(description) for description in formatted_descriptions]

		#NEST THE LISTS
		jobs_info = [[title, location, description] for title, location, description in zip(cleaned_titles, cleaned_locations, cleaned_descriptions)]

		return jobs_info

	jobs_info= rows_to_nested_list(titles, locations, descriptions)


	def raw_descriptions_to_batches(max_tokens: int = 1000, embedding_model: str= embedding_model, print_messages: bool = True) -> list:
		batches = []
		total_tokens = 0
		truncation_counter = 0  # Counter for truncations

		for i in jobs_info:
			text = " ".join(i)  # Join the elements of the list into a single string
			tokens_description = num_tokens(text)
			if tokens_description <= max_tokens:
				batches.append(text)
			else:
				#TRUNCATE IF STRING MORE THAN x TOKENS
				job_truncated = truncated_string(text, model="gpt-3.5-turbo", max_tokens=max_tokens)
				batches.append(job_truncated)
				truncation_counter += 1

			total_tokens += num_tokens(text)  # Update the total tokens by adding the tokens of the current job

		#Get approximate cost for embeddings
		if embedding_model == "openai":
			approximate_cost = round((total_tokens / 1000) * 0.0004, 4)
		elif embedding_model == "e5_base_v2":
			approximate_cost = 0

		average_tokens_per_batch = total_tokens / len(batches)
		content = f"TOTAL NUMBER OF BATCHES: {len(batches)}\n" \
				f"TOTAL NUMBER OF TOKENS: {total_tokens}\n" \
				f"MAX TOKENS PER BATCH: {max_tokens}\n" \
				f"NUMBER OF TRUNCATIONS: {truncation_counter}\n" \
				f"AVERAGE NUMBER OF TOKENS PER BATCH: {average_tokens_per_batch}\n" \
				f"APPROXIMATE COST OF EMBEDDING: ${approximate_cost} USD\n"
		

		logging.info(f"\nRAW BATCHES SPECS: -------\n{content}")

		if print_messages:
			for i, batch in enumerate(batches, start=1):
				print(f"Batch {i}:")
				print("".join(batch))
				print(f"Tokens per batch:", num_tokens(batch))
				print("\n")

			print(content)
		
		return batches


	JOBS_INFO_BATCHES = raw_descriptions_to_batches()

	FORMATTED_E5_QUERY_BATCHES = query_e5_format(JOBS_INFO_BATCHES)


	#Embedding starts
	if embedding_model == "openai":
		embeddings_openai(batches_to_embed= JOBS_INFO_BATCHES, batches_ids=IDS, original_timestamps=TIMESTAMPS, db="parquet", filename="openai_embeddings_summary")
	elif embedding_model == "e5_base_v2":
		df = embeddings_e5_base_v2_to_df(batches_to_embed=FORMATTED_E5_QUERY_BATCHES, jobs_info=JOBS_INFO_BATCHES, batches_ids=IDS, batches_timestamps=TIMESTAMPS)
		try:
			to_embeddings_e5_base_v2(pipeline=pipeline, df=df, db_url=RENDER_POSTGRE_URL)
			#At the end of the script, save max_id to the file
			with open(SAVE_PATH + '/max_id.txt', 'w') as f:
				f.write(str(max_id))
			logging.info(f"PostgreToEmbeddings has finished correctly! Writing the max_id: {max_id}")
		except Exception as e:
			logging.error(f"Exception while sending embeddings to postgre:\n {e}")
			raise Exception


if __name__ == "__main__":
	PostgreToEmbeddings(pipeline="LocalProd", embedding_model="e5_base_v2")
