import pytesseract
from pytesseract import Output
import os
import pandas
import numpy as np


if __name__ == '__main__':
	"""this is the next step in our document processing pipeline.
	here we convert each page to a dataframe containing all the metadata along with the text from the image.
	This does detection as well as recognition together.
	"""    

	#* first we flush the images folder
	for the_file in os.listdir(os.path.join('.', 'images')):
		file_path = os.path.join('.', 'images', the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			# elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)


	#* creating a placeholder dataframe
	df = pandas.DataFrame(columns=['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
		'left', 'top', 'width', 'height', 'conf', 'text']) 

	#* iterating over the images to extract metadata using pytesseract
	for ind, file in enumerate(os.listdir('.\\images\\')):
		temp = pytesseract.image_to_data(os.path.join(f'.\\images\\{file}'), output_type=Output.DATAFRAME)
		temp.page_num = ind
		df = df.append(temp, ignore_index=True, sort=False)

	#* the dataframe contains some junk rows as well, here we filter those out
	df['text'].replace(['', ' '], np.nan, inplace=True)
	df = df.loc[~df['text'].isna()]

	#* saving the dataframe for inspection as well as later use down the pipeline
	df.to_excel('op.xlsx')