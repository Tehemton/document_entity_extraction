from pdf2image import convert_from_path


if __name__ == '__main__':
	"""this is the first step in document digitization and tag extraction
	since our documents are going to be varying between image based text documents and searchable text documents,
	we convert all pdfs to images.
	This achieves the following:
	1) streamlines data extraction pipeline
	2) allows us to leverage powerful imageprocessing models and libraries
	3) allows us to manipulate, sort and group data based on their relative location to each other
	4) we can also fetch rich metadata from images for each datapoint and store those for integrating into a larger framework
	5) for cases where we have scanned pdfs, we have the opton to enhance the image using morphological filtering, sharpening, deblurring etc before processing
	"""    
	filename = 'Assignment-1.pdf'

	pages = convert_from_path(pdf_path=filename,  dpi = 400, fmt='jpeg', use_pdftocairo=True, thread_count=3 )
	for ind, page in enumerate(pages):
		page.save(f'.\\images\\{ind}.jpeg', 'JPEG')
	
