import spacy
import random
TRAIN_DATA = [('1. Site Address 166 Crompton Road', {'entities': [[16, 33, 'ADD']]}), 
			  ('Property name Land Adjoining Norbury Railway Station', {'entities': [[14, 52, 'ADD']]}), 
			  ('Address line 1 Norbury Avenue', {'entities': [[15, 29, 'ADD']]}), 
			  ('Address line 2 Norbury', {'entities': [[15, 22, 'ADD']]}), 
			  ('Address line 3 | Montbury East', {'entities': [[17, 30, 'ADD']]}), 
			  ('Town/city London', {'entities': [[10, 16, 'ADD']]}), 
			  ('agent details', {'entities':[]}),
			  ('Postcode SW16 3RW', {'entities': [[9, 17, 'ADD']]}), 
			  ('First name Tony', {'entities': [[11, 15, 'PER']]}), 
			  ('Surname Amin', {'entities': [[8, 12, 'PER']]}), 
			  ('Company name Britbuild Properties Ltd', {'entities': [[13, 37, 'CMP']]}), 
			  ('Address line 4 166 Weir Road', {'entities': [[15, 28, 'ADD']]}), 
			  ('Address line 2 Downtown Avenue', {'entities': [[15, 30, 'ADD']]}), 
			  ('Address line 3 | Baystreet', {'entities': [[17, 26, 'ADD']]}), 
			  ('Town/city London', {'entities': [[10, 16, 'ADD']]}), 
			  ('Country United Kingdom', {'entities': [[8, 22, 'ADD']]}), 
			  ('Postcode XC87 4YV', {'entities': [[9, 17, 'ADD']]}), 
			  ('First name Ellen', {'entities': [[11, 16, 'PER']]}), 
			  ('site details', {'entities':[]}),
			  ('Surname Creegan', {'entities': [[8, 15, 'PER']]}), 
			  ('Company name Iceni Projects', {'entities': [[13, 27, 'CMP']]}), 
			  ('Address line 1 This is the Space', {'entities': [[15, 32, 'ADD']]}), 
			  ('Address line 2 68 Quay Street', {'entities': [[15, 29, 'ADD']]}), 
			  ('Address line 3 Abbyy Northridge', {'entities': [[15, 31, 'ADD']]}), 
			  ('Town/city Manchester', {'entities': [[10, 20, 'ADD']]}), 
			  ('Country UK', {'entities': [[8, 10, 'ADD']]}), 
			  ('Postcode B4 7FG', {'entities': [[9,15, 'ADD']]}), 
			  ('material details', {'entities':[]}),
			  ('random det', {'entities':[]}),
			  ('Description of proposed materials and finishes: Stock brickwork', {'entities': [[48, 63, 'MAT']]}), 
			  ('Description of proposed materials and finishes: Zinc cladding', {'entities': [[48, 61, 'MAT']]}), 
			  ('Description of proposed materials and finishes: Aluminium clad timber glazed windows', {'entities': [[48, 84, 'MAT']]}), 
			  ('Description of proposed materials and finishes: Aluminium clad timber doors', {'entities': [[48, 75, 'MAT']]}),
			  ('Planning Portal Reference: PP-09239967', {'entities':[]}),
			  ('Postcode UI16 3RW', {'entities': [[9, 17, 'ADD']]}),
			  ('Planning Portal Reference: PP-09234567', {'entities':[]}),
			  ('Planning Portal Reference: PP-06739967', {'entities':[]}),
			  ('Planning Portal Reference: PP-09239867', {'entities':[]}),
			  ('Planning Portal Reference: PP-09231267', {'entities':[]}),
			  ('Post M3 3EJ', {'entities': [[5, 11, 'ADD']]}),
			  ]

def train_spacy(data,iterations):
	TRAIN_DATA = data
	nlp = spacy.blank('en')  # create blank Language class
	# create the built-in pipeline components and add them to the pipeline
	# nlp.create_pipe works for built-ins that are registered with spaCy
	if 'ner' not in nlp.pipe_names:
		ner = nlp.create_pipe('ner')
		nlp.add_pipe(ner, last=True)


	# add labels
	for _, annotations in TRAIN_DATA:
		for ent in annotations.get('entities'):
			ner.add_label(ent[2])

	# get names of other pipes to disable them during training
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
	with nlp.disable_pipes(*other_pipes):  # only train NER
		optimizer = nlp.begin_training()
		for itn in range(iterations):
			print("Statring iteration " + str(itn))
			random.shuffle(TRAIN_DATA)
			losses = {}
			for text, annotations in TRAIN_DATA:
				nlp.update(
					[text],  # batch of texts
					[annotations],  # batch of annotations
					drop=0.2,  # dropout - make it harder to memorise data
					sgd=optimizer,  # callable to update weights
					losses=losses)
			print(losses)
	return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
modelfile = 'en_first_1.0'
prdnlp.to_disk(modelfile)

#Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
	print(ent.text, ent.start_char, ent.end_char, ent.label_)