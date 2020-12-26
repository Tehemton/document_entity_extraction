****accuracy of the NER will improve as we get more training data and examples to model over.
for now the model was trained over a handful of datapoints from this document****


This application has been modelled to have independent modules inside it so as to make it more feasible to plug into a document processing pipeline.
A distributed loosely coupled architecture helps to make things more flexible and modular.
This can easily be plugged into flask REST apis and develop a callback architecture for asynchronous data provessing and parallelization over kubernetes

these modules namely are:
1) pdf to image conversion (read_pdf)
2) image to dataframe using tesseract (tesseract_img_to_data)
3) extracting the data using nlp and rule based approaches (extract_data_nlp)