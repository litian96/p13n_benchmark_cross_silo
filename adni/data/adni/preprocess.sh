# put raw data under $raw_data_path folder, label file under $label_file_location, meta data under $meta_data_path
# specify $png_path to put preprocessed, rescaled png images

python3 generate_label.py $raw_data_path $label_file_location
python3 preprocess.py $meta_data_path $raw_data_path $png_path
python3 generate_json.py $png_path