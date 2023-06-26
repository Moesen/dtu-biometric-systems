clean:
	find ./ -type f -name "*.py[co]" -delete
	find ./ -type d -name "__pycache__" -delete

pts2txt:
	@if [ ! -d "./data/SCUT-FBP5500_v2/facial_landmark_txt/" ]; then mkdir ./data/SCUT-FBP5500_v2/facial_landmark_txt; fi
	@python3 -m src.data.scut -m "conversion" -p "./data/SCUT-FBP5500_v2/facial landmark/" -o "./data/SCUT-FBP5500_v2/facial_landmark_txt"

datatest:
	@python3 -m src.data.scut -m "datatest" -p "./data/SCUT-FBP5500_v2/facial_landmark_txt/" -i "./data/SCUT-FBP5500_v2/Images/" -f  "./data/SCUT-FBP5500_v2/train_test_files/All_labels.txt"
	
