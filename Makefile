run_landmark_detector: 
	python -m src.landmark
	
clean:
	find ./ -type f -name "*.py[co]" -delete
	find ./ -type d -name "__pycache__" -delete
