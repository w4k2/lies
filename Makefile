all:
	python seek.py > log.txt
	python plot_collisions.py

compute:
	python process_2x5cv.py
	python process_kcv.py
