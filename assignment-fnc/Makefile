install:
	conda create --name fnc python=3.6 && \
	pip install -r requirements.txt

copy:
	cp ~/Downloads/Learning\ to\ predict\ the\ stace\ of\ news\ articles.pdf report.pdf

zip: copy
	rm -f files.zip && \
	zip -r files.zip logs sbin *.py *.txt fnc_challenge_utils data/placeholder Makefile README.md --exclude *.pyc

submit:
	give cs9417 assignment files.zip report.pdf

