# Makefile to build the results and cache certain results I dont usually write makefile directly,
# so there are probably several bad practices in here :)

Utilities = scripts/LatexFormat.py scripts/Model.py scripts/Utilities.py
texFiles = ./content.tex ./functions.tex
py = python3
latex = pdflatex -shell-escape

all : results/question-one results/question-two results/question-three Final.pdf

results/question-one : scripts/questionOne.py resources/one.json $(Utilities) 
	$(py) scripts/questionOne.py
	touch results/question-one

results/question-two : scripts/questionTwo.py resources/two.json $(Utilities)
	$(py) scripts/questionTwo.py
	touch results/question-two

results/question-three : scripts/questionThree.py scripts/NonLinearFragment.py resources/three.json $(Utilities)
	$(py) scripts/questionThree.py
	touch results/question-three

#latex has to be run twice because its fun like that
Final.pdf : results/question-one results/question-two results/question-three $(texFiles)
	$(latex) Final.tex
	$(latex) Final.tex

clean :
	rm -rf results/*
	rm -rf __cache__/
	rm -f *.log
	rm -f *.out
	rm -f *.aux
