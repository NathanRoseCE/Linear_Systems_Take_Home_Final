# Makefile to build the results and cache certain results I dont usually write makefile directly,
# so there are probably several bad practices in here :)

Utilities = scripts/LatexFormat.py scripts/Model.py scripts/Utilities.py
py = python3
latex = pdflatex -shell-escape

all : results/question-one results/question-two results/question-three Final.pdf

results/question-one : scripts/questionOne.py scripts/resources/one.json $(Utilities) 
	$(py) scripts/questionOne.py
	touch results/question-one

results/question-two : scripts/questionTwo.py scripts/resources/two.json $(Utilities)
	$(py) scripts/questionTwo.py
	touch results/question-two

results/question-three : scripts/questionThree.py scripts/NonLinearFragment.py scripts/resources/three.json $(Utilities)
	$(py) scripts/questionThree.py
	touch results/question-three

Final.pdf : results/question-one results/question-two results/question-three ./*.tex
	$(latex) Final.tex

clean :
	rm -rf results/*
	rm -rf __cache__/
