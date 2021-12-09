# Makefile to build the results and cache certain results I dont usually write makefile directly,
# so there are probably several bad practices in here :)

Utilities = scripts/LatexFormat.py scripts/Model.py scripts/Utilities.py
texFiles = ./content.tex ./functions.tex
q_one_template = ./resources/templates/one_*.tex.j2
q_two_template = ./resources/templates/two_*.tex.j2
q_three_template = ./resources/templates/three_*.tex.j2
py = python3
latex = pdflatex -shell-escape

all : results/one.tex results/two.tex results/three.tex Final.pdf

results/one.tex : scripts/questionOne.py resources/one.json $(Utilities) $(q_one_template)
	$(py) scripts/questionOne.py
	touch results/question-one

results/two.tex : scripts/questionTwo.py resources/two.json $(Utilities) $(q_two_template)
	$(py) scripts/questionTwo.py
	touch results/question-two

results/three.tex : scripts/questionThree.py scripts/NonLinearFragment.py resources/three.json $(Utilities) $(q_three_template)
	$(py) scripts/questionThree.py
	touch results/question-three

#latex has to be run twice because its fun like that
Final.pdf : results/one.tex results/two.tex results/three.tex $(texFiles)
	$(latex) Final.tex
	$(latex) Final.tex

clean :
	rm -rf results/*
	rm -rf __cache__/
	rm -f *.log
	rm -f *.out
	rm -f *.aux
