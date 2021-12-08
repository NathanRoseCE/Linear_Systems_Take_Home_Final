# Makefile to build the results and cache certain results I dont usually write makefile directly,
# so there are probably several bad practices in here :)

Utilities = LatexFormat.py Model.py Utilities.py
py = python3

all : results/question-one results/question-two results/question-three

results/question-one : questionOne.py NonLinearFragment.py resources/one.json $(Utilities) 
	$(py) questionOne.py
	touch results/question-one

results/question-two : questionTwo.py resources/two.json $(Utilities)
	$(py) questionTwo.py
	touch results/question-two

results/question-three : questionThree.py resources/three.json $(Utilities)
	$(py) questionThree.py
	touch results/question-three

clean :
	rm -rf results/*
	rm -rf __cache__/
